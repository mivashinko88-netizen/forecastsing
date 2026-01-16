# routers/models.py - Model and Prediction endpoints
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
import json

from database import get_db
from db_models import User, Business, TrainedModel, Prediction
from auth import get_current_user
from config import BusinessConfig
from ai.trainer import SalesForecaster
from services.weather import get_forecast_weather
from services.events import get_holidays
from services.sports import get_nfl_games
from services.cycles import get_payday_dates
from services.school import get_school_calendar

router = APIRouter(tags=["Models"])


class PredictRequest(BaseModel):
    days: int = 7
    items: Optional[List[str]] = None


class PredictionResponse(BaseModel):
    date: str
    item_name: str
    predicted_quantity: int


# ============ Model Endpoints ============

@router.get("/businesses/{business_id}/models")
async def get_models(
    business_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all trained models for a business"""
    # Verify business belongs to user
    business = db.query(Business).filter(
        Business.id == business_id,
        Business.user_id == current_user.id
    ).first()

    if not business:
        raise HTTPException(status_code=404, detail="Business not found")

    models = db.query(TrainedModel).filter(
        TrainedModel.business_id == business_id
    ).order_by(TrainedModel.created_at.desc()).all()

    return [
        {
            "id": m.id,
            "business_id": m.business_id,
            "model_type": m.model_name,
            "test_mae": m.test_mae,
            "test_mape": m.test_mape,
            "training_rows": m.training_rows,
            "test_rows": m.test_rows,
            "feature_importance": json.loads(m.feature_importance) if m.feature_importance else {},
            "items_trained": m.items_trained,
            "is_active": m.is_active,
            "created_at": m.created_at.isoformat()
        }
        for m in models
    ]


@router.get("/models/{model_id}")
async def get_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific model"""
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Verify ownership
    business = db.query(Business).filter(
        Business.id == model.business_id,
        Business.user_id == current_user.id
    ).first()

    if not business:
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "id": model.id,
        "business_id": model.business_id,
        "model_type": model.model_name,
        "test_mae": model.test_mae,
        "test_mape": model.test_mape,
        "training_rows": model.training_rows,
        "test_rows": model.test_rows,
        "feature_importance": json.loads(model.feature_importance) if model.feature_importance else {},
        "items_trained": model.items_trained,
        "is_active": model.is_active,
        "created_at": model.created_at.isoformat()
    }


@router.put("/models/{model_id}/activate")
async def activate_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Set a model as the active model for its business"""
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Verify ownership
    business = db.query(Business).filter(
        Business.id == model.business_id,
        Business.user_id == current_user.id
    ).first()

    if not business:
        raise HTTPException(status_code=404, detail="Model not found")

    # Deactivate other models for this business
    db.query(TrainedModel).filter(
        TrainedModel.business_id == model.business_id
    ).update({"is_active": False})

    # Activate this model
    model.is_active = True
    db.commit()

    return {"message": "Model activated"}


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a model"""
    from db_models import Upload
    import os

    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Verify ownership
    business = db.query(Business).filter(
        Business.id == model.business_id,
        Business.user_id == current_user.id
    ).first()

    if not business:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        # Clear model_id from uploads that reference this model
        db.query(Upload).filter(Upload.model_id == model_id).update({"model_id": None})

        # Delete predictions associated with this model
        db.query(Prediction).filter(Prediction.model_id == model_id).delete()

        # Delete the model file from disk if it exists
        if model.model_path and os.path.exists(model.model_path):
            try:
                os.remove(model.model_path)
            except Exception as e:
                print(f"Warning: Could not delete model file: {e}")

        # Delete the model record
        db.delete(model)
        db.commit()

        return {"message": "Model deleted"}
    except Exception as e:
        db.rollback()
        print(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


# ============ Prediction Endpoints ============

@router.post("/models/{model_id}/predict")
async def predict(
    model_id: int,
    request: PredictRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate predictions for the next N days"""
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Verify ownership
    business = db.query(Business).filter(
        Business.id == model.business_id,
        Business.user_id == current_user.id
    ).first()

    if not business:
        raise HTTPException(status_code=404, detail="Model not found")

    # Check if model data exists (either in database or file)
    if not model.model_data and not model.model_path:
        raise HTTPException(status_code=400, detail="Model not trained yet")

    # Load the trained model and generate predictions
    try:
        # Create business config
        config = BusinessConfig(
            business_name=business.name,
            city=business.city,
            state=business.state,
            zipcode=business.zipcode,
            country=business.country or "US",
            timezone=business.timezone or "America/New_York"
        )

        # Get items to predict
        items = request.items
        if not items and model.items_trained:
            items = json.loads(model.items_trained)

        # Generate date range
        start_date = datetime.now().date()
        end_date = start_date + timedelta(days=request.days)
        year = start_date.year

        # Fetch external factors for the forecast period
        weather_data = []
        holidays = []
        sports_games = []
        paydays = []
        school_calendar = []

        if business.latitude and business.longitude:
            try:
                weather_data = await get_forecast_weather(
                    latitude=business.latitude,
                    longitude=business.longitude,
                    days=request.days
                )
            except Exception as e:
                print(f"Weather forecast fetch failed: {e}")

        try:
            holidays = await get_holidays(year, business.country or "US")
        except Exception as e:
            print(f"Holidays fetch failed: {e}")

        try:
            sports_games = await get_nfl_games(year)
        except Exception as e:
            print(f"Sports fetch failed: {e}")

        try:
            payday_dates = get_payday_dates(year)
            # Convert to format expected by trainer
            paydays = [{"date": p.isoformat()} for p in payday_dates]
        except Exception as e:
            print(f"Paydays fetch failed: {e}")
            payday_dates = []

        try:
            school_calendar = get_school_calendar(year)
        except Exception as e:
            print(f"School calendar fetch failed: {e}")

        # Generate list of future dates
        future_dates = []
        current = start_date
        while current <= end_date:
            future_dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        # Load forecaster and generate predictions
        forecaster = SalesForecaster(config)

        # Load model from database (preferred) or file (legacy)
        if model.model_data:
            forecaster.load_model_from_bytes(model.model_data)
        elif model.model_path:
            forecaster.load_model(model.model_path)
        else:
            raise HTTPException(status_code=400, detail="No model data available")

        predictions_df = forecaster.predict(
            future_dates=future_dates,
            items=items or ["All Items"],
            weather_data=weather_data,
            holidays=holidays,
            sports_games=sports_games,
            paydays=paydays,
            school_calendar=school_calendar
        )

        # Convert DataFrame to list of dicts and expand by size
        raw_predictions = predictions_df.to_dict(orient="records")
        predictions = []

        for p in raw_predictions:
            if hasattr(p["date"], "strftime"):
                p["date"] = p["date"].strftime("%Y-%m-%d")

            item_meta = forecaster.item_metadata.get(p["item_name"], {})
            base_qty = p["predicted_quantity"]

            # Check if we have size distribution data
            size_distribution = item_meta.get("size_distribution", {})
            size_prices = item_meta.get("size_prices", {})

            if size_distribution and len(size_distribution) > 1:
                # Split prediction across sizes based on historical distribution
                for size, proportion in size_distribution.items():
                    size_qty = max(1, round(base_qty * proportion))
                    size_price = size_prices.get(size, item_meta.get("unit_price", 0))

                    size_prediction = {
                        "date": p["date"],
                        "item_name": p["item_name"],
                        "predicted_quantity": size_qty,
                        "size": size,
                        "unit_price": round(size_price, 2) if size_price else None,
                        "predicted_revenue": round(size_qty * size_price, 2) if size_price else None,
                        "category": item_meta.get("category"),
                        "available_sizes": item_meta.get("sizes", []),
                        "size_proportion": round(proportion * 100, 1)
                    }
                    predictions.append(size_prediction)
            else:
                # No size distribution, use single prediction with primary size
                p["category"] = item_meta.get("category")
                p["size"] = item_meta.get("primary_size")
                p["available_sizes"] = item_meta.get("sizes", [])
                if "unit_price" in item_meta:
                    p["unit_price"] = round(item_meta["unit_price"], 2)
                    p["predicted_revenue"] = round(base_qty * item_meta["unit_price"], 2)
                predictions.append(p)

        # Build factors in range for UI
        factors_in_range = {
            "holidays": [h for h in holidays if start_date <= datetime.strptime(h["date"], "%Y-%m-%d").date() <= end_date] if holidays else [],
            "sports": [s for s in sports_games if start_date <= datetime.strptime(s["date"], "%Y-%m-%d").date() <= end_date] if sports_games else [],
            "paydays": [{"date": p.isoformat(), "name": "Payday"} for p in payday_dates if start_date <= p <= end_date] if payday_dates else [],
            "weather": weather_data[:5] if weather_data else []
        }

        # Save predictions to database for later comparison with actuals
        # Use raw_predictions (before size expansion) for accurate item-level tracking
        for p in raw_predictions:
            date_str = p["date"] if isinstance(p["date"], str) else p["date"].strftime("%Y-%m-%d")
            pred_date = datetime.strptime(date_str, "%Y-%m-%d").date()

            existing = db.query(Prediction).filter(
                Prediction.model_id == model_id,
                Prediction.prediction_date == pred_date,
                Prediction.item_name == p["item_name"]
            ).first()

            if existing:
                # Update existing prediction
                existing.predicted_quantity = p["predicted_quantity"]
            else:
                # Create new prediction record
                new_pred = Prediction(
                    model_id=model_id,
                    prediction_date=pred_date,
                    item_name=p["item_name"],
                    predicted_quantity=p["predicted_quantity"]
                )
                db.add(new_pred)

        db.commit()
        print(f"Saved {len(raw_predictions)} predictions to database for model {model_id}")

        return {
            "predictions": predictions,
            "factors_in_range": factors_in_range,
            "item_metadata": forecaster.item_metadata,
            "hourly_patterns": forecaster.hourly_patterns,
            "categories": getattr(forecaster, 'categories', {}),
            "model_id": model_id,
            "days": request.days
        }

    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model file not found. Please retrain the model.")
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/businesses/{business_id}/forecasts")
async def get_forecasts(
    business_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get recent forecast predictions for a business"""
    # Verify business belongs to user
    business = db.query(Business).filter(
        Business.id == business_id,
        Business.user_id == current_user.id
    ).first()

    if not business:
        raise HTTPException(status_code=404, detail="Business not found")

    # Get active model
    model = db.query(TrainedModel).filter(
        TrainedModel.business_id == business_id,
        TrainedModel.is_active == True
    ).first()

    if not model:
        return {"predictions": [], "model": None}

    # Get recent predictions
    predictions = db.query(Prediction).filter(
        Prediction.model_id == model.id,
        Prediction.prediction_date >= datetime.now().date()
    ).order_by(Prediction.prediction_date).all()

    return {
        "predictions": [
            {
                "id": p.id,
                "date": p.prediction_date.isoformat(),
                "item_name": p.item_name,
                "predicted_quantity": p.predicted_quantity,
                "actual_quantity": p.actual_quantity
            }
            for p in predictions
        ],
        "model": {
            "id": model.id,
            "test_mape": model.test_mape,
            "created_at": model.created_at.isoformat()
        }
    }


# ============ Actual vs Predicted Tracking ============

class ActualSalesEntry(BaseModel):
    date: str
    item_name: str
    actual_quantity: int


class BulkActualSales(BaseModel):
    entries: List[ActualSalesEntry]


@router.post("/businesses/{business_id}/actuals")
async def save_actual_sales(
    business_id: int,
    data: BulkActualSales,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Save actual sales data and match with predictions"""
    # Verify business belongs to user
    business = db.query(Business).filter(
        Business.id == business_id,
        Business.user_id == current_user.id
    ).first()

    if not business:
        raise HTTPException(status_code=404, detail="Business not found")

    # Get active model
    model = db.query(TrainedModel).filter(
        TrainedModel.business_id == business_id,
        TrainedModel.is_active == True
    ).first()

    if not model:
        raise HTTPException(status_code=400, detail="No active model found")

    updated_count = 0
    created_count = 0

    for entry in data.entries:
        entry_date = datetime.strptime(entry.date, "%Y-%m-%d").date()

        # Try to find existing prediction
        prediction = db.query(Prediction).filter(
            Prediction.model_id == model.id,
            Prediction.prediction_date == entry_date,
            Prediction.item_name == entry.item_name
        ).first()

        if prediction:
            prediction.actual_quantity = entry.actual_quantity
            updated_count += 1
        else:
            # Create a new record for tracking even without prediction
            new_pred = Prediction(
                model_id=model.id,
                prediction_date=entry_date,
                item_name=entry.item_name,
                predicted_quantity=0,
                actual_quantity=entry.actual_quantity
            )
            db.add(new_pred)
            created_count += 1

    db.commit()

    return {
        "message": "Actual sales saved",
        "updated": updated_count,
        "created": created_count
    }


@router.get("/businesses/{business_id}/accuracy")
async def get_accuracy_metrics(
    business_id: int,
    days: int = 30,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get accuracy metrics comparing predictions vs actuals"""
    from sqlalchemy import and_

    # Verify business belongs to user
    business = db.query(Business).filter(
        Business.id == business_id,
        Business.user_id == current_user.id
    ).first()

    if not business:
        raise HTTPException(status_code=404, detail="Business not found")

    # Get predictions with actuals from the last N days (or future dates with actuals for testing)
    cutoff_date = datetime.now().date() - timedelta(days=days)

    # Include any predictions that have actual_quantity set
    # This allows both past comparisons AND testing with "future" dates
    predictions = db.query(Prediction).join(TrainedModel).filter(
        TrainedModel.business_id == business_id,
        Prediction.prediction_date >= cutoff_date,
        Prediction.actual_quantity.isnot(None)
    ).all()

    if not predictions:
        return {
            "has_data": False,
            "message": "No actual sales data recorded yet",
            "metrics": None,
            "daily_comparison": []
        }

    # Calculate metrics
    total_predicted = sum(p.predicted_quantity for p in predictions)
    total_actual = sum(p.actual_quantity for p in predictions)

    errors = []
    daily_data = {}

    for p in predictions:
        error = abs(p.predicted_quantity - p.actual_quantity)
        errors.append(error)

        date_str = p.prediction_date.isoformat()
        if date_str not in daily_data:
            daily_data[date_str] = {"predicted": 0, "actual": 0}
        daily_data[date_str]["predicted"] += p.predicted_quantity
        daily_data[date_str]["actual"] += p.actual_quantity

    mae = sum(errors) / len(errors) if errors else 0

    # Calculate MAPE avoiding division by zero
    mape_values = []
    for p in predictions:
        if p.actual_quantity > 0:
            mape_values.append(abs(p.predicted_quantity - p.actual_quantity) / p.actual_quantity * 100)
    mape = sum(mape_values) / len(mape_values) if mape_values else 0
    accuracy = max(0, 100 - mape)

    # Sort daily comparison by date
    daily_comparison = [
        {"date": date, "predicted": data["predicted"], "actual": data["actual"]}
        for date, data in sorted(daily_data.items())
    ]

    return {
        "has_data": True,
        "metrics": {
            "total_predicted": total_predicted,
            "total_actual": total_actual,
            "mae": round(mae, 2),
            "mape": round(mape, 2),
            "accuracy": round(accuracy, 1),
            "data_points": len(predictions),
            "days_tracked": len(daily_data)
        },
        "daily_comparison": daily_comparison
    }


@router.get("/businesses/{business_id}/history")
async def get_historical_data(
    business_id: int,
    period: str = "month",  # "week", "month", "quarter", "year"
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get historical sales data for comparison views"""
    # Verify business belongs to user
    business = db.query(Business).filter(
        Business.id == business_id,
        Business.user_id == current_user.id
    ).first()

    if not business:
        raise HTTPException(status_code=404, detail="Business not found")

    # Calculate date ranges
    today = datetime.now().date()
    if period == "week":
        current_start = today - timedelta(days=today.weekday())
        previous_start = current_start - timedelta(weeks=1)
        days = 7
    elif period == "month":
        current_start = today.replace(day=1)
        previous_start = (current_start - timedelta(days=1)).replace(day=1)
        days = 30
    elif period == "quarter":
        quarter = (today.month - 1) // 3
        current_start = today.replace(month=quarter * 3 + 1, day=1)
        previous_start = (current_start - timedelta(days=1)).replace(day=1)
        previous_start = (previous_start - timedelta(days=1)).replace(day=1)
        previous_start = (previous_start - timedelta(days=1)).replace(day=1)
        days = 90
    else:  # year
        current_start = today.replace(month=1, day=1)
        previous_start = current_start.replace(year=today.year - 1)
        days = 365

    # Get predictions with actuals
    predictions = db.query(Prediction).join(TrainedModel).filter(
        TrainedModel.business_id == business_id,
        Prediction.actual_quantity.isnot(None)
    ).all()

    # Aggregate by period
    current_period = {"predicted": 0, "actual": 0, "days": 0}
    previous_period = {"predicted": 0, "actual": 0, "days": 0}
    by_item = {}
    by_day_of_week = {i: {"predicted": 0, "actual": 0, "count": 0} for i in range(7)}

    for p in predictions:
        if p.prediction_date >= current_start:
            current_period["predicted"] += p.predicted_quantity
            current_period["actual"] += p.actual_quantity
            current_period["days"] += 1
        elif p.prediction_date >= previous_start and p.prediction_date < current_start:
            previous_period["predicted"] += p.predicted_quantity
            previous_period["actual"] += p.actual_quantity
            previous_period["days"] += 1

        # By item
        if p.item_name not in by_item:
            by_item[p.item_name] = {"predicted": 0, "actual": 0}
        by_item[p.item_name]["predicted"] += p.predicted_quantity
        by_item[p.item_name]["actual"] += p.actual_quantity

        # By day of week
        dow = p.prediction_date.weekday()
        by_day_of_week[dow]["predicted"] += p.predicted_quantity
        by_day_of_week[dow]["actual"] += p.actual_quantity
        by_day_of_week[dow]["count"] += 1

    # Calculate averages for day of week
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_data = []
    for i, name in enumerate(day_names):
        count = by_day_of_week[i]["count"] or 1
        dow_data.append({
            "day": name,
            "avg_predicted": round(by_day_of_week[i]["predicted"] / count, 1),
            "avg_actual": round(by_day_of_week[i]["actual"] / count, 1)
        })

    # Calculate period change
    change = 0
    if previous_period["actual"] > 0:
        change = ((current_period["actual"] - previous_period["actual"]) / previous_period["actual"]) * 100

    return {
        "period": period,
        "current_period": current_period,
        "previous_period": previous_period,
        "change_percent": round(change, 1),
        "by_item": [
            {"item": item, "predicted": data["predicted"], "actual": data["actual"]}
            for item, data in sorted(by_item.items(), key=lambda x: x[1]["actual"], reverse=True)[:10]
        ],
        "by_day_of_week": dow_data
    }
