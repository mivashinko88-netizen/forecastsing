# main.py
from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from processor import process_csv
import os
from config import BusinessConfig
from ai.trainer import SalesForecaster
from services.weather import get_historical_weather
from services.events import get_holidays
from services.sports import get_nfl_games
from services.cycles import get_payday_dates
from services.school import get_school_calendar
import pandas as pd
import numpy as np
import json
import math
from datetime import datetime
from pathlib import Path
import logging
import asyncio

# Settings and monitoring
from settings import get_settings
settings = get_settings()

# Initialize Sentry error monitoring (if configured)
if settings.SENTRY_DSN:
    import sentry_sdk
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.ENVIRONMENT,
        traces_sample_rate=0.1,  # 10% sampling for performance (free tier friendly)
    )
    print(f"Sentry initialized for {settings.ENVIRONMENT} environment")

# Database initialization
from database import engine, Base, init_db
from db_models import User, Business, TrainedModel, Upload, Prediction

# Import routers
from routers import auth, businesses, models, llm, integrations

# Import scheduler
from scheduler import start_scheduler, shutdown_scheduler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where main.py is located (needed for migrations)
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="TrucastAI", version="1.0.0")


def run_migrations():
    """Run Alembic migrations to ensure database schema is up to date."""
    import traceback
    from sqlalchemy import text

    # First, try to apply any missing columns directly (handles edge cases)
    try:
        with engine.connect() as conn:
            # Check if address column exists
            result = conn.execute(text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'businesses' AND column_name = 'address'
            """))
            if not result.fetchone():
                logger.info("Adding missing 'address' column to businesses table...")
                conn.execute(text("ALTER TABLE businesses ADD COLUMN address VARCHAR"))
                conn.commit()
                logger.info("Column 'address' added successfully")

            # Check if model_data column exists in trained_models
            result = conn.execute(text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'trained_models' AND column_name = 'model_data'
            """))
            if not result.fetchone():
                logger.info("Adding missing 'model_data' column to trained_models table...")
                conn.execute(text("ALTER TABLE trained_models ADD COLUMN model_data BYTEA"))
                conn.commit()
                logger.info("Column 'model_data' added successfully")

            # Make model_path nullable if not already (ignore errors)
            try:
                conn.execute(text("ALTER TABLE trained_models ALTER COLUMN model_path DROP NOT NULL"))
                conn.commit()
            except:
                pass  # Column may already be nullable
    except Exception as e:
        logger.warning(f"Direct column check/add failed (may be OK): {e}")

    # Then try Alembic migrations
    try:
        from alembic.config import Config
        from alembic import command

        # Get absolute path to alembic.ini
        alembic_ini_path = BASE_DIR / "alembic.ini"
        logger.info(f"Looking for alembic.ini at: {alembic_ini_path}")

        if not alembic_ini_path.exists():
            raise FileNotFoundError(f"alembic.ini not found at {alembic_ini_path}")

        # Create Alembic config with absolute path
        alembic_cfg = Config(str(alembic_ini_path))
        alembic_cfg.set_main_option("script_location", str(BASE_DIR / "alembic"))

        # Run migrations
        logger.info("Running Alembic migrations...")
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations completed successfully")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        logger.error(traceback.format_exc())
        # Fallback to create_all for fresh databases only
        logger.info("Falling back to create_all...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created via create_all")


# Initialize database and scheduler on startup
@app.on_event("startup")
async def startup_event():
    run_migrations()
    start_scheduler()
    logger.info("Background scheduler started")

@app.on_event("shutdown")
async def shutdown_event():
    shutdown_scheduler()
    logger.info("Background scheduler stopped")

# CORS configuration - use environment-specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Include API routers
app.include_router(auth.router, prefix="/api")
app.include_router(businesses.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(llm.router, prefix="/api")
app.include_router(integrations.router, prefix="/api")

# Frontend directory for static files
FRONTEND_DIR = BASE_DIR / "frontend"

# Mount static files for CSS and JS
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

@app.get("/")
async def serve_frontend():
    """Serve the home/landing page"""
    home_path = FRONTEND_DIR / "pages" / "home.html"
    if not home_path.exists():
        raise HTTPException(status_code=404, detail=f"Home page not found at {home_path}")
    return FileResponse(home_path)

def clean_for_json(obj):
    """Replace NaN and Inf values with None for JSON serialization"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    elif pd.isna(obj):
        return None
    return obj

@app.post("/upload")
async def upload_sales_data(file: UploadFile):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files supported")

    result = await process_csv(file)
    return result

@app.post("/train")
async def train_model(
    file: UploadFile,
    config: str = Form(default=""),
    business_id: int = Form(default=None)
):
    print(f"DEBUG /train: config received = {repr(config)}")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files supported")

    # Parse config from frontend - use defaults if invalid/missing
    config_data = None
    if config and config.strip() and config != "string":
        try:
            config_data = json.loads(config)
            print(f"DEBUG /train: config parsed successfully = {config_data}")
        except json.JSONDecodeError as e:
            print(f"DEBUG /train: JSON parse failed: {e}")
            pass  # Will use defaults below
    else:
        print(f"DEBUG /train: Using default config (config was empty or 'string')")

    # Try to get cached coordinates from database to avoid slow geocoding API call
    cached_lat, cached_lon = None, None
    if business_id:
        from database import SessionLocal
        db = SessionLocal()
        try:
            business = db.query(Business).filter(Business.id == business_id).first()
            if business:
                cached_lat = business.latitude
                cached_lon = business.longitude
        finally:
            db.close()

    if config_data:
        business_config = BusinessConfig(
            business_name=config_data.get("business_name", "My Business"),
            city=config_data.get("city", "Denver"),
            state=config_data.get("state", "CO"),
            zipcode=config_data.get("zipcode", "80202"),
            country=config_data.get("country", "US"),
            timezone=config_data.get("timezone", "America/New_York"),
            latitude=cached_lat,
            longitude=cached_lon
        )
    else:
        business_config = BusinessConfig(
            business_name="Test Shop",
            city="Denver",
            state="CO",
            zipcode="80202",
            latitude=cached_lat,
            longitude=cached_lon
        )
    
    # Read and prepare data
    df = pd.read_csv(file.file)
    df.columns = df.columns.str.lower().str.strip()

    # Flexible column mapping - try common variations
    column_mapping = {
        # Date columns
        "order_date": "date",
        "sale_date": "date",
        "transaction_date": "date",
        "order date": "date",
        "sale date": "date",
        "trans_date": "date",
        # Item/product columns
        "pizza_name": "item_name",
        "product": "item_name",
        "product_name": "item_name",
        "item": "item_name",
        "name": "item_name",
        "product name": "item_name",
        "menu_item": "item_name",
        "item_description": "item_name",
        "description": "item_name",
        # Size columns
        "pizza_size": "size",
        "product_size": "size",
        "item_size": "size",
        "portion": "size",
        "portion_size": "size",
        # Type/category columns
        "pizza_category": "type",
        "pizza_type": "type",
        "product_type": "type",
        "item_type": "type",
        "category": "type",
        "product_category": "type",
        "item_category": "type",
        "food_type": "type",
        "menu_category": "type",
        # Unit price columns (sale price to customer)
        "price": "unit_price",
        "item_price": "unit_price",
        "product_price": "unit_price",
        "sale_price": "unit_price",
        "selling_price": "unit_price",
        "retail_price": "unit_price",
        "price_each": "unit_price",
        "each_price": "unit_price",
        # Cost columns (business cost, separate from sale price)
        "item_cost": "cost",
        "product_cost": "cost",
        "unit_cost": "cost",
        # Total price columns
        "total_price": "total_price",
        "total": "total_price",
        "revenue": "total_price",
        "amount_paid": "total_price",
        "order_total": "total_price",
        "line_total": "total_price",
        "subtotal": "total_price",
        "sales": "total_price",
        "income": "total_price",
        "sale_amount": "total_price",
        # Order ID columns
        "order": "order_id",
        "transaction_id": "order_id",
        "receipt": "order_id",
        "ticket": "order_id",
        "order_details_id": "order_id",
        "invoice": "order_id",
        "invoice_id": "order_id",
        "receipt_id": "order_id",
        # Quantity columns
        "qty": "quantity",
        "amount": "quantity",
        "units": "quantity",
        "units_sold": "quantity",
        "qty_sold": "quantity",
        "order_qty": "quantity",
        "count": "quantity",
        "num_sold": "quantity",
        # Time columns
        "order_time": "time",
        "sale_time": "time",
        "transaction_time": "time",
        "time_of_sale": "time",
    }
    df = df.rename(columns=column_mapping)

    # Check for required columns
    if "date" not in df.columns:
        available_cols = ", ".join(df.columns.tolist())
        raise HTTPException(status_code=400, detail=f"No date column found. Available columns: {available_cols}")

    # If no item_name, try to find one or use a default
    if "item_name" not in df.columns:
        # Check if there's any text column we can use
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        if text_cols and text_cols[0] != "date":
            df = df.rename(columns={text_cols[0]: "item_name"})
        else:
            df["item_name"] = "All Items"  # Aggregate all sales together

    if "quantity" not in df.columns:
        df["quantity"] = 1

    # Get date range from data - handle various date formats
    try:
        # Try to infer format automatically, with dayfirst for international formats
        df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    except Exception:
        try:
            # Fallback to mixed format inference
            df["date"] = pd.to_datetime(df["date"], format='mixed', dayfirst=True)
        except Exception:
            # Last resort - let pandas guess
            df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    year = min_date.year
    
    # Fetch external data in parallel for faster loading
    async def fetch_weather():
        if business_config.latitude and business_config.longitude:
            try:
                return await get_historical_weather(
                    latitude=business_config.latitude,
                    longitude=business_config.longitude,
                    start_date=min_date,
                    end_date=max_date
                )
            except Exception as e:
                print(f"Weather fetch failed: {e}")
        return []

    async def fetch_holidays():
        try:
            return await get_holidays(year, business_config.country)
        except Exception as e:
            print(f"Holidays fetch failed: {e}")
            return []

    async def fetch_sports():
        try:
            return await get_nfl_games(year)
        except Exception as e:
            print(f"Sports fetch failed: {e}")
            return []

    async def fetch_paydays():
        try:
            return get_payday_dates(year)
        except Exception as e:
            print(f"Paydays fetch failed: {e}")
            return []

    async def fetch_school():
        try:
            return get_school_calendar(year)
        except Exception as e:
            print(f"School calendar fetch failed: {e}")
            return []

    # Run all external data fetches in parallel
    weather_data, holidays, sports_games, paydays, school_calendar = await asyncio.gather(
        fetch_weather(),
        fetch_holidays(),
        fetch_sports(),
        fetch_paydays(),
        fetch_school()
    )
    
    # Keep a copy of raw data for metadata extraction (before aggregation)
    df_raw = df.copy()

    # Aggregate sales data
    df_agg = df.groupby(["date", "item_name"]).agg({
        "quantity": "sum"
    }).reset_index()

    df_agg["date"] = df_agg["date"].dt.strftime("%Y-%m-%d")

    # Train with all external data, passing raw_df for metadata extraction
    forecaster = SalesForecaster(business_config)
    results = forecaster.train(
        df_agg,
        weather_data=weather_data,
        holidays=holidays,
        sports_games=sports_games,
        paydays=paydays,
        school_calendar=school_calendar,
        raw_df=df_raw  # Pass raw data for size, time, price extraction
    )
    
    results["external_data"] = {
        "weather_days": len(weather_data),
        "holidays": len(holidays),
        "sports_games": len(sports_games),
        "paydays": len(paydays),
        "school_events": len(school_calendar)
    }

    # Get unique items trained
    items_trained = df_agg["item_name"].unique().tolist()

    # Save model to database if business_id provided
    if business_id:
        from database import SessionLocal
        db = SessionLocal()
        try:
            # Serialize model to bytes for database storage (cloud-persistent)
            model_bytes = forecaster.serialize_model()

            # Deactivate existing models for this business
            db.query(TrainedModel).filter(
                TrainedModel.business_id == business_id
            ).update({"is_active": False})

            # Create new model record with binary data
            trained_model = TrainedModel(
                business_id=business_id,
                model_name="xgboost",
                model_path=None,  # No longer using file path
                model_data=model_bytes,  # Store serialized model in database
                train_mae=results.get("train_mae"),
                test_mae=results.get("test_mae"),
                train_mape=results.get("train_mape"),
                test_mape=results.get("test_mape"),
                training_rows=results.get("training_rows", 0),
                test_rows=results.get("test_rows", 0),
                feature_importance=json.dumps(results.get("feature_importance", {})),
                items_trained=json.dumps(items_trained),
                data_start_date=min_date,
                data_end_date=max_date,
                is_active=True
            )
            db.add(trained_model)
            db.commit()

            results["model_id"] = trained_model.id
            results["model_saved"] = True

            # Auto-compare: Match uploaded actuals with any existing predictions
            # This allows the Analytics page to show accuracy without manual entry
            try:
                # Aggregate actual quantities by date and item from uploaded data
                df_actuals = df.copy()
                df_actuals["date"] = pd.to_datetime(df_actuals["date"]).dt.date
                actual_sales = df_actuals.groupby(["date", "item_name"])["quantity"].sum().reset_index()

                print(f"Auto-compare: Processing {len(actual_sales)} unique date/item combinations")
                print(f"Auto-compare: Date range in uploaded data: {actual_sales['date'].min()} to {actual_sales['date'].max()}")

                # Find predictions for this business (from any model)
                business_models = db.query(TrainedModel).filter(
                    TrainedModel.business_id == business_id
                ).all()
                model_ids = [m.id for m in business_models]
                print(f"Auto-compare: Found {len(model_ids)} models for business {business_id}")

                if model_ids:
                    # Build a lookup dict of actuals for fast O(1) access
                    actuals_lookup = {
                        (row["date"], row["item_name"]): int(row["quantity"])
                        for _, row in actual_sales.iterrows()
                    }

                    # Get date range for efficient query
                    min_actual_date = actual_sales["date"].min()
                    max_actual_date = actual_sales["date"].max()

                    # Fetch only predictions within our date range (much faster than iterating all)
                    predictions_to_update = db.query(Prediction).filter(
                        Prediction.model_id.in_(model_ids),
                        Prediction.prediction_date >= min_actual_date,
                        Prediction.prediction_date <= max_actual_date
                    ).all()

                    print(f"Auto-compare: Found {len(predictions_to_update)} predictions in date range")

                    # Batch update in memory, then commit once
                    updated_count = 0
                    for pred in predictions_to_update:
                        key = (pred.prediction_date, pred.item_name)
                        if key in actuals_lookup:
                            pred.actual_quantity = actuals_lookup[key]
                            updated_count += 1

                    db.commit()
                    print(f"Auto-compare: Updated {updated_count} predictions with actual quantities")
                    results["actuals_matched"] = updated_count

            except Exception as e:
                import traceback
                print(f"Auto-compare failed (non-critical): {e}")
                traceback.print_exc()

        except Exception as e:
            print(f"Error saving model to database: {e}")
            results["model_saved"] = False
        finally:
            db.close()

    # Clean NaN values before returning
    return clean_for_json(results)


@app.post("/train/stream")
async def train_model_with_progress(
    file: UploadFile,
    config: str = Form(default=""),
    business_id: int = Form(default=None)
):
    """Train model with real-time progress updates via Server-Sent Events"""

    async def generate_progress():
        try:
            # Step 1: Processing CSV
            yield f"data: {json.dumps({'step': 'processing', 'message': 'Processing your CSV file...', 'progress': 10})}\n\n"

            if not file.filename.endswith(".csv"):
                yield f"data: {json.dumps({'step': 'error', 'message': 'Only CSV files supported'})}\n\n"
                return

            # Parse config
            config_data = None
            if config and config.strip() and config != "string":
                try:
                    config_data = json.loads(config)
                except json.JSONDecodeError:
                    pass

            # Get cached coordinates
            cached_lat, cached_lon = None, None
            if business_id:
                from database import SessionLocal
                db = SessionLocal()
                try:
                    business = db.query(Business).filter(Business.id == business_id).first()
                    if business:
                        cached_lat = business.latitude
                        cached_lon = business.longitude
                finally:
                    db.close()

            if config_data:
                business_config = BusinessConfig(
                    business_name=config_data.get("business_name", "My Business"),
                    city=config_data.get("city", "Denver"),
                    state=config_data.get("state", "CO"),
                    zipcode=config_data.get("zipcode", "80202"),
                    country=config_data.get("country", "US"),
                    timezone=config_data.get("timezone", "America/New_York"),
                    latitude=cached_lat,
                    longitude=cached_lon
                )
            else:
                business_config = BusinessConfig(
                    business_name="Test Shop",
                    city="Denver",
                    state="CO",
                    zipcode="80202",
                    latitude=cached_lat,
                    longitude=cached_lon
                )

            # Read CSV
            file.file.seek(0)  # Reset file position
            df = pd.read_csv(file.file)
            df.columns = df.columns.str.lower().str.strip()

            yield f"data: {json.dumps({'step': 'processing', 'message': f'Found {len(df):,} rows in your data', 'progress': 20})}\n\n"

            # Column mapping (same as /train)
            column_mapping = {
                "order_date": "date", "sale_date": "date", "transaction_date": "date",
                "order date": "date", "sale date": "date", "trans_date": "date",
                "pizza_name": "item_name", "product": "item_name", "product_name": "item_name",
                "item": "item_name", "name": "item_name", "product name": "item_name",
                "menu_item": "item_name", "item_description": "item_name", "description": "item_name",
                "pizza_size": "size", "product_size": "size", "item_size": "size",
                "portion": "size", "portion_size": "size",
                "pizza_category": "type", "pizza_type": "type", "product_type": "type",
                "item_type": "type", "category": "type", "product_category": "type",
                "pizza_price": "unit_price", "unit_price": "unit_price", "price": "unit_price",
                "order_quantity": "quantity", "qty": "quantity", "units_sold": "quantity",
            }
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

            # Parse dates
            if "date" in df.columns:
                for fmt in [None, True, False]:
                    try:
                        if fmt is None:
                            df["date"] = pd.to_datetime(df["date"], format="mixed")
                        else:
                            df["date"] = pd.to_datetime(df["date"], dayfirst=fmt)
                        break
                    except:
                        continue

            # Ensure quantity column
            if "quantity" not in df.columns:
                df["quantity"] = 1

            min_date = df["date"].min().date()
            max_date = df["date"].max().date()
            year = min_date.year

            # Step 2: Fetch external data
            yield f"data: {json.dumps({'step': 'external', 'message': 'Fetching weather, holidays, and event data...', 'progress': 35})}\n\n"

            async def fetch_weather():
                if business_config.latitude and business_config.longitude:
                    try:
                        return await get_historical_weather(
                            latitude=business_config.latitude,
                            longitude=business_config.longitude,
                            start_date=min_date,
                            end_date=max_date
                        )
                    except Exception as e:
                        print(f"Weather fetch failed: {e}")
                return []

            async def fetch_holidays():
                try:
                    return await get_holidays(year, business_config.country)
                except Exception as e:
                    print(f"Holidays fetch failed: {e}")
                    return []

            async def fetch_sports():
                try:
                    return await get_nfl_games(year)
                except Exception as e:
                    print(f"Sports fetch failed: {e}")
                    return []

            async def fetch_paydays():
                try:
                    return get_payday_dates(year)
                except Exception as e:
                    print(f"Paydays fetch failed: {e}")
                    return []

            async def fetch_school():
                try:
                    return get_school_calendar(year)
                except Exception as e:
                    print(f"School calendar fetch failed: {e}")
                    return []

            weather_data, holidays, sports_games, paydays, school_calendar = await asyncio.gather(
                fetch_weather(),
                fetch_holidays(),
                fetch_sports(),
                fetch_paydays(),
                fetch_school()
            )

            external_count = len(weather_data) + len(holidays) + len(sports_games) + len(paydays) + len(school_calendar)
            yield f"data: {json.dumps({'step': 'external', 'message': f'Loaded {external_count:,} external data points', 'progress': 50})}\n\n"

            # Step 3: Train model
            yield f"data: {json.dumps({'step': 'training', 'message': 'Training AI model (this may take a moment)...', 'progress': 60})}\n\n"

            df_raw = df.copy()
            df_agg = df.groupby(["date", "item_name"]).agg({"quantity": "sum"}).reset_index()
            df_agg["date"] = df_agg["date"].dt.strftime("%Y-%m-%d")

            forecaster = SalesForecaster(business_config)
            results = forecaster.train(
                df_agg,
                weather_data=weather_data,
                holidays=holidays,
                sports_games=sports_games,
                paydays=paydays,
                school_calendar=school_calendar,
                raw_df=df_raw
            )

            yield f"data: {json.dumps({'step': 'training', 'message': 'Model training complete!', 'progress': 80})}\n\n"

            results["external_data"] = {
                "weather_days": len(weather_data),
                "holidays": len(holidays),
                "sports_games": len(sports_games),
                "paydays": len(paydays),
                "school_events": len(school_calendar)
            }

            items_trained = df_agg["item_name"].unique().tolist()

            # Step 4: Save to database
            if business_id:
                yield f"data: {json.dumps({'step': 'saving', 'message': 'Saving model to database...', 'progress': 90})}\n\n"

                from database import SessionLocal
                db = SessionLocal()
                try:
                    model_bytes = forecaster.serialize_model()
                    db.query(TrainedModel).filter(TrainedModel.business_id == business_id).update({"is_active": False})

                    trained_model = TrainedModel(
                        business_id=business_id,
                        model_name="xgboost",
                        model_path=None,
                        model_data=model_bytes,
                        train_mae=results.get("train_mae"),
                        test_mae=results.get("test_mae"),
                        train_mape=results.get("train_mape"),
                        test_mape=results.get("test_mape"),
                        training_rows=results.get("training_rows", 0),
                        test_rows=results.get("test_rows", 0),
                        feature_importance=json.dumps(results.get("feature_importance", {})),
                        items_trained=json.dumps(items_trained),
                        data_start_date=min_date,
                        data_end_date=max_date,
                        is_active=True
                    )
                    db.add(trained_model)
                    db.commit()

                    results["model_id"] = trained_model.id
                    results["model_saved"] = True

                except Exception as e:
                    print(f"Error saving model: {e}")
                    results["model_saved"] = False
                finally:
                    db.close()

            # Final result
            yield f"data: {json.dumps({'step': 'complete', 'message': 'Training complete!', 'progress': 100, 'results': clean_for_json(results)})}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'step': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )