# routers/llm.py - LLM API endpoints
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
from db_models import User, Business, TrainedModel, Prediction
from datetime import date, timedelta
from auth import get_current_user
from services.llm import (
    check_openrouter_available,
    get_available_models,
    generate_day_summary,
    generate_forecast_summary,
    generate_dashboard_insights,
    chat_response
)

router = APIRouter(prefix="/llm", tags=["LLM"])


class DaySummaryRequest(BaseModel):
    date: str
    events: Dict[str, Any] = {}
    weather: Optional[Dict[str, Any]] = None
    predictions: Optional[List[Dict[str, Any]]] = None


class ForecastSummaryRequest(BaseModel):
    predictions: List[Dict[str, Any]]
    factors: Optional[Dict[str, Any]] = None
    date_range: Optional[Dict[str, str]] = None


class DashboardInsightsRequest(BaseModel):
    stats: Dict[str, Any]
    recent_predictions: Optional[List[Dict[str, Any]]] = None


class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = None
    business_id: Optional[int] = None


class LLMResponse(BaseModel):
    success: bool
    content: str
    available: bool = True


@router.get("/status")
async def get_llm_status():
    """Check if OpenRouter LLM is available"""
    import os

    # Debug: Check if key exists in environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    key_exists = api_key is not None and len(api_key) > 0
    key_preview = f"{api_key[:15]}...{api_key[-4:]}" if key_exists and len(api_key) > 20 else "NOT SET"

    available = await check_openrouter_available()
    models = await get_available_models() if available else []

    # Return just model IDs for simpler response
    model_ids = [m.get("id", "") for m in models[:20]] if models else []

    return {
        "available": available,
        "models": model_ids,
        "key_configured": key_exists,
        "key_preview": key_preview,
        "message": "OpenRouter AI is connected" if available else "OpenRouter API key not configured or invalid. Add OPENROUTER_API_KEY to enable AI features."
    }


@router.post("/summarize/day", response_model=LLMResponse)
async def summarize_day(
    request: DaySummaryRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate an AI summary for a specific day"""
    available = await check_openrouter_available()

    if not available:
        return LLMResponse(
            success=False,
            content="AI summary unavailable. Configure OpenRouter API key to enable AI features.",
            available=False
        )

    summary = await generate_day_summary(
        date=request.date,
        events=request.events,
        weather=request.weather,
        predictions=request.predictions
    )

    return LLMResponse(
        success=True,
        content=summary,
        available=True
    )


@router.post("/summarize/forecast", response_model=LLMResponse)
async def summarize_forecast(
    request: ForecastSummaryRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate an AI summary for a forecast period"""
    available = await check_openrouter_available()

    if not available:
        return LLMResponse(
            success=False,
            content="AI insights unavailable. Configure OpenRouter API key to enable AI features.",
            available=False
        )

    summary = await generate_forecast_summary(
        predictions=request.predictions,
        factors=request.factors,
        date_range=request.date_range
    )

    return LLMResponse(
        success=True,
        content=summary,
        available=True
    )


@router.post("/summarize/dashboard", response_model=LLMResponse)
async def summarize_dashboard(
    request: DashboardInsightsRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate AI insights for the dashboard"""
    available = await check_openrouter_available()

    if not available:
        return LLMResponse(
            success=False,
            content="AI insights unavailable. Configure OpenRouter API key to enable AI features.",
            available=False
        )

    insights = await generate_dashboard_insights(
        stats=request.stats,
        recent_predictions=request.recent_predictions
    )

    return LLMResponse(
        success=True,
        content=insights,
        available=True
    )


@router.post("/chat", response_model=LLMResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Chat with the AI assistant"""
    available = await check_openrouter_available()

    if not available:
        return LLMResponse(
            success=False,
            content="Chat unavailable. Configure OpenRouter API key to enable AI features.",
            available=False
        )

    # Build business context
    business_context = {}

    if request.business_id:
        business = db.query(Business).filter(
            Business.id == request.business_id,
            Business.user_id == current_user.id
        ).first()

        if business:
            business_context["business_name"] = business.name
            business_context["business_type"] = business.business_type
            if business.city and business.state:
                business_context["location"] = f"{business.city}, {business.state}"

            # Get active model info
            model = db.query(TrainedModel).filter(
                TrainedModel.business_id == business.id,
                TrainedModel.is_active == True
            ).first()

            if model:
                business_context["model_accuracy"] = f"{model.test_mape:.1f}% MAPE" if model.test_mape else "Unknown"
                business_context["data_range"] = f"{model.data_start_date} to {model.data_end_date}"

                # Get predictions for next 14 days
                today = date.today()
                next_two_weeks = today + timedelta(days=14)

                predictions = db.query(Prediction).filter(
                    Prediction.model_id == model.id,
                    Prediction.prediction_date >= today,
                    Prediction.prediction_date <= next_two_weeks
                ).order_by(Prediction.prediction_date).all()

                if predictions:
                    # Format predictions for context
                    forecast_data = []
                    daily_totals = {}

                    for pred in predictions:
                        date_str = pred.prediction_date.strftime("%Y-%m-%d")
                        day_name = pred.prediction_date.strftime("%A")

                        if date_str not in daily_totals:
                            daily_totals[date_str] = {"day": day_name, "total": 0, "items": []}

                        daily_totals[date_str]["total"] += pred.predicted_quantity or 0
                        if pred.item_name:
                            daily_totals[date_str]["items"].append(f"{pred.item_name}: {pred.predicted_quantity}")

                    # Create summary
                    forecast_summary = []
                    for date_str, data in sorted(daily_totals.items()):
                        forecast_summary.append(f"{data['day']} ({date_str}): {data['total']} units predicted")

                    business_context["forecast_next_14_days"] = "\n".join(forecast_summary)

                    # Find busiest and slowest days
                    if daily_totals:
                        busiest = max(daily_totals.items(), key=lambda x: x[1]["total"])
                        slowest = min(daily_totals.items(), key=lambda x: x[1]["total"])
                        business_context["busiest_day"] = f"{busiest[1]['day']} ({busiest[0]}) with {busiest[1]['total']} units"
                        business_context["slowest_day"] = f"{slowest[1]['day']} ({slowest[0]}) with {slowest[1]['total']} units"

    response_text = await chat_response(
        message=request.message,
        conversation_history=request.conversation_history,
        business_context=business_context
    )

    return LLMResponse(
        success=True,
        content=response_text,
        available=True
    )
