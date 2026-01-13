# routers/llm.py - LLM API endpoints
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
from db_models import User, Business, TrainedModel
from auth import get_current_user
from services.llm import (
    check_ollama_available,
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
    """Check if Ollama LLM is available"""
    available = await check_ollama_available()
    models = await get_available_models() if available else []

    return {
        "available": available,
        "models": models,
        "message": "Ollama is running" if available else "Ollama is not available. Run 'ollama serve' to start it."
    }


@router.post("/summarize/day", response_model=LLMResponse)
async def summarize_day(
    request: DaySummaryRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate an AI summary for a specific day"""
    available = await check_ollama_available()

    if not available:
        return LLMResponse(
            success=False,
            content="AI summary unavailable. Start Ollama with 'ollama serve' to enable AI features.",
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
    available = await check_ollama_available()

    if not available:
        return LLMResponse(
            success=False,
            content="AI insights unavailable. Start Ollama with 'ollama serve' to enable AI features.",
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
    available = await check_ollama_available()

    if not available:
        return LLMResponse(
            success=False,
            content="AI insights unavailable. Start Ollama with 'ollama serve' to enable AI features.",
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
    available = await check_ollama_available()

    if not available:
        return LLMResponse(
            success=False,
            content="Chat unavailable. Start Ollama with 'ollama serve' to enable AI features.",
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
            business_context["location"] = f"{business.city}, {business.state}"

            # Get active model info
            model = db.query(TrainedModel).filter(
                TrainedModel.business_id == business.id,
                TrainedModel.is_active == True
            ).first()

            if model:
                business_context["model_accuracy"] = f"{model.test_mape:.1f}% MAPE" if model.test_mape else "Unknown"
                business_context["data_range"] = f"{model.data_start_date} to {model.data_end_date}"

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
