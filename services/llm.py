# services/llm.py - OpenRouter LLM integration service
import httpx
import os
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "mistralai/devstral-2512:free"  # Free model that returns actual content

def get_api_key() -> Optional[str]:
    """Get OpenRouter API key from environment"""
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        logger.info(f"OpenRouter API key found (starts with: {key[:20]}...)")
    else:
        logger.warning("OPENROUTER_API_KEY not found in environment variables")
    return key


async def check_openrouter_available() -> bool:
    """Check if OpenRouter API is accessible with valid key"""
    api_key = get_api_key()
    if not api_key:
        logger.warning("OpenRouter check failed: No API key")
        return False

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{OPENROUTER_BASE_URL}/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            logger.info(f"OpenRouter API response status: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"OpenRouter API error: {response.text[:200]}")
            return response.status_code == 200
    except Exception as e:
        logger.error(f"OpenRouter connection error: {e}")
        return False


async def get_available_models() -> List[Dict[str, Any]]:
    """Get list of available models from OpenRouter"""
    api_key = get_api_key()
    if not api_key:
        return []

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{OPENROUTER_BASE_URL}/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
    except Exception:
        pass
    return []


async def generate_response(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 500
) -> Optional[str]:
    """Generate a response from OpenRouter"""
    api_key = get_api_key()
    if not api_key:
        return None

    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://trucastai.com"),
            "X-Title": "TrucastAI"
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                json=payload,
                headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
            else:
                print(f"OpenRouter error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"OpenRouter error: {e}")
    return None


async def generate_day_summary(
    date: str,
    events: Dict[str, Any],
    weather: Optional[Dict[str, Any]] = None,
    predictions: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Generate an AI summary for a specific day"""

    # Build context
    context_parts = [f"Date: {date}"]

    # Weather info
    if weather:
        temp_max = weather.get("temp_max", "N/A")
        temp_min = weather.get("temp_min", "N/A")
        precip = weather.get("precipitation", 0)
        conditions = get_weather_description(weather.get("weather_code", 0))
        context_parts.append(f"Weather: {conditions}, High {temp_max}F, Low {temp_min}F, Precipitation: {precip}mm")

    # Events
    holidays = events.get("holidays", [])
    sports = events.get("sports", [])
    paydays = events.get("paydays", [])
    school = events.get("school", [])

    if holidays:
        holiday_names = [h.get("name", "Holiday") for h in holidays]
        context_parts.append(f"Holidays: {', '.join(holiday_names)}")

    if sports:
        game_info = [f"{g.get('home_team', '')} vs {g.get('away_team', '')}" for g in sports]
        context_parts.append(f"Sports events: {', '.join(game_info)}")

    if paydays:
        context_parts.append("This is a payday (15th or end of month)")

    if school:
        school_events = [s.get("name", "School event") for s in school]
        context_parts.append(f"School calendar: {', '.join(school_events)}")

    # Predictions
    if predictions:
        total_predicted = sum(p.get("predicted_quantity", 0) for p in predictions)
        context_parts.append(f"Predicted sales: {total_predicted:.0f} units")

    context = "\n".join(context_parts)

    system_prompt = """You are a business analyst AI assistant for a sales forecasting application.
Your job is to provide brief, actionable insights about how various factors might affect business sales.
Keep responses concise (2-3 sentences max) and focus on practical business implications.
Be specific about expected impact (e.g., "expect 15-25% higher traffic" rather than "might be busier")."""

    prompt = f"""Analyze this day and provide a brief business insight:

{context}

Provide a concise summary of what to expect for business operations on this day. Focus on actionable insights."""

    response = await generate_response(prompt, system_prompt=system_prompt, max_tokens=200)

    if not response:
        return "AI summary unavailable. Check your OpenRouter API configuration."

    return response


async def generate_forecast_summary(
    predictions: List[Dict[str, Any]],
    factors: Optional[Dict[str, Any]] = None,
    date_range: Optional[Dict[str, str]] = None
) -> str:
    """Generate an AI summary for a forecast period"""

    if not predictions:
        return "No predictions available to summarize."

    # Calculate key metrics
    total_predicted = sum(p.get("predicted_quantity", 0) for p in predictions)
    avg_daily = total_predicted / len(predictions) if predictions else 0

    # Find peak and low days
    sorted_preds = sorted(predictions, key=lambda x: x.get("predicted_quantity", 0), reverse=True)
    peak_day = sorted_preds[0] if sorted_preds else None
    low_day = sorted_preds[-1] if sorted_preds else None

    context_parts = [
        f"Forecast period: {len(predictions)} days",
        f"Total predicted sales: {total_predicted:.0f} units",
        f"Average daily: {avg_daily:.0f} units"
    ]

    if peak_day:
        context_parts.append(f"Peak day: {peak_day.get('date', 'N/A')} with {peak_day.get('predicted_quantity', 0):.0f} units")

    if low_day:
        context_parts.append(f"Lowest day: {low_day.get('date', 'N/A')} with {low_day.get('predicted_quantity', 0):.0f} units")

    if factors:
        context_parts.append(f"Key factors: {', '.join(list(factors.keys())[:5])}")

    context = "\n".join(context_parts)

    system_prompt = """You are a business analyst AI for a sales forecasting app.
Provide brief, actionable forecast insights. Focus on trends, peak periods, and preparation recommendations.
Keep responses to 3-4 sentences max."""

    prompt = f"""Summarize this sales forecast:

{context}

Provide key insights about the forecast period and any recommendations for business operations."""

    response = await generate_response(prompt, system_prompt=system_prompt, max_tokens=250)

    if not response:
        return "AI insights unavailable. Check your OpenRouter API configuration."

    return response


async def generate_dashboard_insights(
    stats: Dict[str, Any],
    recent_predictions: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Generate AI insights for the dashboard"""

    context_parts = []

    if stats.get("model_accuracy"):
        context_parts.append(f"Model accuracy (MAPE): {stats['model_accuracy']}%")

    if stats.get("total_data_points"):
        context_parts.append(f"Training data: {stats['total_data_points']} data points")

    if stats.get("forecast_total"):
        context_parts.append(f"7-day forecast total: {stats['forecast_total']} units")

    if recent_predictions:
        high_days = [p for p in recent_predictions if p.get("predicted_quantity", 0) > stats.get("avg_daily", 0)]
        if high_days:
            context_parts.append(f"Above-average days coming up: {len(high_days)}")

    if not context_parts:
        return "Upload data and train a model to see AI insights."

    context = "\n".join(context_parts)

    system_prompt = """You are a business dashboard AI assistant.
Provide a brief overview of business health and upcoming trends.
Keep it to 2-3 sentences, focusing on the most important insight."""

    prompt = f"""Based on this dashboard data, provide a brief business insight:

{context}"""

    response = await generate_response(prompt, system_prompt=system_prompt, max_tokens=150)

    if not response:
        return "AI insights unavailable. Check your OpenRouter API configuration."

    return response


async def chat_response(
    message: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    business_context: Optional[Dict[str, Any]] = None
) -> str:
    """Generate a chat response with business context"""
    api_key = get_api_key()
    if not api_key:
        return "AI chat is unavailable. Please configure your OpenRouter API key."

    system_prompt = """You are an AI assistant for TrucastAI, a sales forecasting application.
You help business owners understand their sales predictions, analyze factors affecting sales, and provide actionable advice.

IMPORTANT: You have access to the business's actual forecast data provided in the context below.
When answering questions about busiest days, slowest days, or forecasts, USE THE DATA PROVIDED.
Do not say you don't have access to data if it's provided in the context.

Be helpful, concise, and focus on practical business insights.
Give specific answers based on the forecast data when available."""

    # Build context from business data
    context_parts = []
    if business_context:
        if business_context.get("business_name"):
            context_parts.append(f"Business: {business_context['business_name']}")
        if business_context.get("business_type"):
            context_parts.append(f"Business Type: {business_context['business_type']}")
        if business_context.get("location"):
            context_parts.append(f"Location: {business_context['location']}")
        if business_context.get("model_accuracy"):
            context_parts.append(f"Model Accuracy: {business_context['model_accuracy']}")
        if business_context.get("busiest_day"):
            context_parts.append(f"Predicted Busiest Day: {business_context['busiest_day']}")
        if business_context.get("slowest_day"):
            context_parts.append(f"Predicted Slowest Day: {business_context['slowest_day']}")
        if business_context.get("products_tracked"):
            context_parts.append(f"Products/Items Tracked ({business_context.get('total_products', 'N/A')} total): {business_context['products_tracked']}")
        if business_context.get("weather_forecast"):
            context_parts.append(f"\nWeather Forecast (7 days):\n{business_context['weather_forecast']}")
        if business_context.get("upcoming_holidays"):
            context_parts.append(f"Upcoming Holidays: {business_context['upcoming_holidays']}")
        if business_context.get("forecast_next_14_days"):
            context_parts.append(f"\nSales Forecast (next 14 days):\n{business_context['forecast_next_14_days']}")

    # Build messages
    messages = []
    if context_parts:
        messages.append({
            "role": "system",
            "content": system_prompt + "\n\nCurrent business context:\n" + "\n".join(context_parts)
        })
    else:
        messages.append({"role": "system", "content": system_prompt})

    # Add conversation history
    if conversation_history:
        for msg in conversation_history[-10:]:  # Last 10 messages for context
            messages.append(msg)

    # Add current message
    messages.append({"role": "user", "content": message})

    try:
        payload = {
            "model": DEFAULT_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://trucastai.com"),
            "X-Title": "TrucastAI"
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                json=payload,
                headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "I couldn't generate a response.")
            else:
                logger.error(f"OpenRouter chat error: {response.status_code} - {response.text}")
                return f"API error ({response.status_code}). Please try again."
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return f"Connection error: {str(e)[:100]}"

    return "I'm currently unavailable. Please check your OpenRouter API configuration."


def get_weather_description(code: int) -> str:
    """Convert weather code to human-readable description"""
    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }
    return weather_codes.get(code, "Unknown")
