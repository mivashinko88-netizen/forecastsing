import requests
from datetime import date
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .cache import cache

# Thread pool for running sync requests without blocking
_executor = ThreadPoolExecutor(max_workers=4)


def _fetch_weather_sync(url: str, params: dict) -> dict:
    """Synchronous weather fetch (runs in thread pool)"""
    try:
        response = requests.get(url, params=params, timeout=5)
        return response.json()
    except Exception as e:
        print(f"Weather fetch error: {e}")
        return {}


async def get_historical_weather(latitude: float, longitude: float, start_date: date, end_date: date):
    """Fetch historical weather data from Open-Meteo (cached for 1 hour)"""

    cache_key = f"weather_hist:{latitude}:{longitude}:{start_date}:{end_date}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "weathercode"],
        "timezone": "auto"
    }

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(_executor, _fetch_weather_sync, url, params)

    daily = data.get("daily", {})
    weather_records = []
    for i, d in enumerate(daily.get("time", [])):
        weather_records.append({
            "date": d,
            "temp_max": daily["temperature_2m_max"][i] if daily.get("temperature_2m_max") else None,
            "temp_min": daily["temperature_2m_min"][i] if daily.get("temperature_2m_min") else None,
            "precipitation": daily["precipitation_sum"][i] if daily.get("precipitation_sum") else None,
            "weather_code": daily["weathercode"][i] if daily.get("weathercode") else None
        })

    cache.set(cache_key, weather_records, ttl_seconds=3600)  # Cache for 1 hour
    return weather_records


async def get_forecast_weather(latitude: float, longitude: float, days: int = 7):
    """Fetch weather forecast from Open-Meteo (cached for 30 minutes)"""

    cache_key = f"weather_forecast:{latitude}:{longitude}:{days}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "weathercode"],
        "timezone": "auto",
        "forecast_days": days
    }

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(_executor, _fetch_weather_sync, url, params)

    daily = data.get("daily", {})
    weather_records = []
    for i, d in enumerate(daily.get("time", [])):
        weather_records.append({
            "date": d,
            "temp_max": daily["temperature_2m_max"][i] if daily.get("temperature_2m_max") else None,
            "temp_min": daily["temperature_2m_min"][i] if daily.get("temperature_2m_min") else None,
            "precipitation": daily["precipitation_sum"][i] if daily.get("precipitation_sum") else None,
            "weather_code": daily["weathercode"][i] if daily.get("weathercode") else None
        })

    cache.set(cache_key, weather_records, ttl_seconds=1800)  # Cache for 30 minutes
    return weather_records