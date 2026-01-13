import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .cache import cache

_executor = ThreadPoolExecutor(max_workers=2)


def _fetch_holidays_sync(url: str) -> list:
    """Synchronous holidays fetch"""
    try:
        response = requests.get(url, timeout=5)
        return response.json()
    except Exception as e:
        print(f"Holidays fetch error: {e}")
        return []


async def get_holidays(year: int, country_code: str = "US"):
    """Fetch public holidays from Nager.Date API (cached for 24 hours)"""

    cache_key = f"holidays:{year}:{country_code}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(_executor, _fetch_holidays_sync, url)

    holidays = []
    for holiday in data:
        if isinstance(holiday, dict):
            holidays.append({
                "date": holiday.get("date", ""),
                "name": holiday.get("localName", holiday.get("name", "")),
                "type": holiday.get("types", ["Public"])[0] if holiday.get("types") else "Public"
            })

    cache.set(cache_key, holidays, ttl_seconds=86400)  # Cache for 24 hours
    return holidays