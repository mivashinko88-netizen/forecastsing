import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .cache import cache

_executor = ThreadPoolExecutor(max_workers=2)


def _fetch_nfl_sync(url: str, params: dict) -> dict:
    """Synchronous NFL fetch"""
    try:
        response = requests.get(url, params=params, timeout=5)
        return response.json()
    except Exception as e:
        print(f"NFL fetch error: {e}")
        return {}


async def get_nfl_games(year: int, season_type: str = "REG"):
    """Fetch NFL games from ESPN API (cached for 6 hours)"""

    cache_key = f"nfl_games:{year}:{season_type}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    params = {
        "dates": year,
        "seasontype": 2 if season_type == "REG" else 3
    }

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(_executor, _fetch_nfl_sync, url, params)

    games = []
    for event in data.get("events", []):
        try:
            competitions = event.get("competitions", [{}])
            competitors = competitions[0].get("competitors", [{}, {}])
            games.append({
                "date": event.get("date", "")[:10],
                "name": event.get("name", ""),
                "home_team": competitors[0].get("team", {}).get("displayName", ""),
                "away_team": competitors[1].get("team", {}).get("displayName", "") if len(competitors) > 1 else "",
                "venue": competitions[0].get("venue", {}).get("fullName", "Unknown")
            })
        except (IndexError, KeyError):
            continue

    cache.set(cache_key, games, ttl_seconds=21600)  # Cache for 6 hours
    return games


async def get_nfl_schedule(year: int, week: int = None):
    """Fetch NFL schedule for a season or specific week (cached for 6 hours)"""

    cache_key = f"nfl_schedule:{year}:{week}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    params = {"dates": year}
    if week:
        params["week"] = week

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(_executor, _fetch_nfl_sync, url, params)

    games = []
    for event in data.get("events", []):
        try:
            competitions = event.get("competitions", [{}])
            competitors = competitions[0].get("competitors", [{}, {}])
            games.append({
                "date": event.get("date", "")[:10],
                "name": event.get("name", ""),
                "home_team": competitors[0].get("team", {}).get("displayName", ""),
                "away_team": competitors[1].get("team", {}).get("displayName", "") if len(competitors) > 1 else ""
            })
        except (IndexError, KeyError):
            continue

    cache.set(cache_key, games, ttl_seconds=21600)  # Cache for 6 hours
    return games