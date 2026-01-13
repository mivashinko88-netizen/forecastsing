# services/cache.py - Simple in-memory cache with TTL
import time
from typing import Any, Optional
from functools import wraps

class SimpleCache:
    """Simple in-memory cache with TTL (time-to-live)"""

    def __init__(self):
        self._cache = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self._cache:
            value, expires_at = self._cache[key]
            if time.time() < expires_at:
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl_seconds: int = 300):
        """Set value in cache with TTL (default 5 minutes)"""
        expires_at = time.time() + ttl_seconds
        self._cache[key] = (value, expires_at)

    def clear(self):
        """Clear all cached values"""
        self._cache = {}


# Global cache instance
cache = SimpleCache()


def cached(ttl_seconds: int = 300):
    """Decorator to cache async function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"

            # Check cache
            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value

            # Call function and cache result
            result = await func(*args, **kwargs)
            cache.set(key, result, ttl_seconds)
            return result

        return wrapper
    return decorator
