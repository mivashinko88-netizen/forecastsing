"""OAuth state management utilities"""
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional
from services.cache import SimpleCache

# Use the existing cache service
_state_cache = SimpleCache()

# State token expiry (5 minutes)
STATE_TTL_SECONDS = 300


class OAuthStateManager:
    """Manage OAuth state tokens for CSRF protection"""

    @staticmethod
    def generate_state(business_id: int, provider: str, user_id: int) -> str:
        """
        Generate and store OAuth state token

        Args:
            business_id: ID of the business connecting the integration
            provider: Integration provider name (square, toast, etc.)
            user_id: ID of the user initiating the OAuth flow

        Returns:
            State token string
        """
        state = secrets.token_urlsafe(32)
        _state_cache.set(
            f"oauth_state:{state}",
            {
                "business_id": business_id,
                "provider": provider,
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat()
            },
            ttl_seconds=STATE_TTL_SECONDS
        )
        return state

    @staticmethod
    def validate_state(state: str) -> Optional[Dict]:
        """
        Validate and consume state token

        Args:
            state: State token from OAuth callback

        Returns:
            Dict with business_id, provider, user_id if valid
            None if invalid or expired
        """
        cache_key = f"oauth_state:{state}"
        data = _state_cache.get(cache_key)

        if data and data.get("used") is not True:
            # Check if not expired (double-check beyond cache TTL)
            created_at = datetime.fromisoformat(data["created_at"])
            if datetime.utcnow() - created_at < timedelta(seconds=STATE_TTL_SECONDS):
                # Mark as used to prevent reuse (set TTL to 1 second to expire quickly)
                _state_cache.set(cache_key, {"used": True}, ttl_seconds=1)
                return {
                    "business_id": data["business_id"],
                    "provider": data["provider"],
                    "user_id": data["user_id"]
                }

        return None

    @staticmethod
    def cleanup_expired_states():
        """Clean up expired state tokens (called periodically if needed)"""
        # The SimpleCache handles TTL automatically, but this method
        # can be used for manual cleanup if needed
        pass
