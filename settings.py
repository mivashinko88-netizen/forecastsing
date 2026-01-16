"""
Centralized application settings using Pydantic.
All configuration is loaded from environment variables.
"""
from pydantic_settings import BaseSettings
from typing import Optional, List
from functools import lru_cache
import json


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "TrucastAI"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"  # development, staging, production

    # Database
    DATABASE_URL: str = "sqlite:///./data/forecasting.db"

    # Security
    JWT_SECRET_KEY: str = "local-dev-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # CORS - accepts JSON string or list
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:8000"]

    # OAuth Base URL (for callbacks)
    OAUTH_REDIRECT_BASE_URL: str = "http://localhost:8000"

    # Square Integration
    SQUARE_APP_ID: Optional[str] = None
    SQUARE_APP_SECRET: Optional[str] = None
    SQUARE_ENVIRONMENT: str = "sandbox"

    # Toast Integration
    TOAST_CLIENT_ID: Optional[str] = None
    TOAST_CLIENT_SECRET: Optional[str] = None
    TOAST_ENVIRONMENT: str = "sandbox"

    # Clover Integration
    CLOVER_APP_ID: Optional[str] = None
    CLOVER_APP_SECRET: Optional[str] = None
    CLOVER_ENVIRONMENT: str = "sandbox"

    # Shopify Integration
    SHOPIFY_API_KEY: Optional[str] = None
    SHOPIFY_API_SECRET: Optional[str] = None

    # Sentry Error Monitoring
    SENTRY_DSN: Optional[str] = None

    # OpenRouter AI
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_SITE_URL: str = "https://trucastai.com"

    # Scheduler
    SYNC_SCHEDULE_HOUR: int = 2  # 2 AM UTC daily sync
    SYNC_ENABLED: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str):
            if field_name == "CORS_ORIGINS":
                # Handle JSON string from environment
                if raw_val.startswith("["):
                    return json.loads(raw_val)
                # Handle comma-separated string
                return [origin.strip() for origin in raw_val.split(",")]
            return raw_val


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
