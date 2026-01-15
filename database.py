# database.py - Database configuration with PostgreSQL and SQLite support
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path
import os

# Import settings - but handle circular import for Alembic
try:
    from settings import get_settings
    settings = get_settings()
    DATABASE_URL = settings.DATABASE_URL
except ImportError:
    # Fallback for Alembic or other tools that import before settings
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/forecasting.db")

# Create data directory for SQLite if needed
if DATABASE_URL.startswith("sqlite"):
    DATA_DIR = Path(__file__).resolve().parent / "data"
    DATA_DIR.mkdir(exist_ok=True)
    # Ensure the URL points to the correct path
    if DATABASE_URL == "sqlite:///./data/forecasting.db":
        DATABASE_URL = f"sqlite:///{DATA_DIR}/forecasting.db"

# Database engine configuration
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False  # Needed for SQLite with FastAPI

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,  # Handle stale connections for PostgreSQL
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency for getting database sessions"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    from db_models import User, Business, TrainedModel, Upload, Prediction, Integration, SyncLog, SyncedProduct, SyncedOrder
    Base.metadata.create_all(bind=engine)
