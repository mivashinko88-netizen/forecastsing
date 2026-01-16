# db_models.py
from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, Text, ForeignKey, Date
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    google_id = Column(String, unique=True, nullable=True)  # Nullable for local auth users
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    picture_url = Column(String, nullable=True)  # For profile pictures only
    password_hash = Column(String, nullable=True)  # Bcrypt hashed password
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_login = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)

    businesses = relationship("Business", back_populates="user", cascade="all, delete-orphan")


class Business(Base):
    __tablename__ = "businesses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    business_type = Column(String, nullable=False)  # 'restaurant', 'retail', 'service', 'fashion', 'electronics', etc.

    # Online store flag
    is_online = Column(Boolean, default=False)

    # Physical store location (nullable for online stores)
    address = Column(String, nullable=True)  # Street address
    city = Column(String, nullable=True)
    state = Column(String, nullable=True)
    zipcode = Column(String, nullable=True)
    country = Column(String, default="US")
    timezone = Column(String, default="America/New_York")
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    # Store hours (stored as HH:MM format, e.g., "09:00")
    open_time = Column(String, default="09:00")
    close_time = Column(String, default="21:00")
    # Days open (stored as comma-separated, e.g., "mon,tue,wed,thu,fri,sat,sun")
    days_open = Column(String, default="mon,tue,wed,thu,fri,sat,sun")

    # Online store marketing channels (comma-separated)
    marketing_channels = Column(String, nullable=True)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    setup_complete = Column(Boolean, default=False)

    user = relationship("User", back_populates="businesses")
    trained_models = relationship("TrainedModel", back_populates="business", cascade="all, delete-orphan")
    uploads = relationship("Upload", back_populates="business", cascade="all, delete-orphan")
    integrations = relationship("Integration", back_populates="business", cascade="all, delete-orphan")


class TrainedModel(Base):
    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True)
    business_id = Column(Integer, ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False)
    model_name = Column(String, nullable=False)
    model_path = Column(String, nullable=False)
    training_rows = Column(Integer)
    test_rows = Column(Integer)
    train_mae = Column(Float)
    test_mae = Column(Float)
    train_mape = Column(Float)
    test_mape = Column(Float)
    feature_importance = Column(Text)  # JSON string
    data_start_date = Column(Date)
    data_end_date = Column(Date)
    items_trained = Column(Text)  # JSON array
    external_data_used = Column(Text)  # JSON
    created_at = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True)

    business = relationship("Business", back_populates="trained_models")
    predictions = relationship("Prediction", back_populates="model", cascade="all, delete-orphan")


class Upload(Base):
    __tablename__ = "uploads"

    id = Column(Integer, primary_key=True, index=True)
    business_id = Column(Integer, ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String, nullable=False)
    row_count = Column(Integer)
    column_mapping = Column(Text)  # JSON
    uploaded_at = Column(DateTime, default=func.now())
    processed = Column(Boolean, default=False)
    model_id = Column(Integer, ForeignKey("trained_models.id"), nullable=True)

    business = relationship("Business", back_populates="uploads")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("trained_models.id", ondelete="CASCADE"), nullable=False)
    prediction_date = Column(Date, nullable=False)
    item_name = Column(String)
    predicted_quantity = Column(Integer)
    actual_quantity = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=func.now())

    model = relationship("TrainedModel", back_populates="predictions")


class Integration(Base):
    """OAuth integrations for POS/e-commerce platforms"""
    __tablename__ = "integrations"

    id = Column(Integer, primary_key=True, index=True)
    business_id = Column(Integer, ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False)
    provider = Column(String, nullable=False)  # 'square', 'toast', 'clover', 'shopify'

    # OAuth tokens
    access_token = Column(Text, nullable=True)
    refresh_token = Column(Text, nullable=True)
    token_expires_at = Column(DateTime, nullable=True)

    # Provider-specific identifiers
    merchant_id = Column(String, nullable=True)  # Square/Clover merchant ID
    location_id = Column(String, nullable=True)  # Square location, Toast restaurant GUID
    shop_domain = Column(String, nullable=True)  # Shopify store domain

    # Connection status
    status = Column(String, default="pending")  # pending, connected, error, disconnected
    last_sync_at = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    business = relationship("Business", back_populates="integrations")
    sync_logs = relationship("SyncLog", back_populates="integration", cascade="all, delete-orphan")


class SyncLog(Base):
    """Track sync history for integrations"""
    __tablename__ = "sync_logs"

    id = Column(Integer, primary_key=True, index=True)
    integration_id = Column(Integer, ForeignKey("integrations.id", ondelete="CASCADE"), nullable=False)
    sync_type = Column(String, nullable=False)  # 'products', 'orders', 'full'
    status = Column(String, nullable=False)  # 'started', 'completed', 'failed'
    records_synced = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)

    integration = relationship("Integration", back_populates="sync_logs")


class SyncedProduct(Base):
    """Products synced from external integrations"""
    __tablename__ = "synced_products"

    id = Column(Integer, primary_key=True, index=True)
    business_id = Column(Integer, ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False)
    integration_id = Column(Integer, ForeignKey("integrations.id", ondelete="CASCADE"), nullable=False)

    external_id = Column(String, nullable=False)  # ID from provider
    name = Column(String, nullable=False)
    category = Column(String, nullable=True)
    sku = Column(String, nullable=True)
    price = Column(Float, nullable=True)
    is_active = Column(Boolean, default=True)

    raw_data = Column(Text, nullable=True)  # JSON of full provider response
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class SyncedOrder(Base):
    """Orders synced from external integrations"""
    __tablename__ = "synced_orders"

    id = Column(Integer, primary_key=True, index=True)
    business_id = Column(Integer, ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False)
    integration_id = Column(Integer, ForeignKey("integrations.id", ondelete="CASCADE"), nullable=False)

    external_id = Column(String, nullable=False)
    order_date = Column(DateTime, nullable=False)
    total_amount = Column(Float, nullable=True)
    item_count = Column(Integer, nullable=True)

    raw_data = Column(Text, nullable=True)  # JSON of full provider response
    created_at = Column(DateTime, default=func.now())
