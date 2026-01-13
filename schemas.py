# schemas.py
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime, date


# Auth schemas
class GoogleAuthRequest(BaseModel):
    token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    picture_url: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AuthResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserResponse


# Business schemas
class BusinessCreate(BaseModel):
    name: str
    business_type: str  # 'restaurant', 'retail', 'service', 'fashion', 'electronics', etc.
    is_online: bool = False
    city: Optional[str] = None
    state: Optional[str] = None
    zipcode: Optional[str] = None
    country: str = "US"
    timezone: str = "America/New_York"
    open_time: str = "09:00"
    close_time: str = "21:00"
    days_open: str = "mon,tue,wed,thu,fri,sat,sun"
    marketing_channels: Optional[str] = None


class BusinessUpdate(BaseModel):
    name: Optional[str] = None
    business_type: Optional[str] = None
    is_online: Optional[bool] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zipcode: Optional[str] = None
    country: Optional[str] = None
    timezone: Optional[str] = None
    open_time: Optional[str] = None
    close_time: Optional[str] = None
    days_open: Optional[str] = None
    marketing_channels: Optional[str] = None


class BusinessResponse(BaseModel):
    id: int
    user_id: int
    name: str
    business_type: str
    is_online: bool = False
    city: Optional[str] = None
    state: Optional[str] = None
    zipcode: Optional[str] = None
    country: Optional[str] = None
    timezone: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    open_time: Optional[str] = "09:00"
    close_time: Optional[str] = "21:00"
    days_open: Optional[str] = "mon,tue,wed,thu,fri,sat,sun"
    marketing_channels: Optional[str] = None
    setup_complete: bool
    created_at: datetime

    class Config:
        from_attributes = True


# Model schemas
class TrainedModelResponse(BaseModel):
    id: int
    business_id: int
    model_name: str
    training_rows: Optional[int] = None
    test_rows: Optional[int] = None
    train_mae: Optional[float] = None
    test_mae: Optional[float] = None
    train_mape: Optional[float] = None
    test_mape: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    data_start_date: Optional[date] = None
    data_end_date: Optional[date] = None
    items_trained: Optional[List[str]] = None
    external_data_used: Optional[Dict[str, int]] = None
    created_at: datetime
    is_active: bool

    class Config:
        from_attributes = True


class TrainingResult(BaseModel):
    model_id: int
    train_mae: float
    test_mae: float
    train_mape: float
    test_mape: float
    feature_importance: Dict[str, float]
    training_rows: int
    test_rows: int
    external_data: Dict[str, int]


# Prediction schemas
class PredictionRequest(BaseModel):
    days: int = 7
    items: Optional[List[str]] = None


class PredictionItem(BaseModel):
    date: str
    item_name: str
    predicted_quantity: int
    factors: Optional[Dict[str, Any]] = None


class PredictionResponse(BaseModel):
    model_id: int
    business_name: str
    predictions: List[PredictionItem]
    date_range: Dict[str, str]
    factors_in_range: Dict[str, List[str]]


# Upload schemas
class UploadResponse(BaseModel):
    id: int
    filename: str
    row_count: int
    their_columns: List[str]
    suggested_mapping: Dict[str, str]
    preview: List[Dict[str, Any]]


class ColumnMappingRequest(BaseModel):
    mapping: Dict[str, str]


# Integration schemas
class IntegrationCreate(BaseModel):
    provider: str  # 'square', 'toast', 'clover', 'shopify'
    access_token: str
    refresh_token: Optional[str] = None
    merchant_id: Optional[str] = None
    location_id: Optional[str] = None
    shop_domain: Optional[str] = None


class IntegrationResponse(BaseModel):
    id: int
    business_id: int
    provider: str
    status: str
    merchant_id: Optional[str] = None
    location_id: Optional[str] = None
    shop_domain: Optional[str] = None
    last_sync_at: Optional[datetime] = None
    last_error: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class SyncLogResponse(BaseModel):
    id: int
    integration_id: int
    sync_type: str
    status: str
    records_synced: int
    error_message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SyncRequest(BaseModel):
    sync_type: str = "full"  # 'products', 'orders', 'full'
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class SyncSummaryResponse(BaseModel):
    order_count: int
    product_count: int
    date_range: Optional[Dict[str, str]] = None
    total_revenue: float


class IntegrationProviderInfo(BaseModel):
    id: str
    name: str
    description: str
    icon: str
    requires_shop_domain: bool
