# routers/integrations.py - POS/e-commerce integration endpoints
import os
from typing import List, Optional
from datetime import datetime, date
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database import get_db
from db_models import User, Business, Integration, SyncLog, SyncedProduct, SyncedOrder
from auth import get_current_user
from services.integrations import (
    OAuthStateManager,
    SyncService,
    DataTransformer,
    get_integration_class
)

router = APIRouter(prefix="/integrations", tags=["Integrations"])


# Response models
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


class AuthUrlResponse(BaseModel):
    authorization_url: str
    state: str


class SyncSummaryResponse(BaseModel):
    order_count: int
    product_count: int
    date_range: Optional[dict] = None
    total_revenue: float


# Valid providers
VALID_PROVIDERS = ["square", "toast", "clover", "shopify"]


def get_redirect_uri(provider: str) -> str:
    """Get OAuth redirect URI for a provider"""
    base_url = os.getenv("OAUTH_REDIRECT_BASE_URL", "http://localhost:8000")
    return f"{base_url}/api/integrations/{provider}/callback"


def verify_business_ownership(
    business_id: int,
    user: User,
    db: Session
) -> Business:
    """Verify user owns the business"""
    business = db.query(Business).filter(
        Business.id == business_id,
        Business.user_id == user.id
    ).first()

    if not business:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Business not found"
        )

    return business


def verify_integration_ownership(
    integration_id: int,
    user: User,
    db: Session
) -> Integration:
    """Verify user owns the integration"""
    integration = db.query(Integration).filter(
        Integration.id == integration_id
    ).first()

    if not integration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Integration not found"
        )

    # Verify business ownership
    verify_business_ownership(integration.business_id, user, db)

    return integration


# ============ OAuth Flow Endpoints ============

@router.get("/{provider}/authorize", response_model=AuthUrlResponse)
async def initiate_oauth(
    provider: str,
    business_id: int = Query(..., description="Business ID to connect"),
    shop_domain: Optional[str] = Query(None, description="Shopify store domain (required for Shopify)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Step 1: Initiate OAuth flow
    Returns authorization URL for frontend to redirect/popup
    """
    if provider not in VALID_PROVIDERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid provider. Must be one of: {', '.join(VALID_PROVIDERS)}"
        )

    # Verify business ownership
    verify_business_ownership(business_id, current_user, db)

    # Check for existing integration
    existing = db.query(Integration).filter(
        Integration.business_id == business_id,
        Integration.provider == provider
    ).first()

    if existing and existing.status == "connected":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{provider.capitalize()} is already connected"
        )

    # Shopify requires shop domain
    if provider == "shopify" and not shop_domain:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="shop_domain is required for Shopify"
        )

    # Generate state token
    state = OAuthStateManager.generate_state(business_id, provider, current_user.id)

    # Get authorization URL
    redirect_uri = get_redirect_uri(provider)
    integration_class = get_integration_class(provider)

    if provider == "shopify":
        auth_url = integration_class.get_authorization_url(redirect_uri, state, shop_domain)
    else:
        auth_url = integration_class.get_authorization_url(redirect_uri, state)

    return AuthUrlResponse(authorization_url=auth_url, state=state)


@router.get("/{provider}/callback")
async def oauth_callback(
    provider: str,
    code: Optional[str] = Query(None),
    state: str = Query(...),
    error: Optional[str] = Query(None),
    error_description: Optional[str] = Query(None),
    shop: Optional[str] = Query(None, description="Shopify shop domain"),
    db: Session = Depends(get_db)
):
    """
    Step 2: OAuth callback handler
    Exchanges code for tokens and stores integration
    Redirects back to frontend with success/error status
    """
    frontend_base = "/frontend/pages/setup"

    # Handle OAuth errors
    if error:
        error_msg = error_description or error
        return RedirectResponse(
            url=f"{frontend_base}/integrations.html?error={error_msg}&provider={provider}"
        )

    # Validate state
    state_data = OAuthStateManager.validate_state(state)
    if not state_data:
        return RedirectResponse(
            url=f"{frontend_base}/integrations.html?error=Invalid+or+expired+state&provider={provider}"
        )

    if not code:
        return RedirectResponse(
            url=f"{frontend_base}/integrations.html?error=No+authorization+code&provider={provider}"
        )

    business_id = state_data["business_id"]

    try:
        # Exchange code for tokens
        redirect_uri = get_redirect_uri(provider)
        integration_class = get_integration_class(provider)

        if provider == "shopify":
            token_data = await integration_class.exchange_code_for_tokens(
                code, redirect_uri, shop_domain=shop
            )
        else:
            token_data = await integration_class.exchange_code_for_tokens(code, redirect_uri)

        # Calculate token expiry
        token_expires_at = None
        if token_data.get("expires_in"):
            if isinstance(token_data["expires_in"], int):
                from datetime import timedelta
                token_expires_at = datetime.utcnow() + timedelta(seconds=token_data["expires_in"])
            else:
                # ISO date string (Square)
                token_expires_at = datetime.fromisoformat(
                    token_data["expires_in"].replace("Z", "+00:00")
                )

        # Check for existing integration
        existing = db.query(Integration).filter(
            Integration.business_id == business_id,
            Integration.provider == provider
        ).first()

        if existing:
            # Update existing
            existing.access_token = token_data["access_token"]
            existing.refresh_token = token_data.get("refresh_token")
            existing.token_expires_at = token_expires_at
            existing.merchant_id = token_data.get("merchant_id")
            existing.location_id = token_data.get("location_id")
            existing.shop_domain = token_data.get("shop_domain") or shop
            existing.status = "connected"
            existing.last_error = None
            integration = existing
        else:
            # Create new integration
            integration = Integration(
                business_id=business_id,
                provider=provider,
                access_token=token_data["access_token"],
                refresh_token=token_data.get("refresh_token"),
                token_expires_at=token_expires_at,
                merchant_id=token_data.get("merchant_id"),
                location_id=token_data.get("location_id"),
                shop_domain=token_data.get("shop_domain") or shop,
                status="connected"
            )
            db.add(integration)

        db.commit()

        return RedirectResponse(
            url=f"{frontend_base}/integrations.html?success=true&provider={provider}"
        )

    except Exception as e:
        return RedirectResponse(
            url=f"{frontend_base}/integrations.html?error={str(e)}&provider={provider}"
        )


# ============ Integration Management ============

@router.get("/businesses/{business_id}", response_model=List[IntegrationResponse])
async def list_integrations(
    business_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all integrations for a business"""
    verify_business_ownership(business_id, current_user, db)

    integrations = db.query(Integration).filter(
        Integration.business_id == business_id
    ).all()

    return integrations


@router.get("/{integration_id}", response_model=IntegrationResponse)
async def get_integration(
    integration_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get integration details and status"""
    integration = verify_integration_ownership(integration_id, current_user, db)
    return integration


@router.delete("/{integration_id}", status_code=status.HTTP_204_NO_CONTENT)
async def disconnect_integration(
    integration_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Disconnect and remove integration"""
    integration = verify_integration_ownership(integration_id, current_user, db)
    db.delete(integration)
    db.commit()


@router.post("/{integration_id}/test")
async def test_integration(
    integration_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Test if integration connection is still valid"""
    integration = verify_integration_ownership(integration_id, current_user, db)

    sync_service = SyncService(db)
    is_valid = await sync_service.test_integration(integration)

    return {"valid": is_valid, "status": integration.status}


# ============ Data Sync Endpoints ============

@router.post("/{integration_id}/sync", response_model=SyncLogResponse)
async def trigger_sync(
    integration_id: int,
    sync_type: str = Query("full", description="'products', 'orders', or 'full'"),
    start_date: Optional[str] = Query(None, description="Start date for orders (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for orders (YYYY-MM-DD)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Trigger data sync from provider"""
    integration = verify_integration_ownership(integration_id, current_user, db)

    if sync_type not in ("products", "orders", "full"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="sync_type must be 'products', 'orders', or 'full'"
        )

    # Parse dates if provided
    parsed_start = None
    parsed_end = None
    if start_date:
        try:
            parsed_start = datetime.strptime(start_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format")
    if end_date:
        try:
            parsed_end = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format")

    sync_service = SyncService(db)
    sync_log = await sync_service.sync_integration(
        integration,
        sync_type=sync_type,
        start_date=parsed_start,
        end_date=parsed_end
    )

    return sync_log


@router.get("/{integration_id}/sync-history", response_model=List[SyncLogResponse])
async def get_sync_history(
    integration_id: int,
    limit: int = Query(10, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get sync history for an integration"""
    verify_integration_ownership(integration_id, current_user, db)

    sync_logs = db.query(SyncLog).filter(
        SyncLog.integration_id == integration_id
    ).order_by(SyncLog.started_at.desc()).limit(limit).all()

    return sync_logs


@router.get("/businesses/{business_id}/synced-data/summary", response_model=SyncSummaryResponse)
async def get_synced_data_summary(
    business_id: int,
    integration_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get summary of synced data for a business"""
    verify_business_ownership(business_id, current_user, db)

    if integration_id:
        verify_integration_ownership(integration_id, current_user, db)

    transformer = DataTransformer(db)
    summary = transformer.get_sync_summary(business_id, integration_id)

    return SyncSummaryResponse(**summary)


@router.post("/businesses/{business_id}/export-csv")
async def export_synced_to_csv(
    business_id: int,
    integration_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Export synced orders to CSV format for training"""
    verify_business_ownership(business_id, current_user, db)

    if integration_id:
        verify_integration_ownership(integration_id, current_user, db)

    transformer = DataTransformer(db)
    csv_data = transformer.orders_to_csv(business_id, integration_id)

    if not csv_data or csv_data.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No synced data to export"
        )

    from fastapi.responses import Response
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=synced_orders_{business_id}.csv"
        }
    )


@router.get("/providers")
async def list_providers():
    """List available integration providers with info"""
    return [
        {
            "id": "square",
            "name": "Square",
            "description": "POS and payment processing for restaurants and retail",
            "icon": "square",
            "requires_shop_domain": False
        },
        {
            "id": "toast",
            "name": "Toast",
            "description": "Restaurant-specific POS system",
            "icon": "toast",
            "requires_shop_domain": False
        },
        {
            "id": "clover",
            "name": "Clover",
            "description": "POS system for various industries",
            "icon": "clover",
            "requires_shop_domain": False
        },
        {
            "id": "shopify",
            "name": "Shopify",
            "description": "E-commerce platform for online stores",
            "icon": "shopify",
            "requires_shop_domain": True
        }
    ]
