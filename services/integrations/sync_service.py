"""Data synchronization service for integrations"""
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session

from db_models import Integration, SyncLog, SyncedProduct, SyncedOrder
from .base import BaseIntegration
from .square import SquareIntegration
from .shopify import ShopifyIntegration
from .toast import ToastIntegration
from .clover import CloverIntegration


def get_integration_class(provider: str) -> type:
    """Get integration class by provider name"""
    providers = {
        "square": SquareIntegration,
        "shopify": ShopifyIntegration,
        "toast": ToastIntegration,
        "clover": CloverIntegration
    }
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}")
    return providers[provider]


def get_integration_client(integration: Integration) -> BaseIntegration:
    """Create integration client from database record"""
    cls = get_integration_class(integration.provider)
    return cls(
        access_token=integration.access_token,
        refresh_token=integration.refresh_token,
        merchant_id=integration.merchant_id,
        location_id=integration.location_id,
        shop_domain=integration.shop_domain
    )


class SyncService:
    """Orchestrates data synchronization from integrations"""

    def __init__(self, db: Session):
        self.db = db

    def _token_needs_refresh(self, integration: Integration) -> bool:
        """Check if access token needs refresh"""
        if not integration.token_expires_at:
            return False
        # Refresh 5 minutes before expiry
        buffer = timedelta(minutes=5)
        return datetime.utcnow() + buffer >= integration.token_expires_at

    async def _refresh_token(self, integration: Integration) -> bool:
        """Attempt to refresh access token"""
        if not integration.refresh_token:
            return False

        try:
            client = get_integration_client(integration)
            new_tokens = await client.refresh_access_token()

            integration.access_token = new_tokens["access_token"]
            if new_tokens.get("refresh_token"):
                integration.refresh_token = new_tokens["refresh_token"]
            if new_tokens.get("expires_in"):
                if isinstance(new_tokens["expires_in"], int):
                    integration.token_expires_at = datetime.utcnow() + timedelta(
                        seconds=new_tokens["expires_in"]
                    )
                else:
                    # ISO date string (Square)
                    integration.token_expires_at = datetime.fromisoformat(
                        new_tokens["expires_in"].replace("Z", "+00:00")
                    )

            self.db.commit()
            return True

        except Exception as e:
            integration.status = "error"
            integration.last_error = f"Token refresh failed: {str(e)}"
            self.db.commit()
            return False

    async def _store_products(
        self,
        integration: Integration,
        products: List[Dict[str, Any]]
    ) -> int:
        """Store products in database"""
        count = 0

        for product in products:
            # Check if product exists
            existing = self.db.query(SyncedProduct).filter(
                SyncedProduct.integration_id == integration.id,
                SyncedProduct.external_id == product["external_id"]
            ).first()

            if existing:
                # Update existing
                existing.name = product["name"]
                existing.category = product.get("category")
                existing.sku = product.get("sku")
                existing.price = product.get("price")
                existing.raw_data = json.dumps(product.get("raw_data", {}))
                existing.updated_at = datetime.utcnow()
            else:
                # Create new
                new_product = SyncedProduct(
                    business_id=integration.business_id,
                    integration_id=integration.id,
                    external_id=product["external_id"],
                    name=product["name"],
                    category=product.get("category"),
                    sku=product.get("sku"),
                    price=product.get("price"),
                    raw_data=json.dumps(product.get("raw_data", {}))
                )
                self.db.add(new_product)
                count += 1

        self.db.commit()
        return count

    async def _store_orders(
        self,
        integration: Integration,
        orders: List[Dict[str, Any]]
    ) -> int:
        """Store orders in database"""
        count = 0

        for order in orders:
            # Check if order exists
            existing = self.db.query(SyncedOrder).filter(
                SyncedOrder.integration_id == integration.id,
                SyncedOrder.external_id == order["external_id"]
            ).first()

            if not existing:
                new_order = SyncedOrder(
                    business_id=integration.business_id,
                    integration_id=integration.id,
                    external_id=order["external_id"],
                    order_date=order["order_date"],
                    total_amount=order.get("total_amount"),
                    item_count=order.get("item_count"),
                    raw_data=json.dumps(order.get("raw_data", {}), default=str)
                )
                self.db.add(new_order)
                count += 1

        self.db.commit()
        return count

    async def sync_integration(
        self,
        integration: Integration,
        sync_type: str = "full",
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> SyncLog:
        """
        Execute sync operation for an integration

        Args:
            integration: Integration database record
            sync_type: 'products', 'orders', or 'full'
            start_date: Start date for orders (default: 365 days ago)
            end_date: End date for orders (default: today)

        Returns:
            SyncLog record with results
        """
        # Create sync log
        sync_log = SyncLog(
            integration_id=integration.id,
            sync_type=sync_type,
            status="started"
        )
        self.db.add(sync_log)
        self.db.commit()

        try:
            # Check/refresh token
            if self._token_needs_refresh(integration):
                refreshed = await self._refresh_token(integration)
                if not refreshed and integration.refresh_token:
                    raise Exception("Token refresh failed, re-authorization required")

            # Get provider client
            client = get_integration_client(integration)

            records_synced = 0

            if sync_type in ("products", "full"):
                products = await client.fetch_products()
                records_synced += await self._store_products(integration, products)

            if sync_type in ("orders", "full"):
                # Default to last 365 days if no dates specified
                if not start_date:
                    start_date = date.today() - timedelta(days=365)
                if not end_date:
                    end_date = date.today()

                orders = await client.fetch_orders(start_date, end_date)
                records_synced += await self._store_orders(integration, orders)

            # Update sync log
            sync_log.status = "completed"
            sync_log.records_synced = records_synced
            sync_log.completed_at = datetime.utcnow()

            # Update integration
            integration.last_sync_at = datetime.utcnow()
            integration.status = "connected"
            integration.last_error = None

        except Exception as e:
            sync_log.status = "failed"
            sync_log.error_message = str(e)
            sync_log.completed_at = datetime.utcnow()

            integration.status = "error"
            integration.last_error = str(e)

        self.db.commit()
        return sync_log

    async def test_integration(self, integration: Integration) -> bool:
        """Test if integration connection is valid"""
        try:
            client = get_integration_client(integration)
            return await client.test_connection()
        except Exception:
            return False
