"""Clover POS integration"""
import os
from typing import Dict, List, Any, Optional
from datetime import date, datetime
from urllib.parse import urlencode
import httpx

from .base import BaseIntegration


class CloverIntegration(BaseIntegration):
    """Clover POS integration for inventory and orders"""

    PROVIDER_NAME = "clover"
    BASE_URL = "https://api.clover.com/v3"
    OAUTH_URL = "https://clover.com/oauth"

    # Sandbox URLs
    SANDBOX_BASE_URL = "https://apisandbox.dev.clover.com/v3"
    SANDBOX_OAUTH_URL = "https://sandbox.dev.clover.com/oauth"

    @classmethod
    def _get_base_url(cls) -> str:
        if os.getenv("CLOVER_ENVIRONMENT", "sandbox") == "production":
            return cls.BASE_URL
        return cls.SANDBOX_BASE_URL

    @classmethod
    def _get_oauth_url(cls) -> str:
        if os.getenv("CLOVER_ENVIRONMENT", "sandbox") == "production":
            return cls.OAUTH_URL
        return cls.SANDBOX_OAUTH_URL

    @classmethod
    def get_required_scopes(cls) -> List[str]:
        return [
            "INVENTORY_R",
            "ORDERS_R",
            "MERCHANT_R"
        ]

    @classmethod
    def get_authorization_url(cls, redirect_uri: str, state: str) -> str:
        """Generate Clover OAuth authorization URL"""
        params = {
            "client_id": os.getenv("CLOVER_APP_ID"),
            "state": state,
            "redirect_uri": redirect_uri
        }
        return f"{cls._get_oauth_url()}/authorize?{urlencode(params)}"

    @classmethod
    async def exchange_code_for_tokens(cls, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{cls._get_oauth_url()}/token",
                params={
                    "client_id": os.getenv("CLOVER_APP_ID"),
                    "client_secret": os.getenv("CLOVER_APP_SECRET"),
                    "code": code
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()

            return {
                "access_token": data["access_token"],
                "refresh_token": None,  # Clover doesn't provide refresh tokens by default
                "expires_in": None,  # Clover tokens don't expire
                "merchant_id": data.get("merchant_id")
            }

    async def refresh_access_token(self) -> Dict[str, Any]:
        """Clover tokens don't typically expire"""
        return {
            "access_token": self.access_token,
            "refresh_token": None,
            "expires_in": None
        }

    async def fetch_products(self) -> List[Dict[str, Any]]:
        """Fetch inventory items from Clover"""
        products = []

        if not self.merchant_id:
            return products

        offset = 0
        limit = 100

        while True:
            params = {
                "offset": offset,
                "limit": limit,
                "expand": "categories"
            }

            data = await self._make_request(
                "GET",
                f"{self._get_base_url()}/merchants/{self.merchant_id}/items",
                params=params
            )

            elements = data.get("elements", [])
            if not elements:
                break

            for item in elements:
                category = None
                if item.get("categories") and item["categories"].get("elements"):
                    category = item["categories"]["elements"][0].get("name")

                products.append({
                    "external_id": item.get("id", ""),
                    "name": item.get("name", "Unknown"),
                    "category": category,
                    "sku": item.get("sku"),
                    "price": item.get("price", 0) / 100,  # Clover uses cents
                    "raw_data": item
                })

            if len(elements) < limit:
                break
            offset += limit

        return products

    async def fetch_orders(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Fetch orders from Clover"""
        orders = []

        if not self.merchant_id:
            return orders

        # Convert dates to timestamps (Clover uses milliseconds)
        start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
        end_ts = int(datetime.combine(end_date, datetime.max.time()).timestamp() * 1000)

        offset = 0
        limit = 100

        while True:
            params = {
                "offset": offset,
                "limit": limit,
                "filter": f"createdTime>={start_ts}&createdTime<={end_ts}",
                "expand": "lineItems"
            }

            data = await self._make_request(
                "GET",
                f"{self._get_base_url()}/merchants/{self.merchant_id}/orders",
                params=params
            )

            elements = data.get("elements", [])
            if not elements:
                break

            for order in elements:
                line_items = []
                line_items_data = order.get("lineItems", {}).get("elements", [])

                for item in line_items_data:
                    line_items.append({
                        "name": item.get("name", "Unknown"),
                        "quantity": 1,  # Clover doesn't have quantity in line items
                        "unit_price": item.get("price", 0) / 100,
                        "total": item.get("price", 0) / 100
                    })

                created_time = order.get("createdTime", 0)
                order_date = datetime.fromtimestamp(created_time / 1000) if created_time else datetime.now()

                orders.append({
                    "external_id": order.get("id", ""),
                    "order_date": order_date,
                    "total_amount": order.get("total", 0) / 100,
                    "item_count": len(line_items),
                    "line_items": line_items,
                    "raw_data": order
                })

            if len(elements) < limit:
                break
            offset += limit

        return orders

    async def test_connection(self) -> bool:
        """Test Clover connection by fetching merchant info"""
        try:
            if self.merchant_id:
                await self._make_request(
                    "GET",
                    f"{self._get_base_url()}/merchants/{self.merchant_id}"
                )
            return True
        except Exception:
            return False
