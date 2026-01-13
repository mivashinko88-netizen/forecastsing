"""Square POS integration"""
import os
from typing import Dict, List, Any, Optional
from datetime import date, datetime
from urllib.parse import urlencode
import httpx

from .base import BaseIntegration


class SquareIntegration(BaseIntegration):
    """Square POS integration for catalog and orders"""

    PROVIDER_NAME = "square"
    BASE_URL = "https://connect.squareup.com/v2"
    OAUTH_URL = "https://connect.squareup.com/oauth2"

    # Sandbox URLs for development
    SANDBOX_BASE_URL = "https://connect.squareupsandbox.com/v2"
    SANDBOX_OAUTH_URL = "https://connect.squareupsandbox.com/oauth2"

    @classmethod
    def _get_base_url(cls) -> str:
        """Get base URL based on environment"""
        if os.getenv("SQUARE_ENVIRONMENT", "sandbox") == "production":
            return cls.BASE_URL
        return cls.SANDBOX_BASE_URL

    @classmethod
    def _get_oauth_url(cls) -> str:
        """Get OAuth URL based on environment"""
        if os.getenv("SQUARE_ENVIRONMENT", "sandbox") == "production":
            return cls.OAUTH_URL
        return cls.SANDBOX_OAUTH_URL

    @classmethod
    def get_required_scopes(cls) -> List[str]:
        return [
            "ITEMS_READ",
            "ORDERS_READ",
            "MERCHANT_PROFILE_READ"
        ]

    @classmethod
    def get_authorization_url(cls, redirect_uri: str, state: str) -> str:
        """Generate Square OAuth authorization URL"""
        params = {
            "client_id": os.getenv("SQUARE_APP_ID"),
            "scope": " ".join(cls.get_required_scopes()),
            "session": "false",
            "state": state,
            "redirect_uri": redirect_uri
        }
        return f"{cls._get_oauth_url()}/authorize?{urlencode(params)}"

    @classmethod
    async def exchange_code_for_tokens(cls, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{cls._get_oauth_url()}/token",
                json={
                    "client_id": os.getenv("SQUARE_APP_ID"),
                    "client_secret": os.getenv("SQUARE_APP_SECRET"),
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri
                },
                headers={"Content-Type": "application/json"},
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()

            return {
                "access_token": data["access_token"],
                "refresh_token": data.get("refresh_token"),
                "expires_in": data.get("expires_at"),  # Square returns ISO date
                "merchant_id": data.get("merchant_id"),
                "token_type": data.get("token_type", "bearer")
            }

    async def refresh_access_token(self) -> Dict[str, Any]:
        """Refresh Square access token"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._get_oauth_url()}/token",
                json={
                    "client_id": os.getenv("SQUARE_APP_ID"),
                    "client_secret": os.getenv("SQUARE_APP_SECRET"),
                    "refresh_token": self.refresh_token,
                    "grant_type": "refresh_token"
                },
                headers={"Content-Type": "application/json"},
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()

            return {
                "access_token": data["access_token"],
                "refresh_token": data.get("refresh_token", self.refresh_token),
                "expires_in": data.get("expires_at")
            }

    async def _get_locations(self) -> List[Dict]:
        """Get merchant locations"""
        data = await self._make_request("GET", f"{self._get_base_url()}/locations")
        return data.get("locations", [])

    async def fetch_products(self) -> List[Dict[str, Any]]:
        """Fetch catalog items from Square"""
        products = []
        cursor = None

        while True:
            params = {"types": "ITEM"}
            if cursor:
                params["cursor"] = cursor

            data = await self._make_request(
                "GET",
                f"{self._get_base_url()}/catalog/list",
                params=params
            )

            for obj in data.get("objects", []):
                if obj.get("type") == "ITEM":
                    item_data = obj.get("item_data", {})

                    # Get price from first variation
                    price = None
                    variations = item_data.get("variations", [])
                    if variations:
                        first_var = variations[0].get("item_variation_data", {})
                        price_money = first_var.get("price_money", {})
                        if price_money:
                            # Square uses cents
                            price = price_money.get("amount", 0) / 100

                    products.append({
                        "external_id": obj["id"],
                        "name": item_data.get("name", "Unknown"),
                        "category": item_data.get("category_id"),
                        "sku": variations[0].get("item_variation_data", {}).get("sku") if variations else None,
                        "price": price,
                        "raw_data": obj
                    })

            cursor = data.get("cursor")
            if not cursor:
                break

        return products

    async def fetch_orders(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Fetch orders from Square"""
        orders = []

        # Get location IDs
        locations = await self._get_locations()
        location_ids = [loc["id"] for loc in locations]

        if not location_ids:
            return orders

        cursor = None

        while True:
            body = {
                "location_ids": location_ids,
                "query": {
                    "filter": {
                        "date_time_filter": {
                            "created_at": {
                                "start_at": f"{start_date}T00:00:00Z",
                                "end_at": f"{end_date}T23:59:59Z"
                            }
                        },
                        "state_filter": {
                            "states": ["COMPLETED"]
                        }
                    },
                    "sort": {
                        "sort_field": "CREATED_AT",
                        "sort_order": "ASC"
                    }
                }
            }

            if cursor:
                body["cursor"] = cursor

            data = await self._make_request(
                "POST",
                f"{self._get_base_url()}/orders/search",
                json=body
            )

            for order in data.get("orders", []):
                line_items = []
                for item in order.get("line_items", []):
                    line_items.append({
                        "name": item.get("name", "Unknown"),
                        "quantity": int(item.get("quantity", "1")),
                        "unit_price": item.get("base_price_money", {}).get("amount", 0) / 100,
                        "total": item.get("total_money", {}).get("amount", 0) / 100
                    })

                total_money = order.get("total_money", {})
                orders.append({
                    "external_id": order["id"],
                    "order_date": datetime.fromisoformat(order["created_at"].replace("Z", "+00:00")),
                    "total_amount": total_money.get("amount", 0) / 100,
                    "item_count": len(line_items),
                    "line_items": line_items,
                    "raw_data": order
                })

            cursor = data.get("cursor")
            if not cursor:
                break

        return orders

    async def test_connection(self) -> bool:
        """Test Square connection by fetching merchant info"""
        try:
            await self._get_locations()
            return True
        except Exception:
            return False
