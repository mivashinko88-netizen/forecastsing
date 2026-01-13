"""Toast POS integration for restaurants"""
import os
from typing import Dict, List, Any, Optional
from datetime import date, datetime
from urllib.parse import urlencode
import httpx

from .base import BaseIntegration


class ToastIntegration(BaseIntegration):
    """Toast POS integration for menu items and orders"""

    PROVIDER_NAME = "toast"
    BASE_URL = "https://ws-api.toasttab.com"
    OAUTH_URL = "https://ws-api.toasttab.com/authentication/v1"

    # Sandbox URLs
    SANDBOX_BASE_URL = "https://ws-sandbox-api.toasttab.com"
    SANDBOX_OAUTH_URL = "https://ws-sandbox-api.toasttab.com/authentication/v1"

    @classmethod
    def _get_base_url(cls) -> str:
        if os.getenv("TOAST_ENVIRONMENT", "sandbox") == "production":
            return cls.BASE_URL
        return cls.SANDBOX_BASE_URL

    @classmethod
    def _get_oauth_url(cls) -> str:
        if os.getenv("TOAST_ENVIRONMENT", "sandbox") == "production":
            return cls.OAUTH_URL
        return cls.SANDBOX_OAUTH_URL

    @classmethod
    def get_required_scopes(cls) -> List[str]:
        return [
            "menus:read",
            "orders:read"
        ]

    @classmethod
    def get_authorization_url(cls, redirect_uri: str, state: str) -> str:
        """Generate Toast OAuth authorization URL"""
        params = {
            "client_id": os.getenv("TOAST_CLIENT_ID"),
            "response_type": "code",
            "scope": " ".join(cls.get_required_scopes()),
            "state": state,
            "redirect_uri": redirect_uri
        }
        return f"{cls._get_oauth_url()}/oauth2/authorize?{urlencode(params)}"

    @classmethod
    async def exchange_code_for_tokens(cls, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{cls._get_oauth_url()}/oauth2/token",
                data={
                    "client_id": os.getenv("TOAST_CLIENT_ID"),
                    "client_secret": os.getenv("TOAST_CLIENT_SECRET"),
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()

            return {
                "access_token": data["access_token"],
                "refresh_token": data.get("refresh_token"),
                "expires_in": data.get("expires_in", 3600),
                "location_id": data.get("restaurantGuid"),  # Toast restaurant GUID
                "token_type": data.get("token_type", "bearer")
            }

    async def refresh_access_token(self) -> Dict[str, Any]:
        """Refresh Toast access token"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._get_oauth_url()}/oauth2/token",
                data={
                    "client_id": os.getenv("TOAST_CLIENT_ID"),
                    "client_secret": os.getenv("TOAST_CLIENT_SECRET"),
                    "refresh_token": self.refresh_token,
                    "grant_type": "refresh_token"
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()

            return {
                "access_token": data["access_token"],
                "refresh_token": data.get("refresh_token", self.refresh_token),
                "expires_in": data.get("expires_in", 3600)
            }

    def _get_headers(self) -> Dict[str, str]:
        """Get Toast-specific headers"""
        headers = super()._get_headers()
        if self.location_id:
            headers["Toast-Restaurant-External-ID"] = self.location_id
        return headers

    async def fetch_products(self) -> List[Dict[str, Any]]:
        """Fetch menu items from Toast"""
        products = []

        if not self.location_id:
            return products

        data = await self._make_request(
            "GET",
            f"{self._get_base_url()}/menus/v2/menus"
        )

        # Toast returns menus with groups and items
        for menu in data if isinstance(data, list) else []:
            for group in menu.get("groups", []):
                for item in group.get("items", []):
                    price = None
                    if item.get("pricing"):
                        price = item["pricing"].get("basePrice", 0)

                    products.append({
                        "external_id": item.get("guid", ""),
                        "name": item.get("name", "Unknown"),
                        "category": group.get("name"),
                        "sku": item.get("sku"),
                        "price": price,
                        "raw_data": item
                    })

        return products

    async def fetch_orders(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Fetch orders from Toast"""
        orders = []

        if not self.location_id:
            return orders

        params = {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "pageSize": 100
        }

        page_token = None

        while True:
            if page_token:
                params["pageToken"] = page_token

            data = await self._make_request(
                "GET",
                f"{self._get_base_url()}/orders/v2/orders",
                params=params
            )

            for order in data.get("orders", []):
                line_items = []
                for check in order.get("checks", []):
                    for selection in check.get("selections", []):
                        line_items.append({
                            "name": selection.get("displayName", "Unknown"),
                            "quantity": selection.get("quantity", 1),
                            "unit_price": selection.get("price", 0),
                            "total": selection.get("price", 0) * selection.get("quantity", 1)
                        })

                orders.append({
                    "external_id": order.get("guid", ""),
                    "order_date": datetime.fromisoformat(order["openedDate"]) if order.get("openedDate") else datetime.now(),
                    "total_amount": sum(c.get("totalAmount", 0) for c in order.get("checks", [])),
                    "item_count": len(line_items),
                    "line_items": line_items,
                    "raw_data": order
                })

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return orders

    async def test_connection(self) -> bool:
        """Test Toast connection"""
        try:
            if self.location_id:
                await self._make_request(
                    "GET",
                    f"{self._get_base_url()}/restaurants/v1/restaurants/{self.location_id}"
                )
            return True
        except Exception:
            return False
