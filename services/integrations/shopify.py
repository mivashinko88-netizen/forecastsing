"""Shopify e-commerce integration"""
import os
from typing import Dict, List, Any, Optional
from datetime import date, datetime
from urllib.parse import urlencode
import httpx

from .base import BaseIntegration


class ShopifyIntegration(BaseIntegration):
    """Shopify e-commerce integration for products and orders"""

    PROVIDER_NAME = "shopify"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.shop_domain:
            raise ValueError("shop_domain is required for Shopify integration")
        self.base_url = f"https://{self.shop_domain}/admin/api/2024-01"

    @classmethod
    def get_required_scopes(cls) -> List[str]:
        return [
            "read_products",
            "read_orders"
        ]

    @classmethod
    def get_authorization_url(cls, redirect_uri: str, state: str, shop_domain: str = None) -> str:
        """Generate Shopify OAuth authorization URL"""
        if not shop_domain:
            # Return a URL that will prompt for shop domain first
            return None

        params = {
            "client_id": os.getenv("SHOPIFY_API_KEY"),
            "scope": ",".join(cls.get_required_scopes()),
            "redirect_uri": redirect_uri,
            "state": state
        }
        return f"https://{shop_domain}/admin/oauth/authorize?{urlencode(params)}"

    @classmethod
    async def exchange_code_for_tokens(
        cls,
        code: str,
        redirect_uri: str,
        shop_domain: str = None
    ) -> Dict[str, Any]:
        """Exchange authorization code for access token"""
        if not shop_domain:
            raise ValueError("shop_domain is required")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://{shop_domain}/admin/oauth/access_token",
                json={
                    "client_id": os.getenv("SHOPIFY_API_KEY"),
                    "client_secret": os.getenv("SHOPIFY_API_SECRET"),
                    "code": code
                },
                headers={"Content-Type": "application/json"},
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()

            return {
                "access_token": data["access_token"],
                "refresh_token": None,  # Shopify doesn't use refresh tokens
                "expires_in": None,  # Shopify tokens don't expire
                "shop_domain": shop_domain,
                "scope": data.get("scope")
            }

    async def refresh_access_token(self) -> Dict[str, Any]:
        """Shopify tokens don't expire, no refresh needed"""
        return {
            "access_token": self.access_token,
            "refresh_token": None,
            "expires_in": None
        }

    def _get_headers(self) -> Dict[str, str]:
        """Get Shopify-specific headers"""
        return {
            "X-Shopify-Access-Token": self.access_token,
            "Content-Type": "application/json"
        }

    async def fetch_products(self) -> List[Dict[str, Any]]:
        """Fetch products from Shopify"""
        products = []
        page_info = None

        while True:
            params = {"limit": 250}
            if page_info:
                params["page_info"] = page_info

            data = await self._make_request(
                "GET",
                f"{self.base_url}/products.json",
                params=params
            )

            for product in data.get("products", []):
                # Get price from first variant
                price = None
                variants = product.get("variants", [])
                if variants:
                    price = float(variants[0].get("price", 0))

                products.append({
                    "external_id": str(product["id"]),
                    "name": product.get("title", "Unknown"),
                    "category": product.get("product_type"),
                    "sku": variants[0].get("sku") if variants else None,
                    "price": price,
                    "raw_data": product
                })

            # Check for pagination
            # Shopify uses cursor-based pagination via Link header
            # For simplicity, we'll fetch first page only in this implementation
            # Full pagination would require parsing Link header
            break

        return products

    async def fetch_orders(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Fetch orders from Shopify"""
        orders = []

        params = {
            "status": "any",
            "created_at_min": f"{start_date}T00:00:00Z",
            "created_at_max": f"{end_date}T23:59:59Z",
            "limit": 250
        }

        data = await self._make_request(
            "GET",
            f"{self.base_url}/orders.json",
            params=params
        )

        for order in data.get("orders", []):
            line_items = []
            for item in order.get("line_items", []):
                line_items.append({
                    "name": item.get("title", "Unknown"),
                    "quantity": item.get("quantity", 1),
                    "unit_price": float(item.get("price", 0)),
                    "total": float(item.get("price", 0)) * item.get("quantity", 1)
                })

            orders.append({
                "external_id": str(order["id"]),
                "order_date": datetime.fromisoformat(order["created_at"].replace("Z", "+00:00")),
                "total_amount": float(order.get("total_price", 0)),
                "item_count": len(line_items),
                "line_items": line_items,
                "raw_data": order
            })

        return orders

    async def test_connection(self) -> bool:
        """Test Shopify connection by fetching shop info"""
        try:
            await self._make_request("GET", f"{self.base_url}/shop.json")
            return True
        except Exception:
            return False
