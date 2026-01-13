"""Base class for all POS/e-commerce integrations"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import date
import httpx


class BaseIntegration(ABC):
    """Abstract base class for all POS/e-commerce integrations"""

    PROVIDER_NAME: str = ""
    BASE_URL: str = ""
    OAUTH_URL: str = ""

    def __init__(
        self,
        access_token: str,
        refresh_token: Optional[str] = None,
        merchant_id: Optional[str] = None,
        location_id: Optional[str] = None,
        shop_domain: Optional[str] = None,
        **kwargs
    ):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.merchant_id = merchant_id
        self.location_id = location_id
        self.shop_domain = shop_domain
        self.extra_config = kwargs

    def _get_headers(self) -> Dict[str, str]:
        """Get default headers for API requests"""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

    async def _make_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict] = None,
        json: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to provider API"""
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method,
                url=url,
                headers=request_headers,
                json=json,
                params=params,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()

    @classmethod
    @abstractmethod
    def get_authorization_url(cls, redirect_uri: str, state: str) -> str:
        """Generate OAuth authorization URL for user to authorize access"""
        pass

    @classmethod
    @abstractmethod
    async def exchange_code_for_tokens(cls, code: str, redirect_uri: str) -> Dict[str, Any]:
        """
        Exchange authorization code for access/refresh tokens

        Returns:
            Dict with keys: access_token, refresh_token (optional),
            expires_in (seconds), merchant_id, location_id, etc.
        """
        pass

    @abstractmethod
    async def refresh_access_token(self) -> Dict[str, Any]:
        """
        Refresh expired access token using refresh_token

        Returns:
            Dict with keys: access_token, refresh_token (optional), expires_in
        """
        pass

    @abstractmethod
    async def fetch_products(self) -> List[Dict[str, Any]]:
        """
        Fetch product catalog from provider

        Returns:
            List of products with standardized keys:
            - external_id: Provider's product ID
            - name: Product name
            - category: Category/type (optional)
            - sku: SKU (optional)
            - price: Unit price (optional)
            - raw_data: Full provider response
        """
        pass

    @abstractmethod
    async def fetch_orders(
        self,
        start_date: date,
        end_date: date
    ) -> List[Dict[str, Any]]:
        """
        Fetch orders within date range

        Returns:
            List of orders with standardized keys:
            - external_id: Provider's order ID
            - order_date: DateTime of order
            - total_amount: Order total
            - item_count: Number of items
            - line_items: List of items with name, quantity, price
            - raw_data: Full provider response
        """
        pass

    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if the integration connection is valid"""
        pass

    @classmethod
    def get_required_scopes(cls) -> List[str]:
        """Get required OAuth scopes for this integration"""
        return []
