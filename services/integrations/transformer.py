"""Transform synced data to training format"""
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy.orm import Session

from db_models import SyncedOrder, SyncedProduct


class DataTransformer:
    """Transform provider data to common training format"""

    def __init__(self, db: Session):
        self.db = db

    def orders_to_dataframe(
        self,
        business_id: int,
        integration_id: int = None
    ) -> pd.DataFrame:
        """
        Convert synced orders to training-ready DataFrame

        Output columns:
        - date: Order date (YYYY-MM-DD)
        - item_name: Product name
        - quantity: Units sold
        - unit_price: Price per unit (optional)
        - total_price: Line total (optional)
        - order_id: Reference ID

        Args:
            business_id: Business to export data for
            integration_id: Optional specific integration

        Returns:
            DataFrame ready for training
        """
        query = self.db.query(SyncedOrder).filter(
            SyncedOrder.business_id == business_id
        )

        if integration_id:
            query = query.filter(SyncedOrder.integration_id == integration_id)

        orders = query.order_by(SyncedOrder.order_date).all()

        rows = []
        for order in orders:
            try:
                raw = json.loads(order.raw_data) if order.raw_data else {}
            except json.JSONDecodeError:
                raw = {}

            # Extract line items from raw data
            line_items = raw.get("line_items", [])

            if line_items:
                # Process each line item
                for item in line_items:
                    rows.append({
                        "date": order.order_date.strftime("%Y-%m-%d"),
                        "item_name": item.get("name", "Unknown"),
                        "quantity": item.get("quantity", 1),
                        "unit_price": item.get("unit_price"),
                        "total_price": item.get("total"),
                        "order_id": order.external_id
                    })
            else:
                # If no line items, create aggregate row
                rows.append({
                    "date": order.order_date.strftime("%Y-%m-%d"),
                    "item_name": "All Items",
                    "quantity": order.item_count or 1,
                    "unit_price": None,
                    "total_price": order.total_amount,
                    "order_id": order.external_id
                })

        return pd.DataFrame(rows)

    def orders_to_csv(
        self,
        business_id: int,
        integration_id: int = None
    ) -> str:
        """
        Export synced orders to CSV string

        Args:
            business_id: Business to export data for
            integration_id: Optional specific integration

        Returns:
            CSV string
        """
        df = self.orders_to_dataframe(business_id, integration_id)
        return df.to_csv(index=False)

    def get_aggregated_sales(
        self,
        business_id: int,
        integration_id: int = None
    ) -> pd.DataFrame:
        """
        Get aggregated sales data by date and item

        This is the format expected by the training endpoint.

        Returns:
            DataFrame with columns: date, item_name, quantity
        """
        df = self.orders_to_dataframe(business_id, integration_id)

        if df.empty:
            return df

        # Aggregate by date and item
        aggregated = df.groupby(["date", "item_name"]).agg({
            "quantity": "sum"
        }).reset_index()

        return aggregated

    def get_products_dict(
        self,
        business_id: int,
        integration_id: int = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get products as dictionary keyed by external_id

        Returns:
            Dict mapping external_id to product info
        """
        query = self.db.query(SyncedProduct).filter(
            SyncedProduct.business_id == business_id,
            SyncedProduct.is_active == True
        )

        if integration_id:
            query = query.filter(SyncedProduct.integration_id == integration_id)

        products = query.all()

        return {
            p.external_id: {
                "name": p.name,
                "category": p.category,
                "sku": p.sku,
                "price": p.price
            }
            for p in products
        }

    def get_sync_summary(
        self,
        business_id: int,
        integration_id: int = None
    ) -> Dict[str, Any]:
        """
        Get summary statistics for synced data

        Returns:
            Dict with counts, date ranges, etc.
        """
        query = self.db.query(SyncedOrder).filter(
            SyncedOrder.business_id == business_id
        )

        if integration_id:
            query = query.filter(SyncedOrder.integration_id == integration_id)

        orders = query.all()

        product_query = self.db.query(SyncedProduct).filter(
            SyncedProduct.business_id == business_id,
            SyncedProduct.is_active == True
        )

        if integration_id:
            product_query = product_query.filter(SyncedProduct.integration_id == integration_id)

        products = product_query.all()

        if not orders:
            return {
                "order_count": 0,
                "product_count": len(products),
                "date_range": None,
                "total_revenue": 0
            }

        dates = [o.order_date for o in orders]
        return {
            "order_count": len(orders),
            "product_count": len(products),
            "date_range": {
                "start": min(dates).isoformat(),
                "end": max(dates).isoformat()
            },
            "total_revenue": sum(o.total_amount or 0 for o in orders)
        }
