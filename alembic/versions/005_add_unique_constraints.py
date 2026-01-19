"""Add unique constraints to prevent duplicate records

Revision ID: 005_add_unique_constraints
Revises: 004_add_performance_indexes
Create Date: 2026-01-19

This migration adds unique constraints to prevent:
- Duplicate predictions for the same model/date/item combination
- Duplicate synced products/orders from the same integration

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '005_add_unique_constraints'
down_revision: Union[str, None] = '004_add_performance_indexes'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # First, remove any duplicate predictions before adding constraint
    # This uses a subquery to find and delete duplicates, keeping the most recent
    conn = op.get_bind()

    # For SQLite compatibility, we use a different approach
    dialect = conn.dialect.name

    if dialect == 'postgresql':
        # PostgreSQL: Delete duplicates keeping the one with highest ID
        conn.execute(sa.text("""
            DELETE FROM predictions p1
            USING predictions p2
            WHERE p1.model_id = p2.model_id
              AND p1.prediction_date = p2.prediction_date
              AND p1.item_name = p2.item_name
              AND p1.id < p2.id
        """))

        # Delete duplicate synced_products
        conn.execute(sa.text("""
            DELETE FROM synced_products p1
            USING synced_products p2
            WHERE p1.integration_id = p2.integration_id
              AND p1.external_id = p2.external_id
              AND p1.id < p2.id
        """))

        # Delete duplicate synced_orders
        conn.execute(sa.text("""
            DELETE FROM synced_orders o1
            USING synced_orders o2
            WHERE o1.integration_id = o2.integration_id
              AND o1.external_id = o2.external_id
              AND o1.id < o2.id
        """))
    else:
        # SQLite: Use a subquery approach
        conn.execute(sa.text("""
            DELETE FROM predictions
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM predictions
                GROUP BY model_id, prediction_date, item_name
            )
        """))

        conn.execute(sa.text("""
            DELETE FROM synced_products
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM synced_products
                GROUP BY integration_id, external_id
            )
        """))

        conn.execute(sa.text("""
            DELETE FROM synced_orders
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM synced_orders
                GROUP BY integration_id, external_id
            )
        """))

    # Now add the unique constraints
    op.create_unique_constraint(
        'uq_predictions_model_date_item',
        'predictions',
        ['model_id', 'prediction_date', 'item_name']
    )

    op.create_unique_constraint(
        'uq_synced_products_integration_external',
        'synced_products',
        ['integration_id', 'external_id']
    )

    op.create_unique_constraint(
        'uq_synced_orders_integration_external',
        'synced_orders',
        ['integration_id', 'external_id']
    )


def downgrade() -> None:
    op.drop_constraint('uq_predictions_model_date_item', 'predictions', type_='unique')
    op.drop_constraint('uq_synced_products_integration_external', 'synced_products', type_='unique')
    op.drop_constraint('uq_synced_orders_integration_external', 'synced_orders', type_='unique')
