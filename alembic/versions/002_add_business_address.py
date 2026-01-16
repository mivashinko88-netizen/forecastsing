"""Add address column to businesses table

Revision ID: 002_add_business_address
Revises: 001_add_password_hash
Create Date: 2026-01-16

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '002_add_business_address'
down_revision: Union[str, None] = '001_add_password_hash'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add address column to businesses table
    op.add_column('businesses', sa.Column('address', sa.String(), nullable=True))


def downgrade() -> None:
    # Remove address column
    op.drop_column('businesses', 'address')
