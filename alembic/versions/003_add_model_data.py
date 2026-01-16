"""Add model_data column to trained_models table

Revision ID: 003_add_model_data
Revises: 002_add_business_address
Create Date: 2026-01-16

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003_add_model_data'
down_revision: Union[str, None] = '002_add_business_address'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add model_data column to trained_models table for storing serialized model binary
    op.add_column('trained_models', sa.Column('model_data', sa.LargeBinary(), nullable=True))

    # Make model_path nullable (was previously required, now optional)
    op.alter_column('trained_models', 'model_path', nullable=True)


def downgrade() -> None:
    # Remove model_data column
    op.drop_column('trained_models', 'model_data')

    # Make model_path required again
    op.alter_column('trained_models', 'model_path', nullable=False)
