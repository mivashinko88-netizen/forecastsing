"""Add password_hash column to users table

Revision ID: 001_add_password_hash
Revises:
Create Date: 2025-01-15

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_add_password_hash'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add password_hash column to users table
    op.add_column('users', sa.Column('password_hash', sa.String(), nullable=True))

    # Make google_id nullable for local auth users
    # Note: SQLite doesn't support ALTER COLUMN, so this is a no-op for SQLite
    # For PostgreSQL, this will work
    try:
        op.alter_column('users', 'google_id',
                        existing_type=sa.String(),
                        nullable=True)
    except Exception:
        # SQLite doesn't support this operation
        pass


def downgrade() -> None:
    # Remove password_hash column
    op.drop_column('users', 'password_hash')

    # Revert google_id to non-nullable (only works on PostgreSQL)
    try:
        op.alter_column('users', 'google_id',
                        existing_type=sa.String(),
                        nullable=False)
    except Exception:
        pass
