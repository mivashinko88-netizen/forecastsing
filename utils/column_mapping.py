# utils/column_mapping.py - Shared column mapping for CSV processing
"""
Centralized column mapping for sales data CSV files.
Maps common variations of column names to standardized internal names.
"""

# Standard column names used internally
STANDARD_COLUMNS = [
    "date", "item_name", "quantity", "size", "type", "unit_price", "total_price",
    "order_id", "time", "location", "employee_id", "payment_method", "notes", "cost"
]

# Comprehensive mapping from common variations to standard names
# Format: {"alias": "standard_name"}
COLUMN_MAPPING = {
    # Date columns
    "order_date": "date",
    "sale_date": "date",
    "transaction_date": "date",
    "order date": "date",
    "sale date": "date",
    "trans_date": "date",
    "day": "date",
    "sold_on": "date",

    # Item/product columns
    "pizza_name": "item_name",
    "product": "item_name",
    "product_name": "item_name",
    "item": "item_name",
    "name": "item_name",
    "product name": "item_name",
    "menu_item": "item_name",
    "item_description": "item_name",
    "description": "item_name",

    # Size columns
    "pizza_size": "size",
    "product_size": "size",
    "item_size": "size",
    "portion": "size",
    "portion_size": "size",
    "variant": "size",
    "oz": "size",
    "volume": "size",

    # Type/category columns
    "pizza_category": "type",
    "pizza_type": "type",
    "product_type": "type",
    "item_type": "type",
    "category": "type",
    "product_category": "type",
    "item_category": "type",
    "food_type": "type",
    "menu_category": "type",
    "group": "type",
    "department": "type",
    "class": "type",

    # Unit price columns (sale price to customer)
    "price": "unit_price",
    "item_price": "unit_price",
    "product_price": "unit_price",
    "sale_price": "unit_price",
    "selling_price": "unit_price",
    "retail_price": "unit_price",
    "price_each": "unit_price",
    "each_price": "unit_price",
    "pizza_price": "unit_price",

    # Cost columns (business cost, separate from sale price)
    "item_cost": "cost",
    "product_cost": "cost",
    "unit_cost": "cost",

    # Total price columns
    "total_price": "total_price",
    "total": "total_price",
    "revenue": "total_price",
    "amount_paid": "total_price",
    "order_total": "total_price",
    "line_total": "total_price",
    "subtotal": "total_price",
    "sales": "total_price",
    "income": "total_price",
    "sale_amount": "total_price",

    # Order ID columns
    "order": "order_id",
    "transaction_id": "order_id",
    "receipt": "order_id",
    "ticket": "order_id",
    "order_details_id": "order_id",
    "invoice": "order_id",
    "invoice_id": "order_id",
    "receipt_id": "order_id",

    # Quantity columns
    "qty": "quantity",
    "amount": "quantity",
    "units": "quantity",
    "units_sold": "quantity",
    "qty_sold": "quantity",
    "order_qty": "quantity",
    "order_quantity": "quantity",
    "count": "quantity",
    "num_sold": "quantity",
    "sold": "quantity",

    # Time columns
    "order_time": "time",
    "sale_time": "time",
    "transaction_time": "time",
    "time_of_sale": "time",
    "timestamp": "time",
    "hour": "time",

    # Location columns
    "store": "location",
    "branch": "location",
    "site": "location",
    "outlet": "location",

    # Employee columns
    "employee": "employee_id",
    "staff": "employee_id",
    "server": "employee_id",
    "cashier": "employee_id",
    "sold_by": "employee_id",

    # Payment columns
    "payment": "payment_method",
    "payment_type": "payment_method",
    "tender": "payment_method",
    "pay_type": "payment_method",

    # Notes columns
    "comments": "notes",
    "remarks": "notes",
    "memo": "notes",
    "pizza_ingredients": "notes",
}

# Reverse mapping: standard column -> list of aliases (for processor.py compatibility)
COLUMN_ALIASES = {}
for alias, standard in COLUMN_MAPPING.items():
    if standard not in COLUMN_ALIASES:
        COLUMN_ALIASES[standard] = [standard]  # Include standard name itself
    if alias not in COLUMN_ALIASES[standard]:
        COLUMN_ALIASES[standard].append(alias)


def apply_column_mapping(df, mapping=None):
    """
    Apply column mapping to a DataFrame.

    Args:
        df: pandas DataFrame
        mapping: Optional custom mapping dict. Uses COLUMN_MAPPING if not provided.

    Returns:
        DataFrame with renamed columns
    """
    if mapping is None:
        mapping = COLUMN_MAPPING
    return df.rename(columns=mapping)


def get_mapping_for_columns(columns, mapping=None):
    """
    Get only the relevant mappings for the given column names.

    Args:
        columns: List of column names to check
        mapping: Optional custom mapping dict

    Returns:
        Dict of mappings that apply to the given columns
    """
    if mapping is None:
        mapping = COLUMN_MAPPING
    return {k: v for k, v in mapping.items() if k in columns}


def auto_map_columns(their_columns):
    """
    Try to match their columns to our standard columns.
    Used by processor.py for column suggestions.

    Args:
        their_columns: List of column names from uploaded file

    Returns:
        Dict mapping their column names to our standard names
    """
    mapping = {}
    for their_col in their_columns:
        for our_col, aliases in COLUMN_ALIASES.items():
            if their_col in aliases:
                mapping[their_col] = our_col
                break
    return mapping


def get_unmapped_theirs(their_columns, mapping):
    """Get columns they have that we couldn't auto-map"""
    return [col for col in their_columns if col not in mapping]


def get_unmapped_ours(mapping):
    """Get our standard columns that didn't get mapped to anything"""
    mapped_ours = list(mapping.values())
    return [col for col in STANDARD_COLUMNS if col not in mapped_ours]
