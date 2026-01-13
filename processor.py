import pandas as pd
import numpy as np
from fastapi import UploadFile, HTTPException


def clean_value(v):
    """Convert NaN/Inf values to None for JSON serialization"""
    try:
        if v is None:
            return None
        if pd.isna(v):
            return None
        if isinstance(v, (float, np.floating)):
            if np.isnan(v) or np.isinf(v):
                return None
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        return v
    except (ValueError, TypeError):
        return None


def clean_record(record):
    """Clean all values in a record dict"""
    return {k: clean_value(v) for k, v in record.items()}

# your standard columns
YOUR_COLUMNS = [
    "date", "item_name", "quantity", "size", "type", "unit_price", "total_price",
    "order_id", "time", "location", "employee_id", "payment_method", "notes"
]

# possible names customers might use for each column
COLUMN_ALIASES = {
    "date": ["date", "sale_date", "transaction_date", "day", "sold_on", "order_date", "trans_date"],
    "item_name": ["item_name", "product", "item", "product_name", "name", "menu_item", "pizza_name", "item_description", "description"],
    "quantity": ["quantity", "qty", "amount", "units", "count", "sold", "units_sold", "qty_sold", "order_qty", "num_sold"],
    "size": ["size", "portion", "variant", "oz", "volume", "pizza_size", "product_size", "item_size", "portion_size"],
    "type": ["type", "category", "pizza_category", "pizza_type", "product_type", "item_type", "product_category", "item_category", "food_type", "menu_category", "group", "department", "class"],
    "unit_price": ["unit_price", "price", "item_price", "product_price", "sale_price", "selling_price", "cost", "retail_price", "price_each", "each_price"],
    "total_price": ["total_price", "total", "revenue", "amount_paid", "order_total", "line_total", "subtotal", "sales", "income", "sale_amount"],
    "order_id": ["order_id", "order", "transaction_id", "receipt", "ticket", "order_details_id", "invoice", "invoice_id", "receipt_id"],
    "time": ["time", "sale_time", "timestamp", "hour", "order_time", "transaction_time", "time_of_sale"],
    "location": ["location", "store", "branch", "site", "outlet"],
    "employee_id": ["employee_id", "employee", "staff", "server", "cashier", "sold_by"],
    "payment_method": ["payment_method", "payment", "payment_type", "tender", "pay_type"],
    "notes": ["notes", "comments", "remarks", "memo", "pizza_ingredients"]
}

async def process_csv(file:UploadFile):

    #read the file into pandas
    df = pd.read_csv(file.file)

    # normalize their column names
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_", regex=False)

    # try to auto-map their columns to ours
    suggested_mapping = auto_map_columns(df.columns.tolist())

    # Convert preview to JSON-safe format - convert everything to strings
    preview_df = df.head(5).astype(str).replace('nan', '').replace('NaN', '').replace('NaT', '')
    preview = preview_df.to_dict(orient="records")

    return {
        "their_columns": df.columns.tolist(),
        "suggested_mapping": suggested_mapping,
        "unmapped_theirs": get_unmapped_theirs(df.columns.tolist(), suggested_mapping),
        "unmapped_ours": get_unmapped_ours(suggested_mapping),
        "row_count": len(df),
        "preview": preview
    }

def auto_map_columns(their_columns):
    """Try to match their columns to our standard columns"""
    mapping = {}
    
    for their_col in their_columns:
        for our_col, aliases in COLUMN_ALIASES.items():
            if their_col in aliases:
                mapping[their_col] = our_col
                break
    
    return mapping

def get_unmapped_theirs(their_columns, mapping):
    """Columns they have that we couldn't auto-map"""
    return [col for col in their_columns if col not in mapping]


def get_unmapped_ours(mapping):
    """Our columns that didn't get mapped to anything"""
    mapped_ours = list(mapping.values())
    return [col for col in YOUR_COLUMNS if col not in mapped_ours]
