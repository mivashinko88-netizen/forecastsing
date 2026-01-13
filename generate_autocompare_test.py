"""
Generate test data for auto-compare feature testing.

This creates TWO files:
1. training_data.csv - Historical data for training the model
2. future_actuals.csv - "Actual" sales for the next 7 days (to test auto-compare)

Testing workflow:
1. Upload training_data.csv and train model
2. Go to Forecasts page and generate 7-day predictions
3. Upload future_actuals.csv
4. Check Analytics tab - accuracy metrics should now show!
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Pizza shop products
products = {
    'Pizza': {
        'items': [
            {'name': 'Pepperoni', 'base_price': 12.99, 'popularity': 1.5},
            {'name': 'Hawaiian', 'base_price': 13.99, 'popularity': 1.0},
            {'name': 'Cheese', 'base_price': 10.99, 'popularity': 1.3},
            {'name': 'Supreme', 'base_price': 15.99, 'popularity': 0.9},
            {'name': 'BBQ Chicken', 'base_price': 14.99, 'popularity': 0.8},
            {'name': 'Veggie', 'base_price': 13.49, 'popularity': 0.6},
            {'name': 'Meat Lovers', 'base_price': 16.99, 'popularity': 1.0},
            {'name': 'Margherita', 'base_price': 12.49, 'popularity': 0.7},
        ],
        'sizes': {'Small': 0.8, 'Medium': 1.0, 'Large': 1.3}
    },
    'Side': {
        'items': [
            {'name': 'Garlic Bread', 'base_price': 5.99, 'popularity': 1.2},
            {'name': 'Chicken Wings', 'base_price': 9.99, 'popularity': 1.0},
            {'name': 'Mozzarella Sticks', 'base_price': 7.99, 'popularity': 0.8},
            {'name': 'Breadsticks', 'base_price': 4.99, 'popularity': 0.9},
        ],
        'sizes': {}
    },
    'Drink': {
        'items': [
            {'name': 'Soda', 'base_price': 2.49, 'popularity': 0.4},
            {'name': 'Water', 'base_price': 1.99, 'popularity': 0.2},
        ],
        'sizes': {}
    },
    'Dessert': {
        'items': [
            {'name': 'Brownie', 'base_price': 4.49, 'popularity': 0.5},
            {'name': 'Cookie', 'base_price': 2.99, 'popularity': 0.6},
        ],
        'sizes': {}
    }
}

def is_weekend(date):
    return date.weekday() >= 5

def generate_daily_orders(date, base_orders=40):
    """Generate number of orders for a given date with realistic patterns"""
    multiplier = 1.0

    # Weekend boost
    if date.weekday() == 4:  # Friday
        multiplier *= 1.3
    elif date.weekday() == 5:  # Saturday
        multiplier *= 1.5
    elif date.weekday() == 6:  # Sunday
        multiplier *= 1.2

    # Slow early week
    if date.weekday() in [0, 1]:  # Mon, Tue
        multiplier *= 0.8

    # Month patterns
    if date.month in [11, 12]:  # Holiday season
        multiplier *= 1.15
    elif date.month in [1, 2]:  # Post-holiday slump
        multiplier *= 0.9

    # Payday boost (1st and 15th)
    if date.day in [1, 2, 15, 16]:
        multiplier *= 1.1

    return int(np.random.poisson(base_orders * multiplier))

def generate_sales_data(start_date, end_date, base_orders=40):
    """Generate sales data for a date range"""
    rows = []
    order_counter = 1

    current = start_date
    while current <= end_date:
        num_orders = generate_daily_orders(current, base_orders)

        for _ in range(num_orders):
            order_id = f"ORD-{current.strftime('%Y%m%d')}-{order_counter:05d}"
            order_counter += 1

            # Generate order time (peaks at lunch and dinner)
            hour_weights = [0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5,
                          2.5, 2.0, 1.5, 1.2, 1.5, 2.0, 3.5, 4.0, 3.5, 2.5, 1.5, 0.8, 0.4]
            hour = random.choices(range(24), weights=hour_weights)[0]
            minute = random.randint(0, 59)
            order_time = f"{hour:02d}:{minute:02d}:00"

            # Items per order (usually 1-4)
            num_items = random.choices([1, 2, 3, 4], weights=[25, 40, 25, 10])[0]

            for item_num in range(num_items):
                # Pick category
                category = random.choices(
                    list(products.keys()),
                    weights=[60, 25, 8, 7]
                )[0]

                # Pick item
                items = products[category]['items']
                item = random.choices(items, weights=[i['popularity'] for i in items])[0]

                # Pick size
                sizes = products[category].get('sizes', {})
                if sizes:
                    size = random.choices(list(sizes.keys()), weights=[0.2, 0.5, 0.3])[0]
                    size_mult = sizes[size]
                else:
                    size = ''
                    size_mult = 1.0

                # Calculate price
                unit_price = round(item['base_price'] * size_mult, 2)
                quantity = random.choices([1, 2], weights=[90, 10])[0]

                rows.append({
                    'order_id': order_id,
                    'order_date': current.strftime('%Y-%m-%d'),
                    'order_time': order_time,
                    'item_name': item['name'],
                    'category': category,
                    'size': size,
                    'unit_price': unit_price,
                    'quantity': quantity,
                    'total_price': round(unit_price * quantity, 2)
                })

        current += timedelta(days=1)

    return pd.DataFrame(rows)

def main():
    print("=" * 60)
    print("Generating Auto-Compare Test Data")
    print("=" * 60)

    today = datetime.now().date()

    # Training data: 6 months of historical data ending yesterday
    training_end = today - timedelta(days=1)
    training_start = training_end - timedelta(days=180)

    print(f"\n1. Generating training data...")
    print(f"   Date range: {training_start} to {training_end}")

    training_df = generate_sales_data(training_start, training_end)
    training_df.to_csv('data/training_data.csv', index=False)

    print(f"   Rows: {len(training_df):,}")
    print(f"   Unique orders: {training_df['order_id'].nunique():,}")
    print(f"   Saved to: data/training_data.csv")

    # Future actuals: Next 7 days starting from today
    future_start = today
    future_end = today + timedelta(days=6)

    print(f"\n2. Generating future actuals data...")
    print(f"   Date range: {future_start} to {future_end}")

    future_df = generate_sales_data(future_start, future_end)
    future_df.to_csv('data/future_actuals.csv', index=False)

    print(f"   Rows: {len(future_df):,}")
    print(f"   Unique orders: {future_df['order_id'].nunique():,}")
    print(f"   Saved to: data/future_actuals.csv")

    # Summary by item for future data (what predictions should roughly match)
    print("\n3. Future actuals summary (daily totals by item):")
    summary = future_df.groupby(['order_date', 'item_name'])['quantity'].sum().reset_index()
    pivot = summary.pivot(index='item_name', columns='order_date', values='quantity').fillna(0)
    print(pivot.head(10).to_string())

    print("\n" + "=" * 60)
    print("TEST WORKFLOW:")
    print("=" * 60)
    print("""
1. Start the server:
   uvicorn main:app --reload

2. Upload training_data.csv on the Upload Data page
   - This trains your model on 6 months of historical data

3. Go to Forecasts page
   - Generate predictions for 7 days
   - Predictions are saved to database automatically

4. Upload future_actuals.csv on the Upload Data page
   - This simulates uploading "actual" sales data
   - Auto-compare will match predictions to actuals!

5. Check Analytics tab
   - You should now see accuracy metrics
   - The Predicted vs Actual chart will show the comparison
""")

if __name__ == "__main__":
    main()
