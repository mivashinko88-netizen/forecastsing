import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define products with their categories, sizes, and pricing
products = {
    'Pizza': {
        'items': [
            {'name': 'Pepperoni', 'base_price': 11.99, 'cost_ratio': 0.38, 'popularity': 1.5},
            {'name': 'Hawaiian', 'base_price': 12.99, 'cost_ratio': 0.40, 'popularity': 1.0},
            {'name': 'Cheese', 'base_price': 9.99, 'cost_ratio': 0.35, 'popularity': 1.2},
            {'name': 'Supreme', 'base_price': 14.99, 'cost_ratio': 0.42, 'popularity': 0.9},
            {'name': 'BBQ Chicken', 'base_price': 13.99, 'cost_ratio': 0.41, 'popularity': 0.85},
            {'name': 'Veggie Delight', 'base_price': 12.49, 'cost_ratio': 0.39, 'popularity': 0.6},
            {'name': 'Meat Lovers', 'base_price': 15.99, 'cost_ratio': 0.45, 'popularity': 1.1},
            {'name': 'Margherita', 'base_price': 11.49, 'cost_ratio': 0.36, 'popularity': 0.75},
            {'name': 'Buffalo Chicken', 'base_price': 13.49, 'cost_ratio': 0.40, 'popularity': 0.7},
            {'name': 'Mushroom Olive', 'base_price': 12.99, 'cost_ratio': 0.38, 'popularity': 0.5}
        ],
        'sizes': {'Small': 0.85, 'Medium': 1.0, 'Large': 1.35, 'XLarge': 1.60}
    },
    'Side': {
        'items': [
            {'name': 'Garlic Bread', 'base_price': 5.99, 'cost_ratio': 0.30, 'popularity': 1.3},
            {'name': 'Cheesy Bread', 'base_price': 6.99, 'cost_ratio': 0.32, 'popularity': 1.1},
            {'name': 'Mozzarella Sticks', 'base_price': 7.49, 'cost_ratio': 0.35, 'popularity': 0.9},
            {'name': 'Chicken Wings', 'base_price': 9.99, 'cost_ratio': 0.40, 'popularity': 1.0},
            {'name': 'Breadsticks', 'base_price': 4.99, 'cost_ratio': 0.28, 'popularity': 0.8}
        ],
        'sizes': {}
    },
    'Salad': {
        'items': [
            {'name': 'Caesar Salad', 'base_price': 6.99, 'cost_ratio': 0.30, 'popularity': 0.9},
            {'name': 'Garden Salad', 'base_price': 5.99, 'cost_ratio': 0.28, 'popularity': 0.7},
            {'name': 'Greek Salad', 'base_price': 7.49, 'cost_ratio': 0.32, 'popularity': 0.5}
        ],
        'sizes': {}
    },
    'Drink': {
        'items': [
            {'name': '2 Liter Soda', 'base_price': 3.49, 'cost_ratio': 0.25, 'popularity': 0.5},
            {'name': 'Bottled Water', 'base_price': 1.99, 'cost_ratio': 0.20, 'popularity': 0.3},
            {'name': 'Juice Box', 'base_price': 2.49, 'cost_ratio': 0.28, 'popularity': 0.2}
        ],
        'sizes': {}
    },
    'Dessert': {
        'items': [
            {'name': 'Chocolate Chip Cookie', 'base_price': 3.99, 'cost_ratio': 0.30, 'popularity': 0.8},
            {'name': 'Cinnamon Bites', 'base_price': 5.99, 'cost_ratio': 0.32, 'popularity': 0.7},
            {'name': 'Brownie', 'base_price': 4.49, 'cost_ratio': 0.30, 'popularity': 0.5}
        ],
        'sizes': {}
    }
}

# US Holidays (approximate dates)
holidays_2023 = [
    '2023-01-01', '2023-01-16', '2023-02-14', '2023-02-20', '2023-03-17',
    '2023-04-09', '2023-05-29', '2023-06-19', '2023-07-04', '2023-09-04',
    '2023-10-31', '2023-11-23', '2023-11-24', '2023-12-24', '2023-12-25', '2023-12-31'
]
holidays_2024 = [
    '2024-01-01', '2024-01-15', '2024-02-14', '2024-02-19', '2024-03-17',
    '2024-03-31', '2024-05-27', '2024-06-19', '2024-07-04', '2024-09-02',
    '2024-10-31', '2024-11-28', '2024-11-29', '2024-12-24', '2024-12-25', '2024-12-31'
]
all_holidays = set(holidays_2023 + holidays_2024)

# Football Sundays (NFL season Sept-Jan, Sundays)
def is_football_day(date):
    if date.weekday() != 6:  # Not Sunday
        return False
    month = date.month
    return month >= 9 or month <= 1

# Generate dates from Jan 2023 to Dec 2024 (2 years)
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)
days = (end_date - start_date).days + 1

rows = []
order_counter = 1

print("Generating sales data...")
for day_offset in range(days):
    current_date = start_date + timedelta(days=day_offset)
    date_str = current_date.strftime('%Y-%m-%d')
    day_name = current_date.strftime('%A')
    dow = current_date.weekday()
    month = current_date.month
    year = current_date.year
    week = current_date.isocalendar()[1]

    # Calculate daily order multiplier based on patterns
    base_orders = 35

    # Weekend boost
    if dow >= 4:  # Fri, Sat, Sun
        base_orders *= 1.4
    if dow == 5:  # Saturday peak
        base_orders *= 1.2

    # Football Sunday boost
    football = is_football_day(current_date)
    if football:
        base_orders *= 1.5

    # Holiday boost
    is_holiday = date_str in all_holidays
    if is_holiday:
        base_orders *= 1.3

    # Seasonal patterns
    if month in [11, 12]:  # Holiday season
        base_orders *= 1.15
    elif month in [6, 7, 8]:  # Summer slowdown
        base_orders *= 0.9

    # Payday boost (1st and 15th)
    if current_date.day in [1, 2, 15, 16]:
        base_orders *= 1.1

    # Add some randomness
    num_orders = int(np.random.poisson(base_orders))

    for order_num in range(num_orders):
        order_id = f'ORD-{date_str.replace("-", "")}-{order_counter:06d}'
        order_counter += 1

        # Generate order time (peak hours 11-14, 17-20)
        hour_weights = [0.5, 0.3, 0.2, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0,
                       3.5, 3.0, 2.5, 2.0, 2.0, 2.5, 4.0, 4.5, 4.0, 3.0, 2.0, 1.0, 0.5]
        hour = random.choices(range(24), weights=hour_weights)[0]
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        order_time = f'{hour:02d}:{minute:02d}:{second:02d}'
        order_datetime = f'{date_str} {order_time}'

        # Number of items in order (1-6, weighted toward 2-3)
        num_items = random.choices([1, 2, 3, 4, 5, 6], weights=[15, 30, 25, 15, 10, 5])[0]

        item_num = 1
        for _ in range(num_items):
            # Pick category (pizza most common, drinks are add-ons not always ordered)
            category = random.choices(
                list(products.keys()),
                weights=[55, 22, 10, 8, 5]  # Pizza, Side, Salad, Drink, Dessert
            )[0]

            # Pick item from category based on popularity
            items = products[category]['items']
            item = random.choices(
                items,
                weights=[i['popularity'] for i in items]
            )[0]

            # Pick size if applicable
            sizes = products[category].get('sizes', {})
            if sizes:
                size = random.choices(
                    list(sizes.keys()),
                    weights=[0.15, 0.35, 0.35, 0.15]
                )[0]
                size_mult = sizes[size]
            else:
                size = ''
                size_mult = 1.0

            # Calculate prices
            unit_price = round(item['base_price'] * size_mult, 2)
            unit_cost = round(unit_price * item['cost_ratio'], 2)
            quantity = random.choices([1, 2, 3], weights=[80, 15, 5])[0]
            line_revenue = round(unit_price * quantity, 2)
            line_cost = round(unit_cost * quantity, 2)
            gross_margin = round(line_revenue - line_cost, 2)

            order_item_id = f'{order_id}-{item_num:02d}'

            event_type = ''
            if is_holiday:
                event_type = 'Holiday'
            elif football:
                event_type = 'Football'

            rows.append({
                'order_id': order_id,
                'order_item_id': order_item_id,
                'order_datetime': order_datetime,
                'order_date': date_str,
                'order_time': order_time,
                'day_of_week': day_name,
                'hour': hour,
                'week_of_year': week,
                'month': month,
                'year': year,
                'item_name': item['name'],
                'category': category,
                'size': size,
                'unit_price': unit_price,
                'unit_cost': unit_cost,
                'quantity': quantity,
                'line_revenue': line_revenue,
                'line_cost': line_cost,
                'gross_margin': gross_margin,
                'is_football_day': football,
                'event_type': event_type
            })

            item_num += 1

    # Progress update every 100 days
    if day_offset % 100 == 0:
        print(f"  Processed {day_offset}/{days} days...")

# Create DataFrame and save
print("\nSaving to CSV...")
df = pd.DataFrame(rows)
df.to_csv('data/test_sales_2yr.csv', index=False)

print(f'\n=== Dataset Created Successfully ===')
print(f'Total rows: {len(df):,}')
print(f'Date range: {df["order_date"].min()} to {df["order_date"].max()}')
print(f'Unique orders: {df["order_id"].nunique():,}')
print(f'\nItems by category:')
print(df['category'].value_counts().to_string())
print(f'\nTop 10 items:')
print(df['item_name'].value_counts().head(10).to_string())
print(f'\nFile saved to: data/test_sales_2yr.csv')
