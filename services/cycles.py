# services/cycles.py
from datetime import date, timedelta

def get_payday_dates(year: int):
    """Get common payday dates (1st, 15th, and every other Friday)"""
    
    paydays = []
    
    # 1st and 15th of each month (common for salaried workers)
    for month in range(1, 13):
        paydays.append({
            "date": f"{year}-{month:02d}-01",
            "type": "monthly_1st"
        })
        paydays.append({
            "date": f"{year}-{month:02d}-15",
            "type": "monthly_15th"
        })
    
    # Every Friday (common for hourly workers)
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    current = start
    
    # Find first Friday
    while current.weekday() != 4:  # 4 = Friday
        current += timedelta(days=1)
    
    while current <= end:
        paydays.append({
            "date": str(current),
            "type": "weekly_friday"
        })
        current += timedelta(days=7)
    
    return paydays


def get_ebt_dates(year: int, state: str = "default"):
    """Get EBT deposit dates - varies by state, using common pattern"""
    
    # Most states deposit between 1st-10th based on case number
    # This gives a general pattern
    ebt_dates = []
    
    for month in range(1, 13):
        for day in range(1, 11):  # 1st through 10th
            ebt_dates.append({
                "date": f"{year}-{month:02d}-{day:02d}",
                "type": "ebt_window"
            })
    
    return ebt_dates