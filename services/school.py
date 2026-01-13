# services/school.py
from datetime import date

def get_school_calendar(year: int, region: str = "US"):
    """Get school calendar events - approximate dates"""
    
    events = []
    
    # Spring semester
    events.append({
        "start_date": f"{year}-01-06",
        "end_date": f"{year}-01-06",
        "name": "school_resumes_spring",
        "type": "school_start"
    })
    
    # Spring break (typically mid-March)
    events.append({
        "start_date": f"{year}-03-10",
        "end_date": f"{year}-03-17",
        "name": "spring_break",
        "type": "break"
    })
    
    # Summer break
    events.append({
        "start_date": f"{year}-05-25",
        "end_date": f"{year}-08-15",
        "name": "summer_break",
        "type": "break"
    })
    
    # Fall semester
    events.append({
        "start_date": f"{year}-08-16",
        "end_date": f"{year}-08-16",
        "name": "back_to_school",
        "type": "school_start"
    })
    
    # Thanksgiving break
    events.append({
        "start_date": f"{year}-11-20",
        "end_date": f"{year}-11-24",
        "name": "thanksgiving_break",
        "type": "break"
    })
    
    # Winter break
    events.append({
        "start_date": f"{year}-12-18",
        "end_date": f"{year+1}-01-05",
        "name": "winter_break",
        "type": "break"
    })
    
    return events


def is_school_in_session(check_date: date, year: int):
    """Check if school is in session on a given date"""
    
    calendar = get_school_calendar(year)
    
    for event in calendar:
        if event["type"] == "break":
            start = date.fromisoformat(event["start_date"])
            end = date.fromisoformat(event["end_date"])
            if start <= check_date <= end:
                return False
    
    return True