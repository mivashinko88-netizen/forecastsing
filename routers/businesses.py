# routers/businesses.py - Business CRUD endpoints
from typing import List, Optional
from datetime import datetime, date
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
import asyncio

from database import get_db
from db_models import User, Business
from auth import get_current_user
from schemas import BusinessCreate, BusinessUpdate, BusinessResponse
from config import BusinessConfig
from services.events import get_holidays
from services.sports import get_nfl_games
from services.cycles import get_payday_dates
from services.school import get_school_calendar
from services.weather import get_forecast_weather

router = APIRouter(prefix="/businesses", tags=["Businesses"])


def get_coordinates(zipcode: str, city: str, state: str, country: str) -> tuple:
    """Get latitude and longitude for a location using Nominatim"""
    try:
        import requests as req
        query = f"{zipcode}, {city}, {state}, {country}"
        url = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=1"
        response = req.get(url, headers={"User-Agent": "ForecastingApp/1.0"}, timeout=5)
        if response.ok and response.json():
            data = response.json()[0]
            return float(data["lat"]), float(data["lon"])
    except Exception:
        pass
    return None, None


@router.get("", response_model=List[BusinessResponse])
async def list_businesses(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all businesses for the current user"""
    businesses = db.query(Business).filter(Business.user_id == current_user.id).all()
    return businesses


@router.post("", response_model=BusinessResponse, status_code=status.HTTP_201_CREATED)
async def create_business(
    data: BusinessCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new business"""

    # Validate business type - physical and online types
    physical_types = ["restaurant", "retail", "service"]
    online_types = ["fashion", "electronics", "food_grocery", "home_garden", "health_beauty", "other_online"]
    valid_types = physical_types + online_types

    if data.business_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid business type. Must be one of: {', '.join(valid_types)}"
        )

    # Get coordinates only for physical stores with location data
    lat, lon = None, None
    if data.city and data.zipcode and not data.is_online:
        lat, lon = get_coordinates(data.zipcode, data.city, data.state, data.country)

    business = Business(
        user_id=current_user.id,
        name=data.name,
        business_type=data.business_type,
        is_online=data.is_online,
        city=data.city,
        state=data.state,
        zipcode=data.zipcode,
        country=data.country,
        timezone=data.timezone,
        latitude=lat,
        longitude=lon,
        open_time=data.open_time,
        close_time=data.close_time,
        days_open=data.days_open,
        marketing_channels=data.marketing_channels
    )

    db.add(business)
    db.commit()
    db.refresh(business)

    return business


@router.get("/{business_id}", response_model=BusinessResponse)
async def get_business(
    business_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific business"""
    business = db.query(Business).filter(
        Business.id == business_id,
        Business.user_id == current_user.id
    ).first()

    if not business:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Business not found"
        )

    return business


@router.put("/{business_id}", response_model=BusinessResponse)
async def update_business(
    business_id: int,
    data: BusinessUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a business"""
    business = db.query(Business).filter(
        Business.id == business_id,
        Business.user_id == current_user.id
    ).first()

    if not business:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Business not found"
        )

    # Update fields
    update_data = data.model_dump(exclude_unset=True)

    # Validate business type if provided
    if "business_type" in update_data:
        physical_types = ["restaurant", "retail", "service"]
        online_types = ["fashion", "electronics", "food_grocery", "home_garden", "health_beauty", "other_online"]
        valid_types = physical_types + online_types
        if update_data["business_type"] not in valid_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid business type. Must be one of: {', '.join(valid_types)}"
            )

    for field, value in update_data.items():
        setattr(business, field, value)

    # Update coordinates if location changed
    if any(f in update_data for f in ["zipcode", "city", "state", "country"]):
        lat, lon = get_coordinates(
            business.zipcode, business.city, business.state, business.country
        )
        business.latitude = lat
        business.longitude = lon

    db.commit()
    db.refresh(business)

    return business


@router.delete("/{business_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_business(
    business_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a business"""
    business = db.query(Business).filter(
        Business.id == business_id,
        Business.user_id == current_user.id
    ).first()

    if not business:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Business not found"
        )

    db.delete(business)
    db.commit()


@router.post("/{business_id}/complete-setup", response_model=BusinessResponse)
async def complete_setup(
    business_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark business setup as complete"""
    business = db.query(Business).filter(
        Business.id == business_id,
        Business.user_id == current_user.id
    ).first()

    if not business:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Business not found"
        )

    business.setup_complete = True
    db.commit()
    db.refresh(business)

    return business


def get_business_config(business: Business) -> BusinessConfig:
    """Convert a Business model to a BusinessConfig for the AI trainer"""
    return BusinessConfig(
        business_name=business.name,
        city=business.city,
        state=business.state,
        zipcode=business.zipcode,
        country=business.country,
        timezone=business.timezone
    )


@router.get("/{business_id}/events")
async def get_events(
    business_id: int,
    date: Optional[str] = Query(None, description="Specific date (YYYY-MM-DD)"),
    start_date: Optional[str] = Query(None, description="Start date for range (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for range (YYYY-MM-DD)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get events for a business within a date or date range"""
    business = db.query(Business).filter(
        Business.id == business_id,
        Business.user_id == current_user.id
    ).first()

    if not business:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Business not found"
        )

    # Determine date range
    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
            start = target_date
            end = target_date
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    elif start_date and end_date:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        # Default to current month
        today = datetime.now().date()
        start = today.replace(day=1)
        if today.month == 12:
            end = today.replace(year=today.year + 1, month=1, day=1)
        else:
            end = today.replace(month=today.month + 1, day=1)

    year = start.year
    country = business.country or "US"

    # Fetch all event types IN PARALLEL for speed
    holidays = []
    sports = []
    paydays = []
    school = []
    weather = []

    # Create tasks for parallel fetching
    async def fetch_weather():
        if business.latitude and business.longitude:
            try:
                days_diff = (end - start).days + 1
                weather_data = await get_forecast_weather(
                    latitude=business.latitude,
                    longitude=business.longitude,
                    days=min(days_diff, 14)
                )
                return [
                    w for w in weather_data
                    if start <= datetime.strptime(w["date"], "%Y-%m-%d").date() <= end
                ]
            except Exception as e:
                print(f"Failed to fetch weather: {e}")
        return []

    async def fetch_holidays():
        try:
            holidays_data = await get_holidays(year, country)
            return [
                h for h in holidays_data
                if h.get("date") and start <= datetime.strptime(h["date"], "%Y-%m-%d").date() <= end
            ]
        except Exception as e:
            print(f"Failed to fetch holidays: {e}")
            return []

    async def fetch_sports():
        try:
            sports_data = await get_nfl_games(year)
            return [
                s for s in sports_data
                if s.get("date") and start <= datetime.strptime(s["date"], "%Y-%m-%d").date() <= end
            ]
        except Exception as e:
            print(f"Failed to fetch sports: {e}")
            return []

    def fetch_paydays():
        try:
            payday_dates = get_payday_dates(year)
            return [
                {"date": p.isoformat(), "type": "payday", "name": "Payday"}
                for p in payday_dates
                if start <= p <= end
            ]
        except Exception as e:
            print(f"Failed to fetch paydays: {e}")
            return []

    def fetch_school():
        try:
            school_data = get_school_calendar(year)
            result = []
            for event in school_data:
                event_start = datetime.strptime(event["start_date"], "%Y-%m-%d").date()
                event_end = datetime.strptime(event.get("end_date", event["start_date"]), "%Y-%m-%d").date()
                if event_start <= end and event_end >= start:
                    result.append(event)
            return result
        except Exception as e:
            print(f"Failed to fetch school calendar: {e}")
            return []

    # Run async fetches in parallel, sync fetches immediately
    paydays = fetch_paydays()
    school = fetch_school()

    # Gather async results
    weather, holidays, sports = await asyncio.gather(
        fetch_weather(),
        fetch_holidays(),
        fetch_sports()
    )

    return {
        "holidays": holidays,
        "sports": sports,
        "paydays": paydays,
        "school": school,
        "weather": weather,
        "date_range": {
            "start": start.isoformat(),
            "end": end.isoformat()
        }
    }
