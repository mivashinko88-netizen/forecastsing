import requests

DEFAULT_TIME_BLOCKS = {
    "morning": ("06:00", "11:00"),
    "afternoon": ("11:00", "16:00"),
    "evening": ("16:00", "20:00"),
    "night": ("20:00", "23:59")
}

class BusinessConfig:
    def __init__(
        self,
        business_name: str,
        city: str,
        state: str,
        zipcode: str,
        country: str = "US",
        timezone: str = "America/New_York",
        time_blocks: dict = None
    ):
        self.business_name = business_name
        self.city = city
        self.state = state
        self.zipcode = zipcode
        self.country = country
        self.timezone = timezone
        self.time_blocks = time_blocks or DEFAULT_TIME_BLOCKS
        
        # Convert to lat/long for weather API
        self.latitude, self.longitude = self._get_coordinates()
    
    def _get_coordinates(self) -> tuple:
        """Convert city/zip to lat/long using free API"""
        
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": f"{self.zipcode}, {self.city}, {self.state}, {self.country}",
            "format": "json",
            "limit": 1
        }
        headers = {"User-Agent": "ForecastingApp/1.0"}
        
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
        
        return None, None
    
    def get_time_block(self, time_str: str) -> str:
        """Given a time, return which block it falls into"""
        from datetime import datetime
        
        time_obj = datetime.strptime(time_str, "%H:%M").time()
        
        for block_name, (start, end) in self.time_blocks.items():
            start_time = datetime.strptime(start, "%H:%M").time()
            end_time = datetime.strptime(end, "%H:%M").time()
            
            if start_time <= time_obj < end_time:
                return block_name
        
        return "other"