# data/api_clients.py
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from urllib.parse import urlencode
from ratelimit import limits, sleep_and_retry

logger = logging.getLogger(__name__)

class NOAAClient:
    BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
    CALLS_PER_SECOND = 5
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"token": api_key}
    
    @sleep_and_retry
    @limits(calls=CALLS_PER_SECOND, period=1)
    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make rate-limited API request to NOAA"""
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_stations(self, lat: float, lon: float, radius: float = 10) -> List[Dict]:
        """Get nearby weather stations"""
        params = {
            "extent": f"{lat-radius},{lon-radius},{lat+radius},{lon+radius}",
            "limit": 1000
        }
        return self._make_request("stations", params)["results"]
    
    def get_weather_data(self, 
                        station_id: str,
                        start_date: datetime,
                        end_date: datetime,
                        dataset_id: str = "GHCND") -> pd.DataFrame:
        """
        Fetch weather data from NOAA
        """
        params = {
            "datasetid": dataset_id,
            "stationid": station_id,
            "startdate": start_date.strftime("%Y-%m-%d"),
            "enddate": end_date.strftime("%Y-%m-%d"),
            "limit": 1000,
            "units": "metric"
        }
        
        all_data = []
        while True:
            data = self._make_request("data", params)
            all_data.extend(data["results"])
            
            if "metadata" not in data or data["metadata"].get("resultset").get("count") < 1000:
                break
                
            params["offset"] = len(all_data)
        
        df = pd.DataFrame(all_data)
        return self._process_noaa_data(df)
    
    def _process_noaa_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process NOAA data into standardized format"""
        # Pivot the datatype column to get features as columns
        df_pivot = df.pivot(
            index='date',
            columns='datatype',
            values='value'
        )
        
        # Rename columns to standard names
        column_mapping = {
            'TMAX': 'temp_max',
            'TMIN': 'temp_min',
            'PRCP': 'precipitation',
            'SNOW': 'snowfall',
            'SNWD': 'snow_depth',
            'AWND': 'avg_wind_speed'
        }
        df_pivot.rename(columns=column_mapping, inplace=True)
        
        return df_pivot

class OpenMeteoClient:
    BASE_URL = "https://api.open-meteo.com/v1"
    
    def __init__(self):
        pass  # No API key needed
    
    @sleep_and_retry
    @limits(calls=10, period=1)
    def get_weather_data(self,
                        lat: float,
                        lon: float,
                        start_date: datetime,
                        end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical weather data from Open-Meteo
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": ["temperature_2m", "precipitation", "windspeed_10m", 
                      "winddirection_10m", "pressure_msl", "relative_humidity_2m"]
        }
        
        url = f"{self.BASE_URL}/forecast"
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data["hourly"])
        df.set_index("time", inplace=True)
        
        # Rename columns to standard format
        column_mapping = {
            "temperature_2m": "temperature",
            "precipitation": "precipitation",
            "windspeed_10m": "wind_speed",
            "winddirection_10m": "wind_direction",
            "pressure_msl": "pressure",
            "relative_humidity_2m": "humidity"
        }
        df.rename(columns=column_mapping, inplace=True)
        
        return df

class WeatherAPIManager:
    def __init__(self, config: Dict):
        self.noaa_client = NOAAClient(config["api_keys"]["noaa"])
        self.open_meteo_client = OpenMeteoClient()
    
    def get_combined_weather_data(self,
                                lat: float,
                                lon: float,
                                start_date: datetime,
                                end_date: datetime) -> pd.DataFrame:
        """
        Get weather data from multiple sources and combine them
        """
        # Get NOAA data
        stations = self.noaa_client.get_stations(lat, lon)
        if stations:
            noaa_data = self.noaa_client.get_weather_data(
                stations[0]["id"],
                start_date,
                end_date
            )
        else:
            noaa_data = None
            logger.warning("No NOAA stations found nearby")
        
        # Get Open-Meteo data
        open_meteo_data = self.open_meteo_client.get_weather_data(
            lat, lon, start_date, end_date
        )
        
        # Combine data sources with priority to NOAA data
        if noaa_data is not None:
            combined_data = noaa_data.copy()
            # Fill missing values with Open-Meteo data
            for col in combined_data.columns:
                if col in open_meteo_data.columns:
                    combined_data[col].fillna(open_meteo_data[col], inplace=True)
        else:
            combined_data = open_meteo_data
        
        return combined_data

# Example usage:
if __name__ == "__main__":
    config = {
        "api_keys": {
            "noaa": "your-noaa-api-key"
        }
    }
    
    manager = WeatherAPIManager(config)
    data = manager.get_combined_weather_data(
        lat=40.7128,
        lon=-74.0060,
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )
    print(data.head())