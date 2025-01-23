# data/data_loader.py
import requests
import pandas as pd
from datetime import datetime

class WeatherDataLoader:
    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    def get_historical_data(self, lat: float, lon: float, 
                          start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical weather data"""
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,precipitation,windspeed_10m,relativehumidity_2m,pressure_msl"  # Changed this line
        }
        
        # Debug the API call
        response = requests.get(self.base_url, params=params)
        print("API URL:", response.url)  # See what URL we're calling
        print("Response status:", response.status_code)
        
        data = response.json()
        print("Response data keys:", data.keys())  # See what we got back
        
        if 'hourly' not in data:
            print("Error in response:", data)
            raise ValueError("Invalid API response")
            
        df = pd.DataFrame(data["hourly"])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # Rename columns for clarity
        df.columns = ['temperature', 'precipitation', 'wind_speed', 
                     'humidity', 'pressure']
        
        # Daily aggregation
        daily_df = df.resample('D').agg({
            'temperature': 'mean',
            'precipitation': 'sum',
            'wind_speed': 'mean',
            'humidity': 'mean',
            'pressure': 'mean'
        })
        
        print("Processed data shape:", daily_df.shape)
        print("Sample data:", daily_df.head())
        
        return daily_df
# Example usage
if __name__ == "__main__":
    loader = WeatherDataLoader()
    
    # Get NYC weather data for 2015-2016
    data = loader.get_historical_data(
        lat=40.7128, 
        lon=-74.0060,
        start_date="2015-01-01",
        end_date="2016-12-31"
    )
    print(data.head())