# api/weather_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List

from models.weather_model import WeatherPredictor
from data.data_loader import WeatherDataLoader

app = FastAPI(title="Weather Prediction API")

class WeatherRequest(BaseModel):
    latitude: float
    longitude: float
    end_date: str  # Format: YYYY-MM-DD

class WeatherResponse(BaseModel):
    prediction_start: str
    prediction_end: str
    predictions: Dict[str, List[float]]  # Each weather parameter will have array of values

@app.post("/predict_future/")
async def predict_future_weather(request: WeatherRequest):
    try:
        # Calculate prediction range
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        pred_start = end_date + timedelta(days=30)
        pred_end = end_date + timedelta(days=365)  # 12 months
        
        # Calculate training period (10 years before end_date)
        train_start = end_date - timedelta(days=365 * 10)
        
        # Initialize components
        loader = WeatherDataLoader()
        predictor = WeatherPredictor()
        
        # Get training data
        train_data = loader.get_historical_data(
            lat=request.latitude,
            lon=request.longitude,
            start_date=train_start.strftime("%Y-%m-%d"),
            end_date=request.end_date
        )
        
        # Train model
        predictor.train(train_data)
        
        # Make predictions for future period
        prediction_days = (pred_end - pred_start).days + 1
        predictions = predictor.predict(train_data, days_to_predict=prediction_days)
        
        # Convert to dictionary with arrays
        response_dict = {
            'prediction_start': pred_start.strftime("%Y-%m-%d"),
            'prediction_end': pred_end.strftime("%Y-%m-%d"),
            'predictions': {
                'temperature': predictions['temperature'].tolist(),
                'precipitation': predictions['precipitation'].tolist(),
                'wind_speed': predictions['wind_speed'].tolist(),
                'humidity': predictions['humidity'].tolist(),
                'pressure': predictions['pressure'].tolist()
            }
        }
        
        return response_dict
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)