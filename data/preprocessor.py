# data/preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherDataPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.scalers = {}
        self.sequence_length = config['sequence_length']
        self.forecast_horizon = config['forecast_horizon']
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in weather data using domain-specific methods
        """
        logger.info("Handling missing values...")
        
        # For temperature: interpolate with consideration of time of day and season
        if 'temperature' in df.columns:
            df['temperature'] = df['temperature'].interpolate(method='time')
            
        # For precipitation: forward fill for short gaps, zero for longer gaps
        if 'precipitation' in df.columns:
            df['precipitation'] = df['precipitation'].fillna(method='ffill', limit=6)
            df['precipitation'] = df['precipitation'].fillna(0)
            
        # For other parameters: use appropriate interpolation
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].interpolate(method='cubic', limit_direction='both')
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features for better prediction
        """
        logger.info("Creating temporal features...")
        
        # Convert index to datetime if not already
        df.index = pd.to_datetime(df.index)
        
        # Add cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
        
        return df
    
    def _normalize_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Normalize features using appropriate scaling methods for different weather parameters
        """
        logger.info("Normalizing features...")
        
        if is_training:
            # Temperature: Standardization
            self.scalers['temperature'] = StandardScaler()
            # Precipitation: Special handling for zero-inflated data
            self.scalers['precipitation'] = MinMaxScaler()
            # Other features: Standard scaling
            for col in df.columns:
                if col not in self.scalers and col in self.config['features']:
                    self.scalers[col] = StandardScaler()
        
        # Apply scaling
        for col, scaler in self.scalers.items():
            if col in df.columns:
                if is_training:
                    df[col] = scaler.fit_transform(df[[col]])
                else:
                    df[col] = scaler.transform(df[[col]])
        
        return df
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for training the model
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[(i + self.sequence_length):(i + self.sequence_length + self.forecast_horizon)])
        return np.array(X), np.array(y)

    def preprocess(self, 
                  df: pd.DataFrame, 
                  is_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main preprocessing pipeline
        """
        logger.info(f"Starting preprocessing pipeline for {'training' if is_training else 'inference'} data...")
        
        # 1. Handle missing values
        df = self._handle_missing_values(df)
        
        # 2. Create temporal features
        df = self._create_temporal_features(df)
        
        # 3. Normalize features
        df = self._normalize_features(df, is_training)
        
        # 4. Create sequences for model input
        X, y = self._create_sequences(df[self.config['features']].values)
        
        logger.info(f"Preprocessing complete. X shape: {X.shape}, y shape: {y.shape}")
        return X, y