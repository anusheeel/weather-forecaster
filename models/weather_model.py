# models/weather_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class WeatherPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def prepare_features(self, dates, weather_data=None):
        """Enhanced feature creation focusing on extreme events"""
        features = pd.DataFrame({
            'day_sin': np.sin(2 * np.pi * dates.dayofyear/365),
            'day_cos': np.cos(2 * np.pi * dates.dayofyear/365),
            'month_sin': np.sin(2 * np.pi * dates.month/12),
            'month_cos': np.cos(2 * np.pi * dates.month/12)
        })
        
        if weather_data is not None:
            # Rolling statistics with multiple windows
            windows = [3, 7, 14, 30]
            for col in weather_data.columns:
                # Basic rolling stats
                for window in windows:
                    features[f'{col}_mean_{window}d'] = weather_data[col].rolling(window, min_periods=1).mean()
                    features[f'{col}_std_{window}d'] = weather_data[col].rolling(window, min_periods=1).std()
                
                # Extreme value indicators
                features[f'{col}_max_7d'] = weather_data[col].rolling(7, min_periods=1).max()
                features[f'{col}_min_7d'] = weather_data[col].rolling(7, min_periods=1).min()
                
                # Rate of change features
                features[f'{col}_change_1d'] = weather_data[col].diff().fillna(0)
                features[f'{col}_change_7d'] = weather_data[col].diff(7).fillna(0)
                
                # Acceleration features
                features[f'{col}_acc'] = features[f'{col}_change_1d'].diff().fillna(0)
            
            # Weather pattern features
            features['temp_pressure_ratio'] = (weather_data['temperature'] / weather_data['pressure']).fillna(0)
            features['temp_humidity_ratio'] = (weather_data['temperature'] / weather_data['humidity'].clip(1)).fillna(0)
            
            # Extreme weather indicators
            features['high_temp'] = (weather_data['temperature'] > weather_data['temperature'].quantile(0.9)).astype(int)
            features['low_temp'] = (weather_data['temperature'] < weather_data['temperature'].quantile(0.1)).astype(int)
            features['high_wind'] = (weather_data['wind_speed'] > weather_data['wind_speed'].quantile(0.9)).astype(int)
            
            # Handle missing values
            for col in features.columns:
                features[col] = pd.to_numeric(features[col], errors='coerce')
                features[col] = features[col].ffill().bfill().fillna(0)
        
        return features
    
    def get_model(self, param):
        """Get appropriate model for each parameter"""
        if param == 'precipitation':
            # Use Gradient Boosting for precipitation (better with sparse events)
            return GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                loss='huber',  # More robust to outliers
                random_state=42
            )
        else:
            # Use Random Forest for other parameters
            return RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42
            )
    
    def train(self, train_data):
        """Train model with validation-based early stopping"""
        features = self.prepare_features(train_data.index, train_data)
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        scaled_features = self.scalers['features'].fit_transform(features)
        scaled_features = pd.DataFrame(scaled_features, columns=features.columns, index=features.index)
        
        for column in train_data.columns:
            print(f"\nTraining model for {column}")
            
            # Scale target
            self.scalers[column] = StandardScaler()
            scaled_target = self.scalers[column].fit_transform(train_data[[column]])
            
            # Split data with time-based validation
            train_size = int(len(scaled_features) * 0.8)
            train_features = scaled_features[:train_size]
            train_target = scaled_target[:train_size]
            val_features = scaled_features[train_size:]
            val_target = scaled_target[train_size:]
            
            # Train model
            model = self.get_model(column)
            model.fit(train_features, train_target.ravel())
            self.models[column] = model
            
            # Print validation metrics
            val_pred = model.predict(val_features)
            val_rmse = np.sqrt(np.mean((val_pred - val_target.ravel()) ** 2))
            print(f"Validation RMSE for {column}: {val_rmse:.4f}")
            
            # Print feature importance
            importance = pd.DataFrame({
                'feature': features.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"\nTop 5 important features for {column}:")
            print(importance.head())
    
    def predict(self, input_data, days_to_predict=365):
        """Make predictions with emphasis on extreme values"""
        future_dates = pd.date_range(
            input_data.index[-1] + pd.Timedelta(days=1),
            periods=days_to_predict,
            freq='D'
        )
        
        predictions = pd.DataFrame(index=future_dates, columns=input_data.columns)
        current_weather = input_data.copy()
        
        for date in future_dates:
            # Prepare and scale features
            current_features = self.prepare_features(pd.DatetimeIndex([date]), current_weather)
            scaled_features = self.scalers['features'].transform(current_features)
            
            # Predict each parameter
            for column in input_data.columns:
                pred = self.models[column].predict(scaled_features)[0]
                
                # Inverse transform
                pred_reshaped = pred.reshape(-1, 1)
                pred_value = self.scalers[column].inverse_transform(pred_reshaped)[0][0]
                
                # Ensure non-negative values for applicable parameters
                if column in ['precipitation', 'wind_speed', 'humidity']:
                    pred_value = max(0, pred_value)
                
                predictions.loc[date, column] = pred_value
            
            # Update current weather data
            current_weather = pd.concat([current_weather[1:], predictions.loc[[date]]])
        
        return predictions

    def evaluate(self, true_data, pred_data):
        """Calculate detailed evaluation metrics"""
        metrics = {}
        for column in true_data.columns:
            mse = np.mean((true_data[column] - pred_data[column]) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(true_data[column] - pred_data[column]))
            max_error = np.max(np.abs(true_data[column] - pred_data[column]))
            r2 = 1 - np.sum((true_data[column] - pred_data[column]) ** 2) / np.sum((true_data[column] - true_data[column].mean()) ** 2)
            
            metrics[column] = {
                'RMSE': rmse,
                'MSE': mse,
                'MAE': mae,
                'MaxError': max_error,
                'R2': r2
            }
        return metrics