# test_weather_prediction.py
from data.data_loader import WeatherDataLoader
from models.weather_model import WeatherPredictor
import matplotlib.pyplot as plt
import numpy as np

# Initialize
loader = WeatherDataLoader()
predictor = WeatherPredictor()

# Get training data (2007-2016)
train_data = loader.get_historical_data(
    lat=40.7128, 
    lon=-74.0060,
    start_date="1977-01-01",
    end_date="2016-12-31"
)

print(f"Training data shape: {train_data.shape}\n")
print("Sample data:", train_data.head(), "\n")

# Train model
predictor.train(train_data)

# Get predictions
predictions = predictor.predict(train_data)

# Get actual 2017 data
actual_2017 = loader.get_historical_data(
    lat=40.7128,
    lon=-74.0060,
    start_date="2017-01-01",
    end_date="2017-12-31"
)

# Calculate and display metrics
print("\nModel Performance:")
metrics = predictor.evaluate(actual_2017, predictions)
for param, values in metrics.items():
    print(f"\n{param}:")
    for metric, value in values.items():
        print(f"{metric}: {value:.4f}")

# Plot all parameters
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()

params = ['temperature', 'precipitation', 'wind_speed', 'humidity', 'pressure']
param_units = {
    'temperature': '°C',
    'precipitation': 'mm',
    'wind_speed': 'm/s',
    'humidity': '%',
    'pressure': 'hPa'
}

for i, param in enumerate(params):
    ax = axes[i]
    
    # Plot actual vs predicted
    ax.plot(actual_2017.index, actual_2017[param], label='Actual', color='blue', alpha=0.7)
    ax.plot(predictions.index, predictions[param], label='Predicted', color='orange', alpha=0.7)
    
    # Add labels and title
    ax.set_title(f'2017 {param.capitalize()}: Predicted vs Actual')
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{param.capitalize()} ({param_units[param]})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45)

# Adjust layout
plt.tight_layout()
plt.show()

# Plot correlation between actual and predicted values
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()

for i, param in enumerate(params):
    ax = axes[i]
    
    # Create scatter plot
    ax.scatter(actual_2017[param], predictions[param], alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(actual_2017[param].min(), predictions[param].min())
    max_val = max(actual_2017[param].max(), predictions[param].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Add labels and title
    ax.set_title(f'{param.capitalize()} Correlation')
    ax.set_xlabel(f'Actual {param} ({param_units[param]})')
    ax.set_ylabel(f'Predicted {param} ({param_units[param]})')
    ax.grid(True, alpha=0.3)
    ax.legend()

# Adjust layout
plt.tight_layout()
plt.show()