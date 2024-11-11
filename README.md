## Developed By : Pravin Raj A
## Register No. : 212222240079

# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 
### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the Bit-Coin time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the dataset
file_path = '/content/BTC-USD(1).csv'
data = pd.read_csv(file_path)

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Set 'Date' as the index
data.set_index('Date', inplace=True)

#Resample to monthly averages for 'Close' column, which represents Bitcoin prices
monthly_data = data['Close'].resample('M').mean()

# Drop any NaN values
monthly_data = monthly_data.dropna()

# --- 1. ACF and PACF Plots ---
# ... (rest of your code remains the same, but using 'monthly_data')

# Plot ACF
plt.figure(figsize=(10, 6))
plot_acf(monthly_data, lags=20)
plt.title('Autocorrelation Function (ACF)')
plt.show()

# Plot PACF
plt.figure(figsize=(10, 6))
plot_pacf(monthly_data, lags=20)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# --- 2. Moving Average Model (MA) ---

# Fit the MA model (order q=2, for instance)
ma_model = ARIMA(monthly_data, order=(0, 0, 2)).fit()

# Make predictions for the last 12 months
ma_predictions = ma_model.predict(start=len(monthly_data) - 12, end=len(monthly_data) - 1)

# Plot the Moving Average Model predictions
plt.figure(figsize=(10, 6))
plt.plot(monthly_data, label='Original Data', color='blue')
plt.plot(ma_predictions, label='MA Model Predictions', color='green')
plt.title('Moving Average (MA) Model Predictions')
plt.legend(loc='best')
plt.show()

# --- 3. Plot Transformed Dataset ---

# Apply log transformation to the dataset (to make trends more visible)
transformed_data = np.log(monthly_data)

# Plot the transformed dataset
plt.figure(figsize=(10, 6))
plt.plot(transformed_data, label='Log-Transformed Data', color='purple')
plt.title('Log-Transformed Monthly Average Prices')
plt.xlabel('Date')
plt.ylabel('Log(Average Price)')
plt.legend(loc='best')
plt.show()

# --- 4. Exponential Smoothing ---

# Fit the Exponential Smoothing model (with trend and seasonal components)
exp_smoothing_model = ExponentialSmoothing(monthly_data, trend='add', seasonal='add', seasonal_periods=12).fit()

# Make predictions for the next 12 months
exp_smoothing_predictions = exp_smoothing_model.forecast(12)

# Plot the Exponential Smoothing predictions
plt.figure(figsize=(10, 6))
plt.plot(monthly_data, label='Original Data', color='blue')
plt.plot(exp_smoothing_predictions, label='Exponential Smoothing Forecast', color='red')
plt.title('Exponential Smoothing Forecast')
plt.legend(loc='best')
plt.show()

```
### OUTPUT:
#### Moving Average
![download](https://github.com/user-attachments/assets/6bcf69cd-ca4f-4e42-ba89-69af2cdb9881)

#### Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
![download](https://github.com/user-attachments/assets/9c485fab-7d77-4150-985a-4942e035bacf)
![download](https://github.com/user-attachments/assets/a01be1d8-35ea-47b3-ac58-a3545b35ec09)

#### Log-Transformed 
![download](https://github.com/user-attachments/assets/be7a2dec-abd1-4dd0-bb9a-29f8cf3fbe8a)

#### Exponential Smoothing Forecast
![download](https://github.com/user-attachments/assets/4cd55122-55db-4658-b209-8396030102b2)

### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
