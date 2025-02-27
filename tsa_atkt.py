"""
# Prac 1,2,3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

CO = pd.read_csv('Crude oil.csv')

def modified_exponential(t, a, b, c):
    return a * np.exp(-b * t) + c

def gompertz_function(t, a, b, c):
    return a * np.exp(-b * np.exp(-c * t))

def logistic_function(t, a, b, c):
    return a / (1 + np.exp(-b * (t - c)))

x_data = np.arange(len(CO))
y_data = CO['Crude-Oil'].values

p0 = [max(y_data), 0.1, min(y_data)]
p0_gom = [max(y_data), 0.1, 0.1]
p0_log = [max(y_data), 0.1, np.median(x_data)]

bounds = ([0, 0, min(y_data)], [2 * max(y_data), 1, max(y_data)])
bounds_gom = ([0, 0, 0], [2 * max(y_data), 1, 1])
bounds_log = ([0, 0, 0], [2 * max(y_data), 1, len(CO)])

popt, pcov = curve_fit(modified_exponential, x_data, y_data, p0=p0, bounds=bounds)

plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Original data')
plt.plot(x_data, modified_exponential(x_data, *popt), 'r-', label='Fitted curve')
plt.xlabel('Year')
plt.ylabel('Crude oil rates')
plt.title('Modified Exponential Curve Fitting')
plt.legend()
plt.show()

"""# Prac 4 Moving Average"""

import pandas as pd
import matplotlib.pyplot as plt

data = {
 'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
 'Sales_000': [125, 145, 186, 131, 151, 192, 137, 157, 198, 143, 163, 204]
}

df = pd.DataFrame(data)

df['3_Month_MA'] = df['Sales_000'].rolling(window=3).mean()

df['Trend'] = df['3_Month_MA'].diff()

df['Seasonal_Variation'] = df['Sales_000'] - df['3_Month_MA']

plt.figure(figsize=(12, 6))
plt.plot(df['Month'], df['Sales_000'], label='Original Sales')
plt.plot(df['Month'], df['3_Month_MA'], label='3-Month Moving Average', linewidth=2, color='orange')
plt.title('Sales with 3-Month Moving Average')
plt.xlabel('Month')
plt.ylabel('Sales ($000)')
plt.legend()
plt.show()

"""# Prac 5 Ratio-to-Trend"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
 'Quarter': ['Q1', 'Q2', 'Q3', 'Q4'],
 'Year1': [150, 200, 250, 300],
 'Year2': [160, 210, 260, 310],
 'Year3': [170, 220, 270, 320]
}

df = pd.DataFrame(data)

# Calculate the trend using the overall average for each year
df['Trend_Year1'] = df[['Year1', 'Year2', 'Year3']].mean(axis=1)
df['Trend_Year2'] = df[['Year1', 'Year2', 'Year3']].mean(axis=1)
df['Trend_Year3'] = df[['Year1', 'Year2', 'Year3']].mean(axis=1)

# Calculate the ratio to trend
df['Ratio_Year1'] = df['Year1'] / df['Trend_Year1']
df['Ratio_Year2'] = df['Year2'] / df['Trend_Year2']
df['Ratio_Year3'] = df['Year3'] / df['Trend_Year3']

# Calculate the seasonal indices
seasonal_indices = {
 'Q1': np.mean([df['Ratio_Year1'][0], df['Ratio_Year2'][0], df['Ratio_Year3'][0]]),
 'Q2': np.mean([df['Ratio_Year1'][1], df['Ratio_Year2'][1], df['Ratio_Year3'][1]]),
 'Q3': np.mean([df['Ratio_Year1'][2], df['Ratio_Year2'][2], df['Ratio_Year3'][2]]),
 'Q4': np.mean([df['Ratio_Year1'][3], df['Ratio_Year2'][3], df['Ratio_Year3'][3]])
}

total_indices = sum(seasonal_indices.values())
normalized_indices = {k: v * 4 / total_indices for k, v in seasonal_indices.items()}

# Deseasonalize the data
df['Deseasonalized_Year1'] = df['Year1'] / df['Quarter'].map(normalized_indices)
df['Deseasonalized_Year2'] = df['Year2'] / df['Quarter'].map(normalized_indices)
df['Deseasonalized_Year3'] = df['Year3'] / df['Quarter'].map(normalized_indices)

# Plotting
plt.figure(figsize=(14, 8))

# Plot original data
plt.subplot(3, 1, 1)
plt.plot(df['Quarter'], df['Year1'], label='Year 1', marker='o')
plt.plot(df['Quarter'], df['Year2'], label='Year 2', marker='o')
plt.plot(df['Quarter'], df['Year3'], label='Year 3', marker='o')
plt.title('Original Data')
plt.xlabel('Quarter')
plt.ylabel('Sales')
plt.legend()

# Plot seasonal indices
plt.subplot(3, 1, 2)
plt.plot(df['Quarter'], [normalized_indices[q] for q in df['Quarter']],
label='Seasonal Index', marker='o', color='orange')
plt.title('Seasonal Indices')
plt.xlabel('Quarter')
plt.ylabel('Index')
plt.legend()

# Plot deseasonalized data
plt.subplot(3, 1, 3)
plt.plot(df['Quarter'], df['Deseasonalized_Year1'], label='Year 1 Deseasonalized', marker='o')
plt.plot(df['Quarter'], df['Deseasonalized_Year2'], label='Year 2 Deseasonalized', marker='o')
plt.plot(df['Quarter'], df['Deseasonalized_Year3'], label='Year 3 Deseasonalized', marker='o')
plt.title('Deseasonalized Data')
plt.xlabel('Quarter')
plt.ylabel('Deseasonalized Sales')
plt.legend()

plt.tight_layout()
plt.show()

"""# Prac 6 Rstio-to-Moving Avg"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
 'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
 'Year1': [150, 180, 200, 220, 250, 270, 300, 320, 350, 370, 400, 420],
 'Year2': [160, 190, 210, 230, 260, 280, 310, 330, 360, 380, 410, 430]
}

df = pd.DataFrame(data)

# Step 1: Calculate Monthly Averages
df['Monthly Average'] = df[['Year1', 'Year2']].mean(axis=1)

# Step 2: Compute Centered Moving Averages
df['Centered Moving Average'] = df['Monthly Average'].rolling(window=2,
center=True).mean()

# Step 3: Calculate the Ratio of Actual to Moving Average
df['Ratio'] = df['Monthly Average'] / df['Centered Moving Average']

# Step 4: Estimate Seasonal Indexes
# Normalize the ratios so they sum to the number of months
sum_ratios = df['Ratio'].sum()
df['Seasonal Index'] = df['Ratio'] * (len(df) / sum_ratios)

# Deseasonalize the data
df['Deseasonalized_Year1'] = df['Year1'] / df['Seasonal Index']
df['Deseasonalized_Year2'] = df['Year2'] / df['Seasonal Index']

plt.figure(figsize=(14, 8))

# Plot original data
plt.subplot(3, 1, 1)
plt.plot(df['Month'], df['Year1'], label='Year 1', marker='o')
plt.plot(df['Month'], df['Year2'], label='Year 2', marker='o')
plt.title('Original Sales Data')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()

# Plot seasonal indices
plt.subplot(3, 1, 2)
plt.plot(df['Month'], df['Seasonal Index'], label='Seasonal Index',
marker='o', color='orange')
plt.title('Seasonal Indices')
plt.xlabel('Month')
plt.ylabel('Index')
plt.legend()

# Plot deseasonalized data
plt.subplot(3, 1, 3)
plt.plot(df['Month'], df['Deseasonalized_Year1'], label='Year 1 Deseasonalized', marker='o')
plt.plot(df['Month'], df['Deseasonalized_Year2'], label='Year 2 Deseasonalized', marker='o')
plt.title('Deseasonalized Data')
plt.xlabel('Month')
plt.ylabel('Deseasonalized Sales')
plt.legend()
plt.tight_layout()
plt.show()

"""# Prac 7 Link Relative"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
 'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
 'Year1': [150, 180, 200, 220, 250, 270, 300, 320, 350, 370, 400, 420],
 'Year2': [160, 190, 210, 230, 260, 280, 310, 330, 360, 380, 410, 430],
 'Year3': [170, 200, 220, 240, 270, 290, 320, 340, 370, 390, 420, 440]
}

df = pd.DataFrame(data)

# Calculate the link relatives
df['Link_Relative_Y2'] = df['Year2'] / df['Year1']
df['Link_Relative_Y3'] = df['Year3'] / df['Year2']

# Calculate the average link relatives for each month
df['Average_Link_Relative'] = df[['Link_Relative_Y2', 'Link_Relative_Y3']].mean(axis=1)

# Normalize the seasonal indices
seasonal_indices = df['Average_Link_Relative']
seasonal_indices /= seasonal_indices.sum()
seasonal_indices *= 12

# Deseasonalize the data
df['Deseasonalized_Year1'] = df['Year1'] / seasonal_indices
df['Deseasonalized_Year2'] = df['Year2'] / seasonal_indices
df['Deseasonalized_Year3'] = df['Year3'] / seasonal_indices

# Plotting
plt.figure(figsize=(14, 8))

# Plot original data
plt.subplot(3, 1, 1)
plt.plot(df['Month'], df['Year1'], label='Year 1', marker='o')
plt.plot(df['Month'], df['Year2'], label='Year 2', marker='o')
plt.plot(df['Month'], df['Year3'], label='Year 3', marker='o')
plt.title('Original Data')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()

# Plot seasonal indices
plt.subplot(3, 1, 2)
plt.plot(df['Month'], seasonal_indices, label='Seasonal Index', marker='o', color='orange')
plt.title('Seasonal Indices')
plt.xlabel('Month')
plt.ylabel('Index')
plt.legend()

# Plot deseasonalized data
plt.subplot(3, 1, 3)
plt.plot(df['Month'], df['Deseasonalized_Year1'], label='Year 1 Deseasonalized', marker='o')
plt.plot(df['Month'], df['Deseasonalized_Year2'], label='Year 2 Deseasonalized', marker='o')
plt.plot(df['Month'], df['Deseasonalized_Year3'], label='Year 3 Deseasonalized', marker='o')
plt.title('Deseasonalized Data')
plt.xlabel('Month')
plt.ylabel('Deseasonalized Sales')
plt.legend()
plt.tight_layout()
plt.show()

"""# Prac 8 Variance diff method"""

import numpy as np
import matplotlib.pyplot as plt

# Given time series data
time_series = [47, 64, 23, 71, 38, 64, 55, 41, 59, 48]

# Calculate differences
differences = np.diff(time_series)

# Calculate mean of differences
mean_diff = np.mean(differences)

# Calculate variance of differences
var_diff = np.var(differences, ddof=1)

# Calculate variance of random components
var_random = var_diff / 2

# Print results
print(f"Mean of Differences: {mean_diff}")
print(f"Variance of Differences: {var_diff}")
print(f"Variance of Random Components: {var_random}")

# Plot the differences
plt.plot(differences, marker='o')
plt.title('Differences Between Successive Observations')
plt.xlabel('Time')
plt.ylabel('Difference Value')
plt.grid(True)
plt.show()

# Create a figure and axis
fig, ax1 = plt.subplots()

# Plot the time series on the primary y-axis
color = 'tab:blue'
ax1.set_xlabel('Time')
ax1.set_ylabel('Time Series', color=color)
ax1.plot(time_series, marker='o', color=color, label='Time Series')
ax1.tick_params(axis='y', labelcolor=color)

# Create a secondary y-axis for the differences
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Differences', color=color)
ax2.plot(range(1, len(time_series)), differences, marker='x',
linestyle='--', color=color, label='Differences')
ax2.tick_params(axis='y', labelcolor=color)

# Add title and grid
plt.title('Time Series and Differences')
fig.tight_layout() # Adjust layout to make room for both y-axes
plt.grid(True)
plt.show()

"""# Prac 9 Forecasting exp smoothing"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

data = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170, 170,
 158, 133, 114, 140, 145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166, 171, 180, 193, 181,
 183, 218, 230, 242, 209, 191, 172, 194, 196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
 204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 242, 233, 267, 269, 270, 315, 364, 347, 312,
 274, 237, 278, 284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306, 315, 301, 356, 348, 355,
 422, 465, 467, 404, 347, 305, 336, 340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337, 360,
 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405, 417, 391, 419, 461, 472, 535, 622, 606, 508,
 461, 390, 432]

# Convert data to pandas DataFrame
df = pd.DataFrame(data, columns=['value'])

# Define the model
model = ExponentialSmoothing(df['value'], trend='add', seasonal='add', seasonal_periods=12)

# Fit the model
fit = model.fit()

# Forecast future values
forecast = fit.forecast(12)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(df['value'], label='Original')
plt.plot(fit.fittedvalues, label='Fitted', linestyle='--')
plt.plot(forecast, label='Forecast', linestyle='--')
plt.legend()
plt.show()

# Print forecasted values
print(forecast)

"""# Prac 10 Short term forecasting methods"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

data = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170, 170,
 158, 133, 114, 140, 145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166, 171, 180, 193, 181]

# Convert data to pandas DataFrame
df = pd.DataFrame(data, columns=['value'])

# Fit the ARIMA model
model = ARIMA(df['value'], order=(5, 1, 1)) # (p,d,q) order
fit = model.fit()

# Forecast future values
forecast = fit.forecast(steps=10) # Forecast next 10 periods

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(df['value'], label='Original')
plt.plot(fit.fittedvalues, label='Fitted', linestyle='--')
plt.plot(range(len(df), len(df) + 10), forecast, label='Forecast',
linestyle='--')
plt.legend()
plt.show()

# Print forecasted values
print(forecast)



