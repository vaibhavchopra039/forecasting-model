import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

df_query='''
Select DIM_Claim_Created_Date 'Date',Product_Code, GMC_Name 'GMC Name', count(distinct Claim_ID) 'Sales' , Sum(cast(Claim_Amount as float)) 'Revenue'
FROM [customermart].[Fact_Claim]
where Product_Code in ('NVDIA01','NVDIA03') and YEAR(DIM_Claim_Created_Date)>='2023' and Claim_Status in ('Paid','Approved','Authorized','Claimed','Redeemed') and GMC_Name is not null
group by DIM_Claim_Created_Date,Product_Code, GMC_Name
order by count(distinct Claim_ID) desc
'''


data = pd.read_csv('data.csv')

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Set the date column as the index
data.set_index('Date', inplace=True)

# Plot the sales data
data['Sales'].plot(figsize=(14, 7))
plt.title('Sales over Time')
plt.show()

# Plot the revenue data
data['Revenue'].plot(figsize=(14, 7))
plt.title('Revenue over Time')
plt.show()

# Resample the data to monthly frequency
monthly_data = data.resample('M').sum()

# Display the resampled data
print(monthly_data.head())



# Split the data into training and testing sets
train, test = train_test_split(monthly_data, test_size=0.2, shuffle=False)
print(f"Training set shape: {train.shape}")
print(f"Testing set shape: {test.shape}")

# Define and fit SARIMAX models for both sales and revenue
model_sales = SARIMAX(train['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_revenue = SARIMAX(train['Revenue'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

model_sales_fit = model_sales.fit(disp=False)
model_revenue_fit = model_revenue.fit(disp=False)

# Forecast the sales and revenue on the test set
forecast_sales = model_sales_fit.forecast(steps=len(test))
forecast_revenue = model_revenue_fit.forecast(steps=len(test))

test['Sales Forecast'] = forecast_sales
test['Revenue Forecast'] = forecast_revenue

# Calculate error metrics for sales
mae_sales = mean_absolute_error(test['Sales'], test['Sales Forecast'])
mse_sales = mean_squared_error(test['Sales'], test['Sales Forecast'])

print(f"Sales - Mean Absolute Error: {mae_sales}")
print(f"Sales - Mean Squared Error: {mse_sales}")

# Calculate error metrics for revenue
mae_revenue = mean_absolute_error(test['Revenue'], test['Revenue Forecast'])
mse_revenue = mean_squared_error(test['Revenue'], test['Revenue Forecast'])

print(f"Revenue - Mean Absolute Error: {mae_revenue}")
print(f"Revenue - Mean Squared Error: {mse_revenue}")

# Plot the actual vs forecasted sales
plt.figure(figsize=(14, 7))
plt.plot(train['Sales'], label='Training Sales')
plt.plot(test['Sales'], label='Actual Sales')
plt.plot(test['Sales Forecast'], label='Forecasted Sales')
plt.legend()
plt.show()

# Plot the actual vs forecasted revenue
plt.figure(figsize=(14, 7))
plt.plot(train['Revenue'], label='Training Revenue')
plt.plot(test['Revenue'], label='Actual Revenue')
plt.plot(test['Revenue Forecast'], label='Forecasted Revenue')
plt.legend()
plt.show()



# Forecast future values for sales and revenue
future_forecast_sales = model_sales_fit.forecast(steps=12)
future_forecast_revenue = model_revenue_fit.forecast(steps=12)

# Create a DataFrame to store the forecast
future_dates = pd.date_range(start=test.index[-1], periods=13, freq='M')[1:]
future_df = pd.DataFrame({'Sales Forecast': future_forecast_sales, 'Revenue Forecast': future_forecast_revenue}, index=future_dates)

# Plot the forecasted sales and revenue
plt.figure(figsize=(14, 7))
plt.plot(monthly_data['Sales'], label='Historical Sales')
plt.plot(future_df['Sales Forecast'], label='Future Sales Forecast')
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(monthly_data['Revenue'], label='Historical Revenue')
plt.plot(future_df['Revenue Forecast'], label='Future Revenue Forecast')
plt.legend()
plt.show()


# Function to forecast based on future date and product code
def forecast_sales_and_revenue(future_date, product_code):
    # Convert the future date to datetime
    future_date = pd.to_datetime(future_date)
    
    # Prepare the future exogenous variables
    future_exog = pd.DataFrame(index=[future_date])
    future_exog['Product_Code_' + product_code] = 1
    
    # Forecast sales and revenue
    future_sales = model_sales_fit.get_forecast(steps=1, exog=future_exog).predicted_mean
    future_revenue = model_revenue_fit.get_forecast(steps=1, exog=future_exog).predicted_mean
    
    return future_sales.iloc[0], future_revenue.iloc[0]


future_date = '2024-06-30'
product_code = 'NVDIA03'
forecasted_sales, forecasted_revenue = forecast_sales_and_revenue(future_date, product_code)

print(f"The forecasted sales for {product_code} on {future_date} is {forecasted_sales}")
print(f"The forecasted revenue for {product_code} on {future_date} is {forecasted_revenue}")

