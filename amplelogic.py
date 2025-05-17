# Install prophet: pip install prophet

import pandas as pd
import numpy as np
from prophet import Prophet

# Simulated demand data
df = pd.read_csv('sample_demand_data.csv')  # should have 'date' and 'demand' columns
df = df.rename(columns={'date': 'ds', 'demand': 'y'})

# Fit Prophet model
model = Prophet(daily_seasonality=True)
model.fit(df)

# Forecast for next 30 days
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Get demand forecast
future_demand = forecast[['ds', 'yhat']].tail(30)

# Calculate ROP and Safety Stock
avg_demand = future_demand['yhat'].mean()
std_demand = future_demand['yhat'].std()
lead_time = 7  # in days
z = 1.65  # 95% service level

safety_stock = z * std_demand
reorder_point = avg_demand * lead_time + safety_stock

print("\n--- Inventory Planning Insights ---")
print(f"🛒 Average Daily Demand: {avg_demand:.2f} units/day → This is what you typically sell.")
print(f"📉 Demand Variability (Std Dev): {std_demand:.2f} → Sales can fluctuate this much.")
print(f"🧯 Safety Stock: {safety_stock:.2f} units → Buffer to handle surprises.")
print(f"🚨 Reorder Point: {reorder_point:.2f} units → Reorder when stock drops below this.")

