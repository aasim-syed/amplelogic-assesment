# Run with: streamlit run inventory_dashboard_xgb.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from datetime import timedelta

st.set_page_config(page_title="Inventory Forecast (XGBoost)", layout="centered")

st.title("📦 Inventory Forecast & Reorder Planner (XGBoost)")
st.write("Upload your daily sales data to forecast future demand using XGBoost and compute reorder planning metrics.")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload your demand CSV", type=["csv"])

def create_features(df):
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['day'] = df['ds'].dt.day
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['lag_1'] = df['y'].shift(1)
    df['lag_7'] = df['y'].shift(7)
    return df.dropna()

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'date' not in df.columns or 'demand' not in df.columns:
        st.error("Your CSV must have columns: `date` and `demand`.")
    else:
        df = df.rename(columns={"date": "ds", "demand": "y"})
        df['ds'] = pd.to_datetime(df['ds'])
        df = create_features(df)

        # Train-test split
        train = df[:-30]
        test = df[-30:]

        features = ['dayofweek', 'day', 'month', 'year', 'lag_1', 'lag_7']
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        model.fit(train[features], train['y'])

        # Forecast next 30 days recursively
        forecast = []
        last_known = df.iloc[-1].copy()

        for i in range(30):
            future_date = last_known['ds'] + timedelta(days=1)
            new_row = {
                'ds': future_date,
                'dayofweek': future_date.dayofweek,
                'day': future_date.day,
                'month': future_date.month,
                'year': future_date.year,
                'lag_1': last_known['y'],
                'lag_7': df.iloc[-7+i]['y'] if i >= 7 else last_known['y']
            }
            input_df = pd.DataFrame([new_row])
            prediction = model.predict(input_df[features])[0]
            new_row['y'] = prediction
            forecast.append(new_row)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            last_known = pd.Series(new_row)

        forecast_df = pd.DataFrame(forecast)

        # Plotting
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'][:-30], y=df['y'][:-30], name="Actual", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['y'], name="Forecast (XGBoost)", line=dict(color="orange")))
        st.plotly_chart(fig, use_container_width=True)

        avg_demand = forecast_df['y'].mean()
        std_demand = forecast_df['y'].std()

        # Sidebar inputs
        st.sidebar.header("⚙️ Settings")
        lead_time = st.sidebar.slider("Lead Time (in days)", 1, 30, 7)
        z = st.sidebar.slider("Service Level (Z-score)", 1.0, 2.0, 1.65)

        safety_stock = z * std_demand
        reorder_point = avg_demand * lead_time + safety_stock

        st.subheader("📊 Inventory Planning Insights")
        st.metric("🛒 Avg Daily Demand", f"{avg_demand:.2f} units")
        st.metric("📉 Demand Variability", f"{std_demand:.2f}")
        st.metric("🧯 Safety Stock", f"{safety_stock:.2f} units")
        st.metric("🚨 Reorder Point", f"{reorder_point:.2f} units")

        with st.expander("📘 Explanation"):
            st.markdown("""
            - **Avg Daily Demand** → Typical units sold per day.
            - **Std Dev (Variability)** → Fluctuation in demand.
            - **Safety Stock** → Cushion for uncertainty.
            - **Reorder Point** → When to place the next order.
            """)
else:
    st.info("Awaiting CSV upload... Please upload a file with `date` and `demand` columns.")
