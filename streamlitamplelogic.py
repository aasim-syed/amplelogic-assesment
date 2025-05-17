# Run with: streamlit run inventory_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go

st.set_page_config(page_title="Inventory Forecast", layout="centered")

st.title("📦 Inventory Forecast & Reorder Planner")
st.write("Upload your daily sales data to forecast future demand and compute reorder planning metrics.")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload your demand CSV", type=["csv"])

if uploaded_file:
    # Read and prepare data
    df = pd.read_csv(uploaded_file)
    if 'date' not in df.columns or 'demand' not in df.columns:
        st.error("Your CSV must have columns: `date` and `demand`.")
    else:
        df = df.rename(columns={"date": "ds", "demand": "y"})
        df['ds'] = pd.to_datetime(df['ds'])

        # Fit model
        model = Prophet(daily_seasonality=True)
        model.fit(df)

        # Forecast future
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Plot actual + forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="Actual", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecast", line=dict(color="green")))
        st.plotly_chart(fig, use_container_width=True)

        # Get forecasted 30 days
        forecast_30 = forecast[['ds', 'yhat']].tail(30)
        avg_demand = forecast_30['yhat'].mean()
        std_demand = forecast_30['yhat'].std()

        # Inputs for business user
        st.sidebar.header("⚙️ Settings")
        lead_time = st.sidebar.slider("Lead Time (in days)", 1, 30, 7)
        z = st.sidebar.slider("Service Level (Z-score)", 1.0, 2.0, 1.65)

        safety_stock = z * std_demand
        reorder_point = avg_demand * lead_time + safety_stock

        st.subheader("📊 Inventory Planning Insights")
        st.metric("🛒 Avg Daily Demand", f"{avg_demand:.2f} units", help="This is what you typically sell daily.")
        st.metric("📉 Demand Variability", f"{std_demand:.2f}", help="Indicates how unpredictable your sales are.")
        st.metric("🧯 Safety Stock", f"{safety_stock:.2f} units", help="Buffer stock to avoid stockouts.")
        st.metric("🚨 Reorder Point", f"{reorder_point:.2f} units", help="Place a new order when stock drops below this.")

        with st.expander("📘 What do these numbers mean?"):
            st.markdown("""
            - **Avg Daily Demand** → Your regular sales volume each day.
            - **Demand Variability (Std Dev)** → How much your sales fluctuate on average.
            - **Safety Stock** → Extra stock to handle unexpected spikes or supplier delays.
            - **Reorder Point** → If your stock drops below this, it’s time to reorder!
            """)

else:
    st.info("Awaiting CSV upload... Please upload a file with `date` and `demand` columns.")
