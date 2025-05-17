# AmpleLogic-assesment
# 📦 Inventory Demand Forecast & Reorder Point Dashboard

This is a Streamlit app that helps you forecast demand and calculate key inventory metrics like safety stock and reorder point using the Prophet forecasting model.

---
## Watch Demo
[▶️ Watch Demo Video](InventoryForecast-Brave2025-05-1719-51-49-ezgif.com-video-to-gif-converter.gif)

## 🚀 Features

- Upload your sales data (`date`, `demand`)
- Forecast next 30 days of demand using  Prophet
- Calculate:
  - Average Daily Demand
  - Demand Variability (Standard Deviation)
  - Safety Stock
  - Reorder Point
- Interactive UI to set lead time and service level
- Live charts using Plotly

---

## 📂 CSV Format

CSV is like this:

```csv
date,demand
2023-01-01,20
2023-01-02,18
2023-01-03,25
...

## Trying on local 

``` cd amplelogic-assesment```

``` pip install --user -r requirements.txt ```

