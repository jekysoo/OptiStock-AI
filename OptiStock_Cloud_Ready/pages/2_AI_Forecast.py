import os  
import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
import re
warnings.filterwarnings('ignore')

st.set_page_config(page_title="OptiStock AI", page_icon="🤖", layout="wide")
st.title("🤖 OptiStock AI: Advanced Hybrid Forecasting")
st.markdown("*Master's Thesis Level Model: Combining Prophet (Trend/Seasonality) with XGBoost (Residuals).*")

@st.cache_data
def load_enterprise_data_v2():
    if os.path.exists('enterprise_database.db'):
        db_path = 'enterprise_database.db'
    else:
        db_path = '/content/drive/MyDrive/Smart_inventory_project/Data/Processed/enterprise_database.db'
        
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM monthly_inventory", conn)
    conn.close()

    df['Out_Qty'] = df['Out_Qty'].astype(str).str.replace(' ', '').str.replace(',', '.')
    df['Out_Qty'] = pd.to_numeric(df['Out_Qty'], errors='coerce').fillna(0)

    def parse_date_ultra(filename):
        try:
            fname = str(filename).lower()
            year_match = re.search(r'(202[0-9])', fname)
            if not year_match: return pd.NaT
            year = year_match.group(1)
            month_dict = {'янв': '01', 'фев': '02', 'мар': '03', 'апр': '04', 'май': '05', 'мая': '05', 'июн': '06', 'июл': '07', 'авг': '08', 'сен': '09', 'окт': '10', 'ноя': '11', 'дек': '12'}
            month = '12' 
            for key, val in month_dict.items():
                if key in fname: month = val; break
            return pd.to_datetime(f"{year}-{month}-01")
        except: return pd.NaT

    df['Date'] = df['Source_File'].apply(parse_date_ultra)
    return df.dropna(subset=['Date']).sort_values('Date')

df_clean = load_enterprise_data_v2()

st.sidebar.markdown("---")
st.sidebar.header("📊 Database Health")
if not df_clean.empty:
    latest_date = df_clean['Date'].max().strftime('%B %Y')
    st.sidebar.success(f"✅ Data connected securely.")
    st.sidebar.info(f"📅 Latest records: **{latest_date}**")

code_col = 'Code' if 'Code' in df_clean.columns else 'code'
if code_col in df_clean.columns:
    unique_items = df_clean.drop_duplicates(subset=[code_col]).copy()
    unique_items['Display_Name'] = "[SKU: " + unique_items[code_col].astype(str) + "] " + unique_items['Name']
    display_to_code = dict(zip(unique_items['Display_Name'], unique_items[code_col]))
    all_display_names = sorted(unique_items['Display_Name'].tolist())
else:
    all_display_names = sorted(df_clean['Name'].unique().tolist())

col_ui1, col_ui2 = st.columns([2, 1])
selected_display = col_ui1.selectbox("🎯 Select Target Product for Visual Analysis:", all_display_names)
horizon = col_ui2.slider("📆 Forecast Horizon (Months):", min_value=3, max_value=12, value=6, step=3)

target_code = display_to_code[selected_display] if code_col in df_clean.columns else None
if target_code: df_item = df_clean[df_clean[code_col] == target_code].groupby('Date')['Out_Qty'].sum().reset_index()
else: df_item = df_clean[df_clean['Name'] == selected_display].groupby('Date')['Out_Qty'].sum().reset_index()

if not df_item.empty:
    all_months = pd.date_range(start=df_item['Date'].min(), end=df_item['Date'].max(), freq='MS')
    df_item = df_item.set_index('Date').reindex(all_months, fill_value=0).rename_axis('Date').reset_index()

if len(df_item) < 6:
    st.warning("⚠️ Insufficient historical data to train the Hybrid AI model for this specific item.")
else:
    tab1, tab2 = st.tabs(["📈 Hybrid Forecast", "🔥 Seasonality Heatmap"])

    with tab1:
        prophet_df = df_item.rename(columns={'Date': 'ds', 'Out_Qty': 'y'})
        model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model_prophet.fit(prophet_df)
        
        prophet_train_pred = model_prophet.predict(prophet_df)
        df_item['Prophet_Pred'] = prophet_train_pred['yhat'].values
        df_item['Residuals'] = df_item['Out_Qty'] - df_item['Prophet_Pred']
        
        df_item['Month'] = df_item['Date'].dt.month
        df_item['Lag_1'] = df_item['Out_Qty'].shift(1).fillna(0)
        df_item['Lag_3'] = df_item['Out_Qty'].shift(3).fillna(0)
        df_item['Rolling_Mean'] = df_item['Out_Qty'].shift(1).rolling(window=3).mean().fillna(0)
        
        X_train = df_item[['Month', 'Lag_1', 'Lag_3', 'Rolling_Mean']]
        y_train_residuals = df_item['Residuals']
        model_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model_xgb.fit(X_train, y_train_residuals)
        
        future_dates_df = model_prophet.make_future_dataframe(periods=horizon, freq='MS')
        future_prophet_pred = model_prophet.predict(future_dates_df)
        
        future_forecast = []
        last_lag_1 = df_item['Out_Qty'].iloc[-1]
        last_lag_3 = df_item['Out_Qty'].iloc[-3] if len(df_item) >=3 else last_lag_1
        last_rolling = df_item['Out_Qty'].tail(3).mean()
        future_dates_only = future_dates_df.tail(horizon).copy()
        
        for i, row in future_dates_only.iterrows():
            X_future = pd.DataFrame({'Month': [row['ds'].month], 'Lag_1': [last_lag_1], 'Lag_3': [last_lag_3], 'Rolling_Mean': [last_rolling]})
            pred_residual = model_xgb.predict(X_future)[0]
            prophet_base = future_prophet_pred.loc[future_prophet_pred['ds'] == row['ds'], 'yhat'].values[0]
            final_pred = max(0, prophet_base + pred_residual)
            future_forecast.append(final_pred)
            
            last_lag_3 = last_lag_1
            last_lag_1 = final_pred
            last_rolling = (last_rolling * 2 + final_pred) / 3
            
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_item['Date'], df_item['Out_Qty'], marker='o', color='black', label='Actual Sales')
        ax.plot(df_item['Date'], np.maximum(df_item['Prophet_Pred'], 0), color='blue', linestyle=':', label='Prophet Base Trend')
        ax.plot(future_dates_only['ds'], future_forecast, marker='s', color='red', linewidth=2, label=f'Hybrid Forecast (Next {horizon} Months)')
        ax.set_title(f"Demand Forecast for {selected_display[:40]}...", fontsize=14, fontweight='bold')
        ax.set_xlabel("Timeline (Years)")
        ax.set_ylabel("Sales Quantity")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    with tab2:
        df_item['Year'] = df_item['Date'].dt.year
        heatmap_data = df_item.pivot_table(index='Year', columns='Month', values='Out_Qty', aggfunc='sum').fillna(0)
        for m in range(1, 13):
            if m not in heatmap_data.columns: heatmap_data[m] = 0
        heatmap_data = heatmap_data.sort_index(axis=1)
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fig_heat = px.imshow(heatmap_data, labels=dict(x="Month", y="Year", color="Sales Qty"), x=month_names, y=heatmap_data.index.astype(str), color_continuous_scale="Reds", text_auto=True, aspect="auto")
        st.plotly_chart(fig_heat, use_container_width=True)
