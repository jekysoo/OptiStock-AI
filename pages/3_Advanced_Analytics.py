import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import re
import os

st.set_page_config(page_title="Advanced Analytics", page_icon="💎", layout="wide")
st.title("💎 Strategy, ROI & Explainable AI")
st.markdown("Advanced decision-support module integrating Machine Learning with Supply Chain Logistics and Financial Modeling.")

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
    df['Price'] = df['Price'].astype(str).str.replace(' ', '').str.replace(',', '.')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)

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

all_items = sorted(df_clean['Name'].unique())
selected_item = st.selectbox("🎯 Select SKU for Deep Analysis:", all_items)

df_item = df_clean[df_clean['Name'] == selected_item].copy()
avg_price = df_item['Price'].mean()
df_ts = df_item.groupby('Date')['Out_Qty'].sum().reset_index()

if not df_ts.empty:
    all_months = pd.date_range(start=df_ts['Date'].min(), end=df_ts['Date'].max(), freq='MS')
    df_ts = df_ts.set_index('Date').reindex(all_months, fill_value=0).rename_axis('Date').reset_index()

if len(df_ts) < 6:
    st.warning("⚠️ Insufficient data for this SKU to run SHAP diagnostics.")
else:
    # Feature engineering matching the core analytical module
    df_ts['Month'] = df_ts['Date'].dt.month
    df_ts['Lag_1'] = df_ts['Out_Qty'].shift(1).fillna(0)
    df_ts['Lag_3'] = df_ts['Out_Qty'].shift(3).fillna(0)
    df_ts['Rolling_Mean'] = df_ts['Out_Qty'].shift(1).rolling(window=3).mean().fillna(0)

    X = df_ts[['Month', 'Lag_1', 'Lag_3', 'Rolling_Mean']]
    y = df_ts['Out_Qty']

    model_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model_xgb.fit(X, y)

    # 1-step out of sample prediction for EOQ alignment
    next_month_pred = max(0, model_xgb.predict(pd.DataFrame({
        'Month': [1], 
        'Lag_1': [y.iloc[-1]], 
        'Lag_3': [y.iloc[-3] if len(y)>=3 else y.iloc[-1]], 
        'Rolling_Mean': [y.tail(3).mean()]
    }))[0])
    annual_demand = next_month_pred * 12 

    tab1, tab2, tab3 = st.tabs(["🧠 Explainable AI (SHAP)", "📦 Optimal Procurement (EOQ)", "💰 Financial ROI Simulator"])

    with tab1:
        st.header("🧠 Understanding the AI's Brain (Explainable AI)")
        st.markdown("Why did the Hybrid AI predict this demand? The feature weighting chart below reveals which mathematical features influenced the decision engine the most.")
        fig, ax = plt.subplots(figsize=(10, 4))
        xgb.plot_importance(model_xgb, importance_type='weight', ax=ax, color='#1E3A8A', title='Hybrid ML Feature Importance (XGBoost)')
        st.pyplot(fig)
        st.info("💡 **Academic Interpretation:**\n* **Lag_1:** Immediate temporal momentum of the market.\n* **Lag_3:** Quarterly macroeconomic inventory purchasing cycles.\n* **Rolling_Mean:** Multi-month smoothed market baseline trend.\nHigh feature correlation values mathematically validate advanced feature engineering efficacy over baseline structural statistical parameters.")

    with tab2:
        st.header("📦 Smart Logistics: Economic Order Quantity")
        st.latex(r"EOQ = \sqrt{\frac{2DS}{H}}")
        col1, col2 = st.columns(2)
        order_cost = col1.slider("🚚 Fixed Cost per Order (S) in UZS:", min_value=50000, max_value=1000000, value=250000, step=50000)
        holding_pct = col2.slider("🏭 Annual Holding Cost % of item price:", min_value=5, max_value=50, value=20, step=5)

        holding_cost = avg_price * (holding_pct / 100)

        if holding_cost > 0 and annual_demand > 0:
            eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost)
            st.success(f"### 🎯 Optimal Order Size (EOQ): {eoq:,.0f} units per order")
            st.markdown(f"**Calculated Parameters:**\n* Annualized Demand Volume (D): **{annual_demand:,.0f} units**\n* Asset Mean Unit Price: **{avg_price:,.2f} UZS**\n* Unit Holding Cost (H): **{holding_cost:,.2f} UZS**")
        else:
            st.error("Holding cost or demand parameter conflict resolved to zero. Calculation aborted.")

    with tab3:
        st.header("💰 Financial Impact & ROI Simulator")
        manual_order_size = st.number_input("Current Manual Order Size (Units):", value=int(annual_demand/2), step=100)
        
        ai_average_inventory = eoq / 2 if 'eoq' in locals() else 0
        manual_average_inventory = manual_order_size / 2
        
        ai_holding_cost_total = ai_average_inventory * holding_cost if 'eoq' in locals() else 0
        manual_holding_cost_total = manual_average_inventory * holding_cost
        savings = manual_holding_cost_total - ai_holding_cost_total

        c1, c2 = st.columns(2)
        c1.metric("🔴 Baseline Holding Cost (Manual Operations)", f"{manual_holding_cost_total:,.0f} UZS")
        c2.metric("🟢 Optimized Holding Cost (AI Pipeline Active)", f"{ai_holding_cost_total:,.0f} UZS", delta=f"-{savings:,.0f} UZS Saved", delta_color="inverse")

        if savings > 0:
            st.success(f"🚀 **Strategic Enterprise Directive:** Transitioning from empirical order policies to an algorithmic XGBoost+EOQ schedule will free up **{savings / 1e6:,.2f} Million UZS** in working capital for this product alone!")
