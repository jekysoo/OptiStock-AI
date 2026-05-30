import streamlit as st
import sqlite3
import pandas as pd
import os

# Application page configuration
st.set_page_config(page_title="OptiStock AI", page_icon="📦", layout="wide")

st.title("📦 OptiStock AI: Enterprise Decision Platform")
st.markdown("### Intelligent B2B Supply Chain & Inventory Management Framework")

# Smart path resolution for multi-environment deployment (Colab vs Streamlit Cloud)
if os.path.exists('enterprise_database.db'):
    db_path = 'enterprise_database.db' # Path for GitHub / Streamlit Cloud production
else:
    db_path = '/content/drive/MyDrive/Smart_inventory_project/Data/Processed/enterprise_database.db' # Path for Google Colab development

@st.cache_data
def load_homepage_metrics():
    try:
        conn = sqlite3.connect(db_path)
        # Fast SQL queries to keep homepage rendering instantaneous
        total_rows = pd.read_sql_query("SELECT COUNT(*) as cnt FROM monthly_inventory", conn).iloc[0]['cnt']
        unique_skus = pd.read_sql_query("SELECT COUNT(DISTINCT Code) as cnt FROM monthly_inventory", conn).iloc[0]['cnt']
        
        df_dead = pd.read_sql_query("SELECT Name, Out_Qty, End_Sum FROM monthly_inventory", conn)
        conn.close()
        
        # Numeric string parsing and cleanup for standard 1C anomalies
        df_dead['Out_Qty'] = pd.to_numeric(df_dead['Out_Qty'].astype(str).str.replace(' ', '').str.replace(',', '.'), errors='coerce').fillna(0)
        df_dead['End_Sum'] = pd.to_numeric(df_dead['End_Sum'].astype(str).str.replace(' ', '').str.replace(',', '.'), errors='coerce').fillna(0)
        
        item_perf = df_dead.groupby('Name').agg({'Out_Qty': 'sum', 'End_Sum': 'last'})
        frozen_capital = item_perf[(item_perf['Out_Qty'] == 0) & (item_perf['End_Sum'] > 0)]['End_Sum'].sum()
        
        return total_rows, unique_skus, frozen_capital
    except:
        return 0, 0, 0

total_rows, unique_skus, frozen_capital = load_homepage_metrics()

# Display core business KPIs in styled layout containers
st.markdown("---")
st.markdown("#### 📊 Real-Time Enterprise Operational Metrics:")
col1, col2, col3 = st.columns(3)

col1.metric(label="🌟 Active SKU Count in System", value=f"{unique_skus:,} Items")
col2.error(label="🛑 Frozen Capital (Dead Stock)", value=f"{frozen_capital / 1e9:,.2f} Billion UZS")
col3.info(label="📈 Parsed 1C Transaction Records", value=f"{total_rows:,} Entries")

st.markdown("---")
st.info("💡 **Navigation Guide:** Use the sidebar on the left 👈 to navigate through different analytical modules.")

st.markdown("""
### Dissertation Research Modules Integrated into the Platform:
1. **📊 Module 1. Financial Audit:** Identification of non-moving assets and warehouse capital optimization.
2. **🤖 Module 2. AI Forecast:** Intermittent demand forecasting utilizing advanced ML ensembles (Prophet + XGBoost).
3. **💎 Module 3. Advanced Analytics:** Explainable AI diagnostics via SHAP feature importance and financial ROI simulation.
4. **🏢 Module 4. Supplier Analytics:** Supply chain risk assessment based on end-to-end ABC analysis of inbound invoices.
5. **✉️ Module 5. Procurement Alerts:** Automated Generation of Master Procurement Schedules with real-time Excel email routing.
""")
