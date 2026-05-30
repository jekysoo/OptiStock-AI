import streamlit as st
import sqlite3
import pandas as pd

st.set_page_config(page_title="Financial Audit", page_icon="📊", layout="wide")
st.title("📊 Financial Audit: Frozen Capital (Dead Stock)")

@st.cache_data
def load_and_clean_data():
    db_path = '/content/drive/MyDrive/Smart_inventory_project/Data/Processed/enterprise_database.db'
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM monthly_inventory", conn)
    conn.close()

    numeric_columns = ['Price', 'Start_Qty', 'Start_Sum', 'In_Qty', 'In_Sum', 'Out_Qty', 'Out_Sum', 'End_Qty', 'End_Sum']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(' ', '', regex=False).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

df_clean = load_and_clean_data()

item_perf = df_clean.groupby('Name').agg(
    Total_Sold_Qty=('Out_Qty', 'sum'),
    Latest_Stock_Qty=('End_Qty', 'last'),
    Latest_Stock_Value=('End_Sum', 'last')
).reset_index()

dead_stock = item_perf[(item_perf['Total_Sold_Qty'] == 0) & (item_perf['Latest_Stock_Value'] > 0)].sort_values(by='Latest_Stock_Value', ascending=False)
total_frozen = dead_stock['Latest_Stock_Value'].sum()

col1, col2 = st.columns(2)
col1.error(f"⚠️ Frozen Capital in Warehouse: \n### {total_frozen / 1e9:,.2f} Billion UZS")
col2.warning(f"🛑 Non-Moving SKU Count: \n### {len(dead_stock)} items")

st.markdown("### Top 15 Worst Performing Assets (Frozen Capital)")
formatted_df = dead_stock.rename(columns={
    'Name': 'Item Name',
    'Latest_Stock_Qty': 'Current Stock Qty',
    'Latest_Stock_Value': 'Value (UZS)'
}).head(15)
st.dataframe(formatted_df, use_container_width=True)
