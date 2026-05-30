import os  
import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Supplier Analytics", page_icon="🏢", layout="wide")
st.title("🏢 Supplier & Inbound Logistics Analytics")
st.markdown("Analysis of incoming invoices to evaluate supplier dependency and inbound cash flow.")

@st.cache_data
def load_invoices():
    
    if os.path.exists('enterprise_database.db'):
        db_path = 'enterprise_database.db'
    else:
        db_path = '/content/drive/MyDrive/Smart_inventory_project/Data/Processed/enterprise_database.db'
        
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM incoming_invoices", conn)
    except:
        return pd.DataFrame()
    conn.close()

    if not df.empty:
        df['Sum_With_NDS'] = df['Sum_With_NDS'].astype(str).str.replace(' ', '', regex=False).str.replace(',', '.', regex=False)
        df['Sum_With_NDS'] = pd.to_numeric(df['Sum_With_NDS'], errors='coerce').fillna(0)
        df['Doc_Date'] = pd.to_datetime(df['Doc_Date'], errors='coerce')
        df['Organization'] = df['Organization'].fillna('Unknown Supplier')
    return df

df_invoices = load_invoices()

if df_invoices.empty:
    st.error("⚠️ No invoice data found. Please run Step 8 in Data Consolidation script.")
else:
    total_inbound = df_invoices['Sum_With_NDS'].sum()
    total_invoices = len(df_invoices)
    
    col1, col2 = st.columns(2)
    col1.success(f"📦 Total Inbound Volume: \n### {total_inbound / 1e9:,.2f} Billion UZS")
    col2.info(f"🧾 Total Invoices Processed: \n### {total_invoices}")

    st.markdown("---")
    st.subheader("📊 Top Suppliers by Volume (ABC Analysis)")
    
    supplier_perf = df_invoices.groupby('Organization').agg(
        Total_Value=('Sum_With_NDS', 'sum'),
        Invoice_Count=('Reg_Number', 'count')
    ).reset_index().sort_values(by='Total_Value', ascending=False)
    
    supplier_perf['% of Total Volume'] = (supplier_perf['Total_Value'] / total_inbound) * 100
    
    top_5_suppliers = supplier_perf.head(5)
    other_suppliers_value = supplier_perf.iloc[5:]['Total_Value'].sum()
    
    plot_data = top_5_suppliers[['Organization', 'Total_Value']].copy()
    if other_suppliers_value > 0:
        plot_data.loc[len(plot_data)] = ['Other Suppliers', other_suppliers_value]
        
    fig = px.pie(plot_data, values='Total_Value', names='Organization', hole=0.4, 
                 title="Inbound Cash Flow Distribution by Supplier",
                 color_discrete_sequence=px.colors.sequential.Teal)
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Supplier Registry Directory")
    st.dataframe(supplier_perf.style.format({
        'Total_Value': '{:,.2f} UZS',
        '% of Total Volume': '{:.2f}%'
    }), use_container_width=True)
