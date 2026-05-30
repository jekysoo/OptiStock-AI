import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import io
import re
import os

st.set_page_config(page_title="Procurement Alerts", page_icon="✉️", layout="wide")
st.title("✉️ Global Procurement Alert Center")
st.markdown("Generate an automated, rolling 3-month demand forecast for **all 2,500+ items** simultaneously and transmit the Master Schedule as an Excel attachment directly to purchasing executives.")

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
code_col = 'Code' if 'Code' in df_clean.columns else 'code'

st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Simple Mail Transfer Protocol (SMTP) Settings")
    recipient_email = st.text_input("📩 To (Purchasing Manager Email):", "purchasing_manager@company.com")
    st.info("💡 **Security Protocol:** To route emails through Google SMTP, pass a 16-character 'App Password' without blank spaces. Standard user passwords will prompt a 535 rejection.")
    sender_email = st.text_input("📧 From (System Authentication Gmail):", placeholder="your.email@gmail.com")
    app_password = st.text_input("🔑 Google App Password:", type="password", placeholder="abcdefghijklmnop")

with col2:
    st.subheader("2. Enterprise Data Batch Scan Status")
    st.markdown(f"Analytical pipeline connected to active registry containing **{len(df_clean['Name'].unique())} verified SKU profiles**. Heuristic engines will map out-of-sample velocity across all product vertices.")
    
    if st.button("🚀 EXECUTE GLOBAL GENERATION & SEND EMAIL REPORT", use_container_width=True):
        if not sender_email or not app_password:
            st.error("❌ Configuration Exception: Credentials missing. Please authenticate via a valid Google gateway profile.")
        else:
            with st.spinner("Processing time-series frames across all nodes and spinning up Excel matrix buffers..."):
                try:
                    # Rolling average velocity approximation for fast global compilation matrix
                    recent_cutoff = df_clean['Date'].max() - pd.DateOffset(months=6)
                    recent_data = df_clean[df_clean['Date'] >= recent_cutoff]
                    
                    if code_col in df_clean.columns:
                        global_demand = recent_data.groupby([code_col, 'Name'])['Out_Qty'].mean().reset_index()
                    else:
                        global_demand = recent_data.groupby('Name')['Out_Qty'].mean().reset_index()
                        
                    global_demand['Forecast_3_Months'] = np.ceil(global_demand['Out_Qty'] * 3)
                    global_demand = global_demand.drop(columns=['Out_Qty'])
                    global_demand = global_demand[global_demand['Forecast_3_Months'] > 0].sort_values('Forecast_3_Months', ascending=False)
                    
                    if code_col in df_clean.columns:
                        global_demand.rename(columns={code_col: 'SKU Code', 'Name': 'Product Description', 'Forecast_3_Months': 'Recommended Order Qty (Next 3 Months)'}, inplace=True)
                    else:
                        global_demand.rename(columns={'Name': 'Product Description', 'Forecast_3_Months': 'Recommended Order Qty (Next 3 Months)'}, inplace=True)

                    # Output compilation into virtual binary memory stream
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        global_demand.to_excel(writer, index=False, sheet_name='Procurement_Plan')
                    excel_buffer.seek(0)
                    
                    # MIME Multi-part envelope construction
                    msg = MIMEMultipart()
                    msg['From'] = sender_email
                    msg['To'] = recipient_email
                    msg['Subject'] = "GLOBAL ALERT: Master Procurement Schedule (Next 3 Months)"
                    
                    html_content = f"""
                    <html><body style="font-family: Arial, sans-serif; color: #333;">
                        <h2 style="color: #2e6c80;">OptiStock AI Autonomous Logistics Node</h2>
                        <p>Attention Logistics Executive,</p>
                        <p>The system has finalized an exhaustive pipeline evaluation of historical material streams.</p>
                        <p>Please find the compiled spreadsheet attached as <b>Global_Restock_Report.xlsx</b>. It specifies strategic requirements for <b>{len(global_demand)} items</b> for the immediate operational quarter.</p>
                        <br>
                        <p><i>Generated autonomously by the OptiStock AI Enterprise Framework.</i></p>
                    </body></html>
                    """
                    msg.attach(MIMEText(html_content, 'html'))
                    
                    # Injecting spreadsheet attachment stream into data payload envelope
                    part = MIMEBase('application', 'vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                    part.set_payload(excel_buffer.read())
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', 'attachment; filename="Global_Restock_Report.xlsx"')
                    msg.attach(part)
                    
                    # Enforce strict spatial parsing to protect password injection from input spaces
                    clean_password = app_password.replace(" ", "")
                    
                    # Establish secure TLS channel with Google SMTP server
                    server = smtplib.SMTP('smtp.gmail.com', 587)
                    server.starttls()
                    server.login(sender_email, clean_password)
                    server.send_message(msg)
                    server.quit()
                    
                    st.success(f"✅ Master Schedule compiled successfully. Report transmitted to: {recipient_email}")
                    st.balloons()
                except smtplib.SMTPAuthenticationError:
                    st.error("❌ SMTP Authentication Failure (535): Access denied by Google. Re-verify your 16-character App Password profile and strip any accidental input spaces.")
                except Exception as e:
                    st.error(f"❌ System Exception: {e}")
