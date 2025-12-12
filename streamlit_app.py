# =================================================================
# TAHAP 3.2: STREAMLIT DEPLOYMENT APP
# File: streamlit_app.py
# =================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. Pemuatan Model dan Preprocessor ---
# Asumsi: File model dan preprocessor berada di folder 'models'

@st.cache_resource
def load_assets():
    """Memuat model dan preprocessor yang sudah dilatih."""
    try:
        # Pemuatan Model Terbaik (Logistic Regression Tuned)
        model = joblib.load('models/best_churn_model.joblib')
        
        # Pemuatan Preprocessor (ColumnTransformer)
        preprocessor = joblib.load('models/preprocessor.joblib')
        
        return model, preprocessor
    except FileNotFoundError:
        st.error("File model atau preprocessor tidak ditemukan. Pastikan 'models/best_churn_model.joblib' dan 'models/preprocessor.joblib' ada.")
        return None, None

model, preprocessor = load_assets()

# --- 2. Konfigurasi Aplikasi Streamlit ---

st.set_page_config(
    page_title="Prediksi Churn Pelanggan Telco",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📞 Prediksi Churn Pelanggan Telco")
st.markdown("Aplikasi Machine Learning untuk memprediksi apakah seorang pelanggan akan *churn* (berhenti berlangganan) atau tidak, menggunakan model **Logistic Regression** yang telah dioptimalkan.")

if model and preprocessor:
    st.sidebar.header("Fitur Layanan Pelanggan")

    # --- 3. Form Input Fitur ---

    # Demografi
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.sidebar.selectbox('Senior Citizen', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    partner = st.sidebar.selectbox('Partner', ['Yes', 'No'])
    dependents = st.sidebar.selectbox('Dependents', ['Yes', 'No'])
    
    # Lama Berlangganan
    tenure = st.sidebar.slider('Lama Berlangganan (Bulan)', 1, 72, 12)
    
    # Layanan
    phone_service = st.sidebar.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
    internet_service = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.sidebar.selectbox('Online Security', ['No internet service', 'No', 'Yes'])
    online_backup = st.sidebar.selectbox('Online Backup', ['No internet service', 'No', 'Yes'])
    device_protection = st.sidebar.selectbox('Device Protection', ['No internet service', 'No', 'Yes'])
    tech_support = st.sidebar.selectbox('Tech Support', ['No internet service', 'No', 'Yes'])
    streaming_tv = st.sidebar.selectbox('Streaming TV', ['No internet service', 'No', 'Yes'])
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ['No internet service', 'No', 'Yes'])
    
    # Kontrak dan Biaya
    contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.sidebar.number_input('Monthly Charges', 18.00, 118.75, 70.00, step=0.01)
    
    # TotalCharges akan dihitung berdasarkan MonthlyCharges * Tenure jika tidak ada input eksplisit
    total_charges = monthly_charges * tenure 


    # Membuat DataFrame Input
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [senior_citizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })
    
    # Tampilkan input pengguna
    st.subheader("Data Input Pelanggan:")
    st.dataframe(input_data)

    # --- 4. Proses Prediksi ---
    if st.button("Prediksi Churn"):
        with st.spinner('Melakukan Preprocessing dan Prediksi...'):
            try:
                # 1. Terapkan Preprocessor
                # Gunakan .transform() pada ColumnTransformer yang sudah dilatih
                input_processed = preprocessor.transform(input_data)
                
                # 2. Prediksi
                prediction = model.predict(input_processed)
                prediction_proba = model.predict_proba(input_processed)
                
                churn_status = "CHURN" if prediction[0] == 1 else "TIDAK CHURN"
                
                st.subheader(f"Hasil Prediksi:")
                
                if prediction[0] == 1:
                    st.error(f"⚠️ Pelanggan ini **DIPREDIKSI CHURN**.")
                else:
                    st.success(f"✅ Pelanggan ini **DIPREDIKSI TIDAK CHURN**.")
                
                st.write(f"Probabilitas Tidak Churn (No): **{prediction_proba[0][0]:.2f}**")
                st.write(f"Probabilitas Churn (Yes): **{prediction_proba[0][1]:.2f}**")
            
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
                st.warning("Pastikan data input sesuai dengan format data pelatihan.")

# --- 5. Elemen Pendukung ---

st.markdown("---")
st.caption("Proyek UAS Bengkel Koding Data Science | Model: Logistic Regression (Tuned)")