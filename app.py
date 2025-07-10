import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="📊 Analisis Credit Approval (UCI)", layout="wide")
st.title("📊 Analisis Dataset Credit Approval (UCI)")
st.write("Aplikasi ini memprediksi kelayakan kredit menggunakan model Random Forest yang sudah dilatih berdasarkan dataset Credit Approval dari UCI.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("credit_model.pkl")

model = load_model()

# Input fitur (15 fitur sesuai dataset)
input_data = {}
feature_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9',
                 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']

st.subheader("🔢 Masukkan Nilai Atribut (Sudah Ter-encode dan Preprocessed)")
for feature in feature_names:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

# Prediksi
if st.button("🔍 Prediksi Kelayakan Kredit"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    hasil = "✅ Disetujui" if prediction == 1 else "❌ Ditolak"
    st.success(f"Hasil Prediksi: {hasil}")
