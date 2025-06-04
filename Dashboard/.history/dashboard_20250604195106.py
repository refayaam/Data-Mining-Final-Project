import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Dashboard Datmin", layout="wide")

# Judul
st.title("📊 Dashboard Data Mining")

# Sidebar
st.sidebar.title("Navigasi")
option = st.sidebar.selectbox("Pilih menu:", ["Beranda", "Data", "Visualisasi"])

# Beranda
if option == "Beranda":
    st.header("Selamat Datang di Dashboard Data Mining")
    st.write("Gunakan menu di samping untuk eksplorasi data dan visualisasi.")

# Data
elif option == "Data":
    st.header("📁 Dataset Sample")
    df = pd.DataFrame({
        'Nama': ['Andi', 'Budi', 'Citra', 'Dewi'],
        'Umur': [23, 21, 25, 22],
        'Skor': [89, 75, 91, 88]
    })
    st.dataframe(df)

# Visualisasi
elif option == "Visualisasi":
    st.header("📈 Visualisasi Data")
    data = pd.DataFrame({
        'Kategori': ['A', 'B', 'C', 'D'],
        'Jumlah': [10, 15, 7, 20]
    })
    st.bar_chart(data.set_index('Kategori'))
