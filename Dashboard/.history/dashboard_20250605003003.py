import streamlit as st
import pandas as pd
import numpy as np # Import ini akan tetap ada jika Anda berniat menggunakannya nanti, jika tidak bisa dihapus
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans # Pindahkan import ini ke bagian atas untuk konsistensi

st.set_page_config(page_title="Dashboard Datmin", layout="wide")

# Judul
st.title("📊 Dashboard Data Mining")

# Sidebar
st.sidebar.title("Navigasi")
option = st.sidebar.selectbox("Pilih menu:", ["Beranda", "Data", "Visualisasi"])

# Caching data agar tidak dibaca ulang terus-menerus
@st.cache_data
def load_data():
    try:
        df_raw = pd.read_csv("prepared_traffic_accidents.csv")
        df_unscaled = pd.read_csv("prepared_data_unscaled.csv")
        df_scaled = pd.read_csv("prepared_data_scaled.csv")
        return df_raw, df_unscaled, df_scaled
    except FileNotFoundError:
        st.error("Pastikan file CSV (prepared_traffic_accidents.csv, prepared_data_unscaled.csv, prepared_data_scaled.csv) berada di direktori yang sama dengan aplikasi Streamlit ini.")
        st.stop() # Menghentikan eksekusi aplikasi jika file tidak ditemukan
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}")
        st.stop()

df_raw, df_unscaled, df_scaled = load_data()

# Beranda
if option == "Beranda":
    st.header("Selamat Datang di Dashboard Data Mining")
    st.write("Gunakan menu di samping untuk eksplorasi data dan visualisasi.")

# Data
elif option == "Data":
    st.header("📁 Tiga Dataset yang Tersedia")
    data_option = st.radio("Pilih dataset:", ["Data Mentah", "Data Unscaled", "Data Scaled"])

    if data_option == "Data Mentah":
        st.write("### Data: prepared_traffic_accidents.csv")
        st.dataframe(df_raw)
    elif data_option == "Data Unscaled":
        st.write("### Data: prepared_data_unscaled.csv")
        st.dataframe(df_unscaled)
    else:
        st.write("### Data: prepared_data_scaled.csv")
        st.dataframe(df_scaled)

# Visualisasi
elif option == "Visualisasi":
    st.header("📈 Visualisasi Data Kecelakaan")

    # 1. Distribusi Kategorikal (df_raw)
    st.subheader("1. Distribusi Variabel Kategorikal")
    fig1, ax1 = plt.subplots(figsize=(10, 5)) # Ukuran figure sedikit lebih besar
    sns.countplot(data=df_raw, x='Weather_Condition', order=df_raw['Weather_Condition'].value_counts().index, ax=ax1)
    ax1.set_title("Distribusi Cuaca Saat Kecelakaan", fontsize=14)
    ax1.set_xlabel("Kondisi Cuaca", fontsize=12)
    ax1.set_ylabel("Jumlah Kecelakaan", fontsize=12)
    ax1.tick_params(axis='x', rotation=60, ha='right') # Rotasi dan alignment yang lebih baik
    plt.tight_layout() # Menyesuaikan layout agar label tidak terpotong
    st.pyplot(fig1)

    # 2. Jumlah Kendaraan Terlibat (df_raw)
    st.subheader("2. Jumlah Kendaraan Terlibat")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df_raw, x='Number_of_Vehicles', ax=ax2)
    ax2.set_title("Distribusi Jumlah Kendaraan yang Terlibat", fontsize=14)
    ax2.set_xlabel("Jumlah Kendaraan", fontsize=12)
    ax2.set_ylabel("Jumlah Kecelakaan", fontsize=12)
    st.pyplot(fig2)

    # 3. Pola Temporal (df_raw)
    st.subheader("3. Pola Kecelakaan per Jam")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df_raw, x='Hour', bins=24, kde=True, ax=ax3)
    ax3.set_title("Distribusi Kecelakaan Berdasarkan Jam", fontsize=14)
    ax3.set_xlabel("Jam (0-23)", fontsize=12)
    ax3.set_ylabel("Frekuensi", fontsize=12)
    ax3.set_xticks(range(0, 24, 2)) # Menampilkan label jam setiap 2 jam
    st.pyplot(fig3)

    # 4. Keparahan Berdasarkan Lokasi (df_raw)
    st.subheader("4. Keparahan Berdasarkan Lokasi")
    fig4, ax4 = plt.subplots(figsize=(12, 6)) # Ukuran figure lebih besar
    sns.boxplot(data=df_raw, x='Location', y='Severity', ax=ax4)
    ax4.set_title("Tingkat Keparahan (Severity) per Lokasi", fontsize=14)
    ax4.set_xlabel("Lokasi", fontsize=12)
    ax4.set_ylabel("Severity", fontsize=12)
    ax4.tick_params(axis='x', rotation=60, ha='right') # Rotasi dan alignment yang lebih baik
    plt.tight_layout()
    st.pyplot(fig4)

    # 5. Optimasi Cluster (df_scaled)
    st.subheader("5. Optimasi Cluster (Elbow Method)")
    # Memastikan df_scaled tidak kosong dan memiliki dimensi yang cukup
    if not df_scaled.empty and df_scaled.shape[0] >= 2:
        wcss = []
        max_clusters = min(11, df_scaled.shape[0]) # Batasi jumlah cluster agar tidak melebihi jumlah data
        for i in range(1, max_clusters):
            # n_init='auto' atau nilai numerik spesifik seperti n_init=10
            kmeans = KMeans(n_clusters=i, random_state=0, n_init='auto')
            kmeans.fit(df_scaled)
            wcss.append(kmeans.inertia_)
        fig5, ax5 = plt.subplots(figsize=(8, 4))
        ax5.plot(range(1, max_clusters), wcss, marker='o')
        ax5.set_title("Metode Elbow untuk Menentukan Jumlah Cluster Optimal (K-Means)", fontsize=14)
        ax5.set_xlabel("Jumlah Cluster", fontsize=12)
        ax5.set_ylabel("WCSS (Within-Cluster Sum of Squares)", fontsize=12)
        st.pyplot(fig5)
    else:
        st.warning("Data scaled kosong atau tidak cukup untuk metode Elbow.")

    # 6. Korelasi Antar Fitur (df_unscaled)
    st.subheader("6. Korelasi Antar Variabel")
    fig6, ax6 = plt.subplots(figsize=(12, 8)) # Ukuran figure lebih besar
    sns.heatmap(df_unscaled.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax6) # Format anotasi dan garis antar sel
    ax6.set_title("Heatmap Korelasi Antar Variabel", fontsize=14)
    st.pyplot(fig6)

    # 7. Rata-rata Severity per Kondisi Cuaca (df_raw)
    st.subheader("7. Dampak Cuaca terhadap Keparahan")
    cond_df = df_raw.groupby('Weather_Condition')['Severity'].mean().reset_index().sort_values(by='Severity', ascending=False)
    fig7, ax7 = plt.subplots(figsize=(10, 5)) # Ukuran figure sedikit lebih besar
    sns.barplot(data=cond_df, x='Weather_Condition', y='Severity', ax=ax7)
    ax7.set_title("Rata-rata Tingkat Keparahan (Severity) Berdasarkan Kondisi Cuaca", fontsize=14)
    ax7.set_xlabel("Kondisi Cuaca", fontsize=12)
    ax7.set_ylabel("Rata-rata Severity", fontsize=12)
    ax7.tick_params(axis='x', rotation=60, ha='right') # Rotasi dan alignment yang lebih baik
    plt.tight_layout()
    st.pyplot(fig7)