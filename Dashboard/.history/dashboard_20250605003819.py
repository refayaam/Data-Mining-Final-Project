import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
# from sklearn.preprocessing import LabelEncoder # Tidak digunakan langsung di sini, bisa dihapus jika tidak ada keperluan lain

# --- 1. Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Dashboard Data Mining Kecelakaan Lalu Lintas",
    layout="wide", # Menggunakan layout lebar
    initial_sidebar_state="expanded" # Sidebar dibuka secara default
)

# --- 2. Fungsi Pemuatan Data (dengan Caching dan Penanganan Error) ---
@st.cache_data # Menggunakan caching untuk menghindari pemuatan ulang data setiap kali interaksi
def load_data():
    """
    Memuat tiga dataset dari file CSV.
    Menyertakan penanganan kesalahan untuk FileNotFoundError.
    """
    try:
        df_raw = pd.read_csv("prepared_traffic_accidents.csv")
        df_unscaled = pd.read_csv("prepared_data_unscaled.csv")
        df_scaled = pd.read_csv("prepared_data_scaled.csv")
        
        # Contoh: Jika DayOfWeek adalah numerik (0=Minggu, ..., 6=Sabtu)
        # Anda bisa menambahkan kolom nama hari untuk visualisasi yang lebih baik
        day_map = {0: 'Minggu', 1: 'Senin', 2: 'Selasa', 3: 'Rabu', 4: 'Kamis', 5: 'Jumat', 6: 'Sabtu'}
        if 'DayOfWeek' in df_raw.columns:
            df_raw['NamaHari'] = df_raw['DayOfWeek'].map(day_map)

        return df_raw, df_unscaled, df_scaled
    except FileNotFoundError as e:
        st.error(f"Kesalahan: Salah satu file data tidak ditemukan. Pastikan file CSV berada di direktori yang sama dengan skrip. ({e})")
        st.stop() # Menghentikan aplikasi Streamlit
    except pd.errors.EmptyDataError:
        st.error("Kesalahan: Salah satu file CSV mungkin kosong. Mohon periksa isinya.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}. Mohon periksa format file CSV Anda.")
        st.stop()

# Memuat data
df_raw, df_unscaled, df_scaled = load_data()

# --- 3. Header Utama Aplikasi ---
st.title("📊 Dashboard Data Mining Kecelakaan Lalu Lintas")
st.markdown("Dashboard ini menyajikan eksplorasi dan visualisasi dari data kecelakaan lalu lintas.")

# --- 4. Sidebar Navigasi ---
st.sidebar.title("Navigasi")
option = st.sidebar.selectbox(
    "Pilih menu:",
    ["Beranda", "Data", "Visualisasi", "Tentang Aplikasi"]
)

# --- 5. Konten Halaman Berdasarkan Pilihan Navigasi ---

if option == "Beranda":
    st.header("Selamat Datang di Dashboard Data Mining")
    st.write("Gunakan menu di samping untuk eksplorasi dataset dan melihat berbagai visualisasi terkait kecelakaan lalu lintas.")
    # PERBAIKAN 1: use_column_width diganti use_container_width
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Traffic_accident_on_a_highway_in_Italy.jpg/1200px-Traffic_accident_on_a_highway_in_Italy.jpg", use_container_width=True, caption="Ilustrasi Kecelakaan Lalu Lintas")
    st.markdown("""
        <p style="font-size: 1.1em;">
        Dashboard ini dikembangkan untuk membantu analisis pola kecelakaan, faktor-faktor yang mempengaruhinya,
        dan potensi segmentasi data menggunakan teknik data mining.
        </p>
    """, unsafe_allow_html=True)

elif option == "Data":
    st.header("📁 Dataset yang Tersedia")
    st.write("Anda dapat melihat pratinjau dari tiga dataset yang digunakan dalam analisis ini.")

    data_option = st.radio(
        "Pilih dataset:",
        ["Data Mentah (`prepared_traffic_accidents.csv`)",
         "Data Unscaled (`prepared_data_unscaled.csv`)",
         "Data Scaled (`prepared_data_scaled.csv`)"]
    )

    if data_option == "Data Mentah (`prepared_traffic_accidents.csv`)":
        st.subheader("Data Mentah (df_raw)")
        st.dataframe(df_raw)
        st.write(f"Jumlah baris: {df_raw.shape[0]}, Jumlah kolom: {df_raw.shape[1]}")
        st.write("Kolom yang tersedia:", df_raw.columns.tolist())
    elif data_option == "Data Unscaled (`prepared_data_unscaled.csv`)":
        st.subheader("Data Unscaled (df_unscaled)")
        st.dataframe(df_unscaled)
        st.write(f"Jumlah baris: {df_unscaled.shape[0]}, Jumlah kolom: {df_unscaled.shape[1]}")
        st.write("Kolom yang tersedia:", df_unscaled.columns.tolist())
    else:
        st.subheader("Data Scaled (df_scaled)")
        st.dataframe(df_scaled)
        st.write(f"Jumlah baris: {df_scaled.shape[0]}, Jumlah kolom: {df_scaled.shape[1]}")
        st.write("Kolom yang tersedia:", df_scaled.columns.tolist())

elif option == "Visualisasi":
    st.header("📈 Visualisasi Data Kecelakaan")
    st.write("Berbagai visualisasi untuk memahami pola dan hubungan dalam data kecelakaan.")

    # Catatan: Pastikan nama kolom yang digunakan di bawah ini sesuai dengan data Anda.
    # Daftar kolom yang Anda berikan:
    # Month,Day,DayOfWeek,Location_Encoded,Road_Condition_Encoded,Weather_Condition_Encoded,Vehicles_Involved,Severity_Encoded

    # 1. Distribusi Kondisi Cuaca (df_raw)
    st.subheader("1. Distribusi Kecelakaan Berdasarkan Kondisi Cuaca")
    try:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        # PERBAIKAN 2: Ganti 'Weather_Condition' menjadi 'Weather_Condition_Encoded'
        sns.countplot(data=df_raw, x='Weather_Condition_Encoded',
                      order=df_raw['Weather_Condition_Encoded'].value_counts().index, ax=ax1, palette='viridis')
        ax1.set_title("Distribusi Kecelakaan Berdasarkan Kondisi Cuaca", fontsize=16)
        ax1.set_xlabel("Kondisi Cuaca (Encoded)", fontsize=12)
        ax1.set_ylabel("Jumlah Kecelakaan", fontsize=12)
        # PERBAIKAN 3: Hapus 'ha' dari tick_params
        ax1.tick_params(axis='x', rotation=45)
        # PERBAIKAN 4: Atur alignment secara terpisah jika diperlukan
        plt.setp(ax1.get_xticklabels(), ha='right')
        plt.tight_layout()
        st.pyplot(fig1)
    except KeyError:
        st.warning("Kolom 'Weather_Condition_Encoded' tidak ditemukan di data mentah. Mohon periksa nama kolom.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membuat plot cuaca: {e}")

    # 2. Distribusi Jumlah Kendaraan Terlibat (df_raw)
    st.subheader("2. Distribusi Jumlah Kendaraan yang Terlibat")
    try:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        # PERBAIKAN: Ganti 'Number_of_Vehicles' menjadi 'Vehicles_Involved'
        sns.countplot(data=df_raw, x='Vehicles_Involved', ax=ax2, palette='magma')
        ax2.set_title("Distribusi Jumlah Kendaraan yang Terlibat", fontsize=16)
        ax2.set_xlabel("Jumlah Kendaraan", fontsize=12)
        ax2.set_ylabel("Jumlah Kecelakaan", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig2)
    except KeyError:
        st.warning("Kolom 'Vehicles_Involved' tidak ditemukan di data mentah. Mohon periksa nama kolom.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membuat plot kendaraan: {e}")

    # 3. Pola Temporal: Kecelakaan per Hari dalam Seminggu (df_raw)
    st.subheader("3. Pola Kecelakaan per Hari dalam Seminggu")
    try:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        # Jika Anda memiliki kolom 'NamaHari' dari pemrosesan di load_data(), gunakan itu
        if 'NamaHari' in df_raw.columns:
            sns.countplot(data=df_raw, x='NamaHari', order=['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu'], ax=ax3, palette='cividis')
            ax3.set_xlabel("Hari dalam Seminggu", fontsize=12)
        else:
            # Menggunakan DayOfWeek langsung jika NamaHari tidak ada
            sns.countplot(data=df_raw, x='DayOfWeek', ax=ax3, palette='cividis')
            ax3.set_xlabel("Hari dalam Seminggu (0=Minggu, 6=Sabtu)", fontsize=12)
        
        ax3.set_title("Distribusi Kecelakaan Berdasarkan Hari dalam Seminggu", fontsize=16)
        ax3.set_ylabel("Jumlah Kecelakaan", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig3)
    except KeyError:
        st.warning("Kolom 'DayOfWeek' atau 'NamaHari' tidak ditemukan di data mentah. Mohon periksa nama kolom.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membuat plot hari dalam seminggu: {e}")

    # 4. Keparahan Berdasarkan Lokasi (df_raw)
    st.subheader("4. Tingkat Keparahan (Severity) Berdasarkan Lokasi")
    try:
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        # PERBAIKAN: Ganti 'Location' dan 'Severity' menjadi 'Location_Encoded' dan 'Severity_Encoded'
        sns.boxplot(data=df_raw, x='Location_Encoded', y='Severity_Encoded', ax=ax4, palette='plasma')
        ax4.set_title("Tingkat Keparahan (Severity) per Lokasi", fontsize=16)
        ax4.set_xlabel("Lokasi (Encoded)", fontsize=12)
        ax4.set_ylabel("Severity (Encoded)", fontsize=12)
        # PERBAIKAN 3: Hapus 'ha' dari tick_params
        ax4.tick_params(axis='x', rotation=60)
        # PERBAIKAN 4: Atur alignment secara terpisah jika diperlukan
        plt.setp(ax4.get_xticklabels(), ha='right')
        plt.tight_layout()
        st.pyplot(fig4)
    except KeyError:
        st.warning("Kolom 'Location_Encoded' atau 'Severity_Encoded' tidak ditemukan di data mentah. Mohon periksa nama kolom.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membuat plot lokasi: {e}")

    # 5. Optimasi Cluster (Metode Elbow) (df_scaled)
    st.subheader("5. Optimasi Cluster (Metode Elbow untuk K-Means)")
    st.write("Metode Elbow membantu menentukan jumlah cluster optimal untuk algoritma K-Means.")
    try:
        if not df_scaled.empty and df_scaled.shape[0] >= 2:
            wcss = []
            # Batasi iterasi max_clusters agar tidak melebihi jumlah data - 1 atau 10
            max_clusters_allowed = min(11, df_scaled.shape[0])
            for i in range(1, max_clusters_allowed):
                kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
                kmeans.fit(df_scaled)
                wcss.append(kmeans.inertia_)
            fig5, ax5 = plt.subplots(figsize=(8, 5))
            ax5.plot(range(1, max_clusters_allowed), wcss, marker='o', linestyle='--', color='blue')
            ax5.set_title("Metode Elbow untuk K-Means", fontsize=16)
            ax5.set_xlabel("Jumlah Cluster (K)", fontsize=12)
            ax5.set_ylabel("WCSS (Within-Cluster Sum of Squares)", fontsize=12)
            ax5.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig5)
        else:
            st.warning("Data scaled kosong atau tidak cukup untuk menjalankan metode Elbow.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membuat plot Elbow Method: {e}")


    # 6. Heatmap Korelasi Antar Fitur (df_unscaled)
    st.subheader("6. Heatmap Korelasi Antar Variabel")
    st.write("Melihat hubungan linear antar variabel numerik dalam data.")
    try:
        if not df_unscaled.empty:
            fig6, ax6 = plt.subplots(figsize=(12, 9))
            sns.heatmap(df_unscaled.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax6, cbar_kws={'label': 'Koefisien Korelasi'})
            ax6.set_title("Heatmap Korelasi Antar Variabel", fontsize=16)
            plt.tight_layout()
            st.pyplot(fig6)
        else:
            st.warning("Data unscaled kosong, tidak dapat membuat heatmap korelasi.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membuat heatmap korelasi: {e}")

    # 7. Rata-rata Severity per Kondisi Cuaca (df_raw)
    st.subheader("7. Dampak Kondisi Cuaca terhadap Rata-rata Severity")
    try:
        # PERBAIKAN: Ganti 'Weather_Condition' dan 'Severity' menjadi 'Weather_Condition_Encoded' dan 'Severity_Encoded'
        cond_df = df_raw.groupby('Weather_Condition_Encoded')['Severity_Encoded'].mean().reset_index().sort_values(by='Severity_Encoded', ascending=False)
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=cond_df, x='Weather_Condition_Encoded', y='Severity_Encoded', ax=ax7, palette='rocket')
        ax7.set_title("Rata-rata Severity Berdasarkan Kondisi Cuaca", fontsize=16)
        ax7.set_xlabel("Kondisi Cuaca (Encoded)", fontsize=12)
        ax7.set_ylabel("Rata-rata Severity (Encoded)", fontsize=12)
        # PERBAIKAN 3: Hapus 'ha' dari tick_params
        ax7.tick_params(axis='x', rotation=45)
        # PERBAIKAN 4: Atur alignment secara terpisah jika diperlukan
        plt.setp(ax7.get_xticklabels(), ha='right')
        plt.tight_layout()
        st.pyplot(fig7)
    except KeyError:
        st.warning("Kolom 'Weather_Condition_Encoded' atau 'Severity_Encoded' tidak ditemukan di data mentah. Mohon periksa nama kolom.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membuat plot severity per cuaca: {e}")

elif option == "Tentang Aplikasi":
    st.header("ℹ️ Tentang Aplikasi Ini")
    st.write("""
    Aplikasi Dashboard Data Mining ini dibangun menggunakan Streamlit untuk memvisualisasikan dan menganalisis
    data kecelakaan lalu lintas. Tujuannya adalah untuk:
    * **Eksplorasi Data**: Memberikan gambaran umum tentang struktur dan karakteristik data.
    * **Identifikasi Pola**: Mengungkap tren dan pola dalam kecelakaan (misalnya, kondisi cuaca, waktu, lokasi).
    * **Persiapan Data**: Menampilkan data dalam bentuk mentah, belum diskalakan, dan sudah diskalakan.
    * **Analisis Clustering**: Menggunakan metode Elbow untuk membantu menentukan jumlah cluster optimal dalam data.

    **Teknologi yang Digunakan:**
    * [Streamlit](https://streamlit.io/) untuk pembuatan aplikasi web interaktif.
    * [Pandas](https://pandas.pydata.org/) untuk manipulasi dan analisis data.
    * [Matplotlib](https://matplotlib.org/) dan [Seaborn](https://seaborn.pydata.org/) untuk visualisasi data yang menarik.
    * [Scikit-learn](https://scikit-learn.org/stable/) untuk algoritma K-Means.

    **Pengembang:** [Nama Anda / Tim Anda]
    **Tanggal Pembuatan:** Juni 2025
    """)
    st.info("Untuk informasi lebih lanjut, silakan hubungi pengembang.")

# --- Footer (Opsional) ---
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: #808080;
        text-align: center;
        padding: 5px;
        font-size: 0.8em;
    }
    </style>
    <div class="footer">
        Powered by Streamlit | Dashboard Data Mining Kecelakaan Lalu Lintas
    </div>
    """,
    unsafe_allow_html=True
)