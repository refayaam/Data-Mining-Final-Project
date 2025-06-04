import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
# from sklearn.preprocessing import LabelEncoder # Not used directly here, can be removed if no other purpose

# --- 1. Streamlit Page Configuration ---
st.set_page_config(
    page_title="Traffic Accident Data Mining Dashboard",
    layout="wide", # Use wide layout
    initial_sidebar_state="expanded" # Sidebar expanded by default
)

# --- 2. Data Loading Function (with Caching and Error Handling) ---
@st.cache_data # Use caching to avoid reloading data on every user interaction
def load_data():
    """
    Loads three datasets from CSV files.
    Includes error handling for FileNotFoundError.
    """
    try:
        df_raw = pd.read_csv("prepared_traffic_accidents.csv")
        df_unscaled = pd.read_csv("prepared_data_unscaled.csv")
        df_scaled = pd.read_csv("prepared_data_scaled.csv")
        
        # Example: If DayOfWeek is numerical (0=Sunday, ..., 6=Saturday)
        # You can add a day name column for better visualization
        day_map = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
        if 'DayOfWeek' in df_raw.columns:
            df_raw['DayName'] = df_raw['DayOfWeek'].map(day_map)

        return df_raw, df_unscaled, df_scaled
    except FileNotFoundError as e:
        st.error(f"Error: One or more data files not found. Please ensure the CSV files are in the same directory as the script. ({e})")
        st.stop() # Stop the Streamlit application
    except pd.errors.EmptyDataError:
        st.error("Error: One or more CSV files might be empty. Please check their content.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}. Please check your CSV file format.")
        st.stop()

# Load the data
df_raw, df_unscaled, df_scaled = load_data()

# --- 3. Main Application Header ---
st.title("📊 Traffic Accident Data Mining Dashboard")
st.markdown("This dashboard provides exploration and visualization of traffic accident data.")

# --- 4. Sidebar Navigation ---
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Select Menu:",
    ["Home", "Data", "Visualization", "About App"]
)

# --- 5. Page Content Based on Navigation Choice ---

if option == "Home":
    st.header("Welcome to the Data Mining Dashboard")
    st.write("Use the menu on the side to explore datasets and view various visualizations related to traffic accidents.")
    # FIX 1: use_column_width replaced with use_container_width
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Traffic_accident_on_a_highway_in_Italy.jpg/1200px-Traffic_accident_on_a_highway_in_Italy.jpg", use_container_width=True, caption="Traffic Accident Illustration")
    st.markdown("""
        <p style="font-size: 1.1em;">
        This dashboard is developed to assist in analyzing accident patterns, influencing factors,
        and potential data segmentation using data mining techniques.
        </p>
    """, unsafe_allow_html=True)

elif option == "Data":
    st.header("📁 Available Datasets")
    st.write("You can preview the three datasets used in this analysis.")

    data_option = st.radio(
        "Select dataset:",
        ["Raw Data (`prepared_traffic_accidents.csv`)",
         "Unscaled Data (`prepared_data_unscaled.csv`)",
         "Scaled Data (`prepared_data_scaled.csv`)"]
    )

    if data_option == "Raw Data (`prepared_traffic_accidents.csv`)":
        st.subheader("Raw Data (df_raw)")
        st.dataframe(df_raw)
        st.write(f"Number of rows: {df_raw.shape[0]}, Number of columns: {df_raw.shape[1]}")
        st.write("Available columns:", df_raw.columns.tolist())
    elif data_option == "Unscaled Data (`prepared_data_unscaled.csv`)":
        st.subheader("Unscaled Data (df_unscaled)")
        st.dataframe(df_unscaled)
        st.write(f"Number of rows: {df_unscaled.shape[0]}, Number of columns: {df_unscaled.shape[1]}")
        st.write("Available columns:", df_unscaled.columns.tolist())
    else:
        st.subheader("Scaled Data (df_scaled)")
        st.dataframe(df_scaled)
        st.write(f"Number of rows: {df_scaled.shape[0]}, Number of columns: {df_scaled.shape[1]}")
        st.write("Available columns:", df_scaled.columns.tolist())

elif option == "Visualization":
    st.header("📈 Data Visualization of Traffic Accidents")
    st.write("Various visualizations to understand patterns and relationships in accident data.")

    # Note: Ensure column names used below match your actual data.
    # Provided column list:
    # Month,Day,DayOfWeek,Location_Encoded,Road_Condition_Encoded,Weather_Condition_Encoded,Vehicles_Involved,Severity_Encoded

    # 1. Distribution of Weather Conditions (df_raw)
    st.subheader("1. Accident Distribution by Weather Condition")
    try:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        # FIX 2: Changed 'Weather_Condition' to 'Weather_Condition_Encoded'
        sns.countplot(data=df_raw, x='Weather_Condition_Encoded',
                      order=df_raw['Weather_Condition_Encoded'].value_counts().index, ax=ax1, palette='viridis')
        ax1.set_title("Accident Distribution by Weather Condition", fontsize=16)
        ax1.set_xlabel("Weather Condition (Encoded)", fontsize=12)
        ax1.set_ylabel("Number of Accidents", fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        # FIX 3 & 4: Set alignment separately
        plt.setp(ax1.get_xticklabels(), ha='right')
        plt.tight_layout()
        st.pyplot(fig1)
    except KeyError:
        st.warning("Column 'Weather_Condition_Encoded' not found in raw data. Please check column names.")
    except Exception as e:
        st.error(f"An error occurred while creating the weather plot: {e}")

    # 2. Distribution of Vehicles Involved (df_raw)
    st.subheader("2. Distribution of Vehicles Involved")
    try:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        # FIX: Changed 'Number_of_Vehicles' to 'Vehicles_Involved'
        sns.countplot(data=df_raw, x='Vehicles_Involved', ax=ax2, palette='magma')
        ax2.set_title("Distribution of Vehicles Involved", fontsize=16)
        ax2.set_xlabel("Number of Vehicles", fontsize=12)
        ax2.set_ylabel("Number of Accidents", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig2)
    except KeyError:
        st.warning("Column 'Vehicles_Involved' not found in raw data. Please check column names.")
    except Exception as e:
        st.error(f"An error occurred while creating the vehicle plot: {e}")

    # 3. Temporal Pattern: Accidents per Day of Week (df_raw)
    st.subheader("3. Accident Patterns per Day of Week")
    try:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        # If you have 'DayName' column from processing in load_data(), use it
        if 'DayName' in df_raw.columns:
            sns.countplot(data=df_raw, x='DayName', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ax=ax3, palette='cividis')
            ax3.set_xlabel("Day of Week", fontsize=12)
        else:
            # Use DayOfWeek directly if DayName is not available
            sns.countplot(data=df_raw, x='DayOfWeek', ax=ax3, palette='cividis')
            ax3.set_xlabel("Day of Week (0=Sunday, 6=Saturday)", fontsize=12)
        
        ax3.set_title("Accident Distribution by Day of Week", fontsize=16)
        ax3.set_ylabel("Number of Accidents", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig3)
    except KeyError:
        st.warning("Column 'DayOfWeek' or 'DayName' not found in raw data. Please check column names.")
    except Exception as e:
        st.error(f"An error occurred while creating the day of week plot: {e}")

    # 4. Severity by Location (df_raw)
    st.subheader("4. Severity by Location")
    try:
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        # FIX: Changed 'Location' and 'Severity' to 'Location_Encoded' and 'Severity_Encoded'
        sns.boxplot(data=df_raw, x='Location_Encoded', y='Severity_Encoded', ax=ax4, palette='plasma')
        ax4.set_title("Severity per Location", fontsize=16)
        ax4.set_xlabel("Location (Encoded)", fontsize=12)
        ax4.set_ylabel("Severity (Encoded)", fontsize=12)
        ax4.tick_params(axis='x', rotation=60)
        # FIX 3 & 4: Set alignment separately
        plt.setp(ax4.get_xticklabels(), ha='right')
        plt.tight_layout()
        st.pyplot(fig4)
    except KeyError:
        st.warning("Columns 'Location_Encoded' or 'Severity_Encoded' not found in raw data. Please check column names.")
    except Exception as e:
        st.error(f"An error occurred while creating the location plot: {e}")

    # 5. Cluster Optimization (Elbow Method) (df_scaled)
    st.subheader("5. Cluster Optimization (Elbow Method for K-Means)")
    st.write("The Elbow Method helps determine the optimal number of clusters for the K-Means algorithm.")
    try:
        if not df_scaled.empty and df_scaled.shape[0] >= 2:
            wcss = []
            # Limit max_clusters iteration to not exceed number of data points - 1 or 10
            max_clusters_allowed = min(11, df_scaled.shape[0])
            for i in range(1, max_clusters_allowed):
                kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
                kmeans.fit(df_scaled)
                wcss.append(kmeans.inertia_)
            fig5, ax5 = plt.subplots(figsize=(8, 5))
            ax5.plot(range(1, max_clusters_allowed), wcss, marker='o', linestyle='--', color='blue')
            ax5.set_title("Elbow Method for K-Means", fontsize=16)
            ax5.set_xlabel("Number of Clusters (K)", fontsize=12)
            ax5.set_ylabel("WCSS (Within-Cluster Sum of Squares)", fontsize=12)
            ax5.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig5)
        else:
            st.warning("Scaled data is empty or insufficient to run the Elbow method.")
    except Exception as e:
        st.error(f"An error occurred while creating the Elbow Method plot: {e}")


    # 6. Feature Correlation Heatmap (df_unscaled)
    st.subheader("6. Feature Correlation Heatmap")
    st.write("Visualizes the linear relationships between numerical variables in the data.")
    try:
        if not df_unscaled.empty:
            fig6, ax6 = plt.subplots(figsize=(12, 9))
            sns.heatmap(df_unscaled.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax6, cbar_kws={'label': 'Correlation Coefficient'})
            ax6.set_title("Feature Correlation Heatmap", fontsize=16)
            plt.tight_layout()
            st.pyplot(fig6)
        else:
            st.warning("Unscaled data is empty, cannot create correlation heatmap.")
    except Exception as e:
        st.error(f"An error occurred while creating the correlation heatmap: {e}")

    # 7. Average Severity per Weather Condition (df_raw)
    st.subheader("7. Impact of Weather Condition on Average Severity")
    try:
        # FIX: Changed 'Weather_Condition' and 'Severity' to 'Weather_Condition_Encoded' and 'Severity_Encoded'
        cond_df = df_raw.groupby('Weather_Condition_Encoded')['Severity_Encoded'].mean().reset_index().sort_values(by='Severity_Encoded', ascending=False)
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=cond_df, x='Weather_Condition_Encoded', y='Severity_Encoded', ax=ax7, palette='rocket')
        ax7.set_title("Average Severity by Weather Condition", fontsize=16)
        ax7.set_xlabel("Weather Condition (Encoded)", fontsize=12)
        ax7.set_ylabel("Average Severity (Encoded)", fontsize=12)
        ax7.tick_params(axis='x', rotation=45)
        # FIX 3 & 4: Set alignment separately
        plt.setp(ax7.get_xticklabels(), ha='right')
        plt.tight_layout()
        st.pyplot(fig7)
    except KeyError:
        st.warning("Columns 'Weather_Condition_Encoded' or 'Severity_Encoded' not found in raw data. Please check column names.")
    except Exception as e:
        st.error(f"An error occurred while creating the severity per weather plot: {e}")

elif option == "About App":
    st.header("ℹ️ About This Application")
    st.write("""
    This Data Mining Dashboard application is built using Streamlit to visualize and analyze
    traffic accident data. Its goals are to:
    * **Data Exploration**: Provide an overview of the data structure and characteristics.
    * **Pattern Identification**: Uncover trends and patterns in accidents (e.g., weather conditions, time, location).
    * **Data Preparation**: Display data in its raw, unscaled, and scaled forms.
    * **Clustering Analysis**: Utilize the Elbow method to help determine the optimal number of clusters in the data.

    **Technologies Used:**
    * [Streamlit](https://streamlit.io/) for creating interactive web applications.
    * [Pandas](https://pandas.pydata.org/) for data manipulation and analysis.
    * [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for appealing data visualizations.
    * [Scikit-learn](https://scikit-learn.org/stable/) for K-Means algorithms.
    """)
    # Removed the developer info and contact message as requested.

# --- Footer (Optional) ---
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
        Powered by Streamlit | Traffic Accident Data Mining Dashboard
    </div>
    """,
    unsafe_allow_html=True
)