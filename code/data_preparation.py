import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set the style for all visualizations
plt.style.use('seaborn-v0_8-darkgrid')  # Using a built-in style
sns.set_palette("husl")

print("=== Data Preparation and Exploratory Data Analysis ===")

# Load the dataset
print("\n1. Loading the dataset...")
df = pd.read_csv('traffic_accidents_indonesia.csv')

# Create a directory for visualizations
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Initial Data Exploration
print("\n2. Initial Data Exploration")
print("\nDataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData Types:")
print(df.dtypes)

# Distribution Analysis
print("\n3. Distribution Analysis")

# Create a figure for numerical distributions
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='Vehicles_Involved', kde=True, bins=20)
plt.title('Distribution of Vehicles Involved')
plt.xlabel('Number of Vehicles')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.boxplot(data=df, y='Vehicles_Involved')
plt.title('Box Plot of Vehicles Involved')
plt.ylabel('Number of Vehicles')

plt.tight_layout()
plt.savefig('visualizations/vehicles_analysis.png')
plt.close()

# Categorical Variables Analysis
fig = plt.figure(figsize=(20, 15))
categorical_cols = ['Location', 'Road_Condition', 'Weather_Condition', 'Severity']
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(2, 2, i)
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.savefig('visualizations/categorical_analysis.png')
plt.close()

# Severity by Location
plt.figure(figsize=(12, 6))
severity_location = pd.crosstab(df['Location'], df['Severity'])
severity_location.plot(kind='bar', stacked=True)
plt.title('Accident Severity by Location')
plt.xlabel('Location')
plt.ylabel('Number of Accidents')
plt.legend(title='Severity')
plt.tight_layout()
plt.savefig('visualizations/severity_by_location.png')
plt.close()

# Weather and Road Condition Analysis
fig = plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.countplot(data=df, x='Weather_Condition', hue='Severity')
plt.title('Severity by Weather Condition')
plt.xticks(rotation=45)
plt.xlabel('Weather Condition')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.countplot(data=df, x='Road_Condition', hue='Severity')
plt.title('Severity by Road Condition')
plt.xticks(rotation=45)
plt.xlabel('Road Condition')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('visualizations/conditions_analysis.png')
plt.close()

# Missing Values Analysis
print("\n4. Missing Values Analysis")
missing_values = df.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)
if missing_values.sum() > 0:
    plt.figure(figsize=(10, 6))
    missing_values.plot(kind='bar')
    plt.title('Missing Values by Column')
    plt.tight_layout()
    plt.savefig('visualizations/missing_values.png')
    plt.close()
    print("\nHandling missing values...")
else:
    print("No missing values found.")

# Outlier Detection using IQR for Vehicles_Involved
print("\n6. Outlier Detection and Analysis")
Q1 = df['Vehicles_Involved'].quantile(0.25)
Q3 = df['Vehicles_Involved'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['Vehicles_Involved'] < lower_bound) | 
              (df['Vehicles_Involved'] > upper_bound)]['Vehicles_Involved']
print("\nOutliers in Vehicles_Involved:")
print(f"Number of outliers: {len(outliers)}")
print(f"Percentage of outliers: {(len(outliers)/len(df))*100:.2f}%")
print(f"IQR bounds: {lower_bound:.2f} to {upper_bound:.2f}")

# Duplicate Records Analysis
print("\n5. Duplicate Records Analysis")
duplicates = df.duplicated()
n_duplicates = duplicates.sum()
print(f"\nNumber of duplicate rows: {n_duplicates}")
if n_duplicates > 0:
    print("Removing duplicate records...")
    df = df.drop_duplicates()
    print(f"Dataset shape after removing duplicates: {df.shape}")

# Data Transformation
print("\n4. Data Transformation")
# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Handle Time conversion
def extract_hour(time_str):
    try:
        for fmt in ['%H:%M', '%H:%M:%S', '%I:%M %p']:
            try:
                return pd.to_datetime(time_str, format=fmt).hour
            except ValueError:
                continue
        return pd.to_datetime(time_str).hour
    except:
        print(f"Warning: Could not parse time: {time_str}")
        return None

df['Hour'] = df['Time'].apply(extract_hour)

# Temporal Analysis
fig = plt.figure(figsize=(20, 10))

# Hour of Day Analysis
plt.subplot(2, 2, 1)
sns.histplot(data=df, x='Hour', bins=24, kde=True)
plt.title('Accidents by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Number of Accidents')

# Day of Week Analysis
plt.subplot(2, 2, 2)
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.countplot(data=df, x='DayOfWeek', order=range(7))
plt.xticks(range(7), day_names, rotation=45)
plt.title('Accidents by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Accidents')

# Month Analysis
plt.subplot(2, 2, 3)
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
sns.countplot(data=df, x='Month', order=range(1, 13))
plt.xticks(range(12), month_names, rotation=45)
plt.title('Accidents by Month')
plt.xlabel('Month')
plt.ylabel('Number of Accidents')

# Vehicles Involved by Severity
plt.subplot(2, 2, 4)
sns.boxplot(data=df, x='Severity', y='Vehicles_Involved')
plt.title('Vehicles Involved by Severity')
plt.xlabel('Severity')
plt.ylabel('Number of Vehicles')

plt.tight_layout()
plt.savefig('visualizations/temporal_analysis.png')
plt.close()

# Feature Encoding
print("\n5. Feature Encoding")

# Label Encoding for Severity
le = LabelEncoder()
df['Severity_Encoded'] = le.fit_transform(df['Severity'])
severity_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

# One-Hot Encoding for nominal features
nominal_features = ['Location', 'Road_Condition', 'Weather_Condition']
onehot = OneHotEncoder(sparse_output=False)
onehot_encoded = onehot.fit_transform(df[nominal_features])

# Get feature names after one-hot encoding
onehot_columns = []
for i, feature in enumerate(nominal_features):
    categories = onehot.categories_[i]
    onehot_columns.extend([f"{feature}_{cat}" for cat in categories])

# Create DataFrame with one-hot encoded features
df_onehot = pd.DataFrame(onehot_encoded, columns=onehot_columns)

# Combine all features
df_encoded = pd.concat([
    df_onehot,
    df[['Month', 'Day', 'DayOfWeek', 'Hour', 'Vehicles_Involved']],
    df[['Severity_Encoded']]
], axis=1)

# Correlation Analysis
plt.figure(figsize=(15, 12))
correlation_matrix = df_encoded.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            cmap='coolwarm', 
            fmt='.2f',
            square=True,
            linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('visualizations/correlation_matrix.png')
plt.close()

# Feature Selection
print("\n6. Feature Selection")
correlations_with_severity = abs(correlation_matrix['Severity_Encoded']).sort_values(ascending=False)
print("\nFeature correlations with Severity (absolute values):")
print(correlations_with_severity)

# Select features based on correlation threshold
correlation_threshold = 0.1
selected_features = correlations_with_severity[correlations_with_severity > correlation_threshold].index.tolist()
selected_features.remove('Severity_Encoded')

# Visualize feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x=correlations_with_severity[1:], y=correlations_with_severity.index[1:])
plt.title('Feature Importance (Correlation with Severity)')
plt.xlabel('Absolute Correlation')
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png')
plt.close()

# Prepare final dataset
df_final = df_encoded[selected_features + ['Severity_Encoded']]

# Scale the features
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_final),
    columns=df_final.columns
)

# Save prepared datasets
df_final.to_csv('prepared_data_unscaled.csv', index=False)
df_scaled.to_csv('prepared_data_scaled.csv', index=False)

print("\n=== Data Preparation Complete ===")
print("\nFiles generated:")
print("1. prepared_data_unscaled.csv - Original encoded data")
print("2. prepared_data_scaled.csv - Scaled data for modeling")
print("\nVisualizations generated in 'visualizations' directory:")
print("1. vehicles_analysis.png - Distribution and box plot of vehicles involved")
print("2. categorical_analysis.png - Distribution of categorical variables")
print("3. severity_by_location.png - Stacked bar plot of severity by location")
print("4. conditions_analysis.png - Weather and road condition analysis")
print("5. temporal_analysis.png - Time-based patterns analysis")
print("6. correlation_matrix.png - Feature correlation heatmap")
print("7. feature_importance.png - Feature importance based on correlation")

print("\nFinal dataset shape:", df_final.shape)
print("\nSummary statistics of prepared data:")
print(df_final.describe()) 