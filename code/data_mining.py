import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the prepared dataset
print("Loading the prepared dataset...")
df = pd.read_csv('prepared_traffic_accidents.csv')

# Standardize the features for K-means
print("\n=== K-means Clustering Analysis ===")
scaler = StandardScaler()
features_for_clustering = ['Month', 'Day', 'DayOfWeek', 'Location_Encoded', 
                         'Road_Condition_Encoded', 'Weather_Condition_Encoded', 
                         'Vehicles_Involved']
X_scaled = scaler.fit_transform(df[features_for_clustering])

# Determine optimal number of clusters using elbow method
inertias = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# Apply K-means with optimal k (k=3 for this example, but adjust based on elbow curve)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"\nCluster Distribution:")
print(df['Cluster'].value_counts())

# Analyze clusters
print("\nCluster Characteristics:")
for cluster in range(optimal_k):
    print(f"\nCluster {cluster}:")
    cluster_data = df[df['Cluster'] == cluster]
    print(cluster_data.describe().round(2))

# Logistic Regression
print("\n=== Logistic Regression Analysis ===")

# Prepare features and target for classification
X = df[features_for_clustering]  # Features
y = df['Severity_Encoded']      # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = lr_model.predict(X_test_scaled)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print accuracy score
print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred):.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features_for_clustering,
    'Importance': np.abs(lr_model.coef_[0])
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance) 