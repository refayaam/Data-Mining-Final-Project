import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

print("=== Data Modeling and Evaluation ===")

# Load the prepared data
print("\n1. Loading prepared data...")
df = pd.read_csv('prepared_data_unscaled.csv')

# Separate features and target
X = df.drop('Severity_Encoded', axis=1)
y = df['Severity_Encoded']

print("\nFeatures used:", X.columns.tolist())
print("Target variable: Severity_Encoded")

# Part 1: Logistic Regression (Supervised Learning)
print("\n=== Logistic Regression Modeling ===")

# Data splitting for supervised learning
print("\n2. Data Splitting for Logistic Regression")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,            # 80% training, 20% testing
    random_state=42,          # For reproducibility
    stratify=y                # Maintain class distribution
)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
print("\n3. Training Logistic Regression Model")
lr_model = LogisticRegression(
    max_iter=1000,            # Increase iterations for convergence
    random_state=42,          # For reproducibility
    multi_class='multinomial' # Handle multiple severity classes
)
lr_model.fit(X_train_scaled, y_train)

# Evaluate Logistic Regression
print("\n4. Logistic Regression Evaluation")
y_pred = lr_model.predict(X_test_scaled)

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.mean(np.abs(lr_model.coef_), axis=0)
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print("\nTop 5 Most Important Features for Logistic Regression:")
print(feature_importance.head())

# Part 2: K-means Clustering (Unsupervised Learning)
print("\n=== K-means Clustering ===")

# Load scaled data for clustering
print("\n5. Preparing Data for K-means")
df_scaled = pd.read_csv('prepared_data_scaled.csv')
X_clustering = df_scaled.drop('Severity_Encoded', axis=1)

# Find optimal number of clusters using elbow method
print("\n6. Finding Optimal Number of Clusters")
inertias = []
silhouette_scores = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_clustering)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_clustering, kmeans.labels_))

# Plot elbow curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')

# Plot silhouette scores
plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'rx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')

plt.tight_layout()
plt.savefig('kmeans_optimization.png')
plt.close()

# Train final K-means model
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 because we started from K=2
print(f"\n7. Training K-means with optimal clusters (k={optimal_k})")
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = final_kmeans.fit_predict(X_clustering)

# Analyze clusters
print("\n8. Cluster Analysis")
df_with_clusters = pd.DataFrame(X_clustering, columns=X.columns)
df_with_clusters['Cluster'] = cluster_labels
df_with_clusters['True_Severity'] = df['Severity_Encoded']

# Calculate cluster statistics
print("\nCluster Statistics:")
for cluster in range(optimal_k):
    cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
    print(f"\nCluster {cluster}:")
    print(f"Size: {len(cluster_data)} samples")
    print(f"Average True Severity: {cluster_data['True_Severity'].mean():.2f}")
    print("Most common features:")
    for col in X.columns[:3]:  # Show top 3 features
        print(f"{col}: {cluster_data[col].mean():.2f}")

print("\n=== Modeling Complete ===")
print("Files generated:")
print("1. kmeans_optimization.png - Elbow and Silhouette plots for optimal k") 



