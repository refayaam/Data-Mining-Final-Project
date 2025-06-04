import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('traffic_accidents_indonesia.csv')

print("\n=== Basic Data Exploration ===")

print("\n1. First few rows of the dataset:")
print(df.head())

print("\n2. Dataset information:")
print(df.info())

print("\n3. Summary statistics:")
print(df.describe())

print("\n4. Correlation Analysis:")
# Create dummy variables for categorical columns
df_encoded = pd.get_dummies(df, columns=['Location', 'Road_Condition', 'Weather_Condition', 'Severity'])

# Drop Date and Time columns as they're not relevant for correlation
df_encoded = df_encoded.drop(['Date', 'Time'], axis=1)

# Create correlation matrix
correlation_matrix = df_encoded.corr()

# Create heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

print("\nCorrelation heatmap has been saved as 'correlation_heatmap.png'") 