import pandas as pd

# Load the dataset
df = pd.read_csv('traffic_accidents_indonesia.csv')

print("\nFirst few rows of the dataset:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nSummary statistics:")
print(df.describe()) 