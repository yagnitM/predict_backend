import pandas as pd

# Load the processed dataset
df = pd.read_csv("data/processed_matches.csv")

# Print all column names
print("📊 Columns in processed_matches.csv:")
print(df.columns.tolist())

# Show first few rows
print("\n🔍 Sample rows:")
print(df.head())
