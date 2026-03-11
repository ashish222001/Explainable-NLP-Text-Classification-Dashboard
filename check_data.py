import pandas as pd

# Load the dataset
df = pd.read_csv("data/raw/Indicators_of_Anxiety_or_Depression.csv")

print("✅ Dataset loaded successfully")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nSample rows:\n", df.head())
