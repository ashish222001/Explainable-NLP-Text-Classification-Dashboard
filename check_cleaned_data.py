import pandas as pd

df = pd.read_csv("data/processed/merged.csv")

print("✅ Cleaned Dataset loaded successfully")
print("Shape:", df.shape)
print("\nLabel Distribution:\n", df['label'].value_counts())
print("\nSample rows:\n", df.sample(5))

