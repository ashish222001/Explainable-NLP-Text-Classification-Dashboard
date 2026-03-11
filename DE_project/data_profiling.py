import pandas as pd

# Load CSV
df = pd.read_csv("student_marks.csv")

# Basic info
print("📘 Data Info:")
print(df.info())

# Check missing values
print("\n🧩 Missing Values:")
print(df.isnull().sum())

# Summary statistics
print("\n📊 Descriptive Statistics:")
print(df.describe())

# Display first few rows
print("\n🔹 Sample Data:")
print(df.head())
