import pandas as pd

df = pd.read_csv("student_marks.csv")

# Average marks per student
df["Total"] = df.iloc[:, 2:].sum(axis=1)
df["Average"] = df.iloc[:, 2:].mean(axis=1)

# Top 3 performers
top3 = df.sort_values(by="Average", ascending=False).head(3)

# Weak subjects (lowest average)
weak_subjects = df.iloc[:, 2:7].mean().sort_values().head(2)

print("🏆 Top 3 Performers:")
print(top3[["Name", "Average"]])

print("\n📉 Weak Subjects:")
print(weak_subjects)

print("\n📘 Subject Averages:")
print(df.iloc[:, 2:7].mean())
