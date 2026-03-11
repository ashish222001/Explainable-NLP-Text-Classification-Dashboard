import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("student_marks.csv")

# Average per subject
subject_avg = df.iloc[:, 2:].mean()

plt.figure(figsize=(8,5))
sns.barplot(x=subject_avg.index, y=subject_avg.values)
plt.title("📊 Subject-Wise Average Marks")
plt.ylabel("Average Marks")
plt.show()

# Individual student performance
plt.figure(figsize=(10,6))
for i in range(len(df)):
    plt.plot(df.columns[2:], df.iloc[i, 2:], marker='o', label=df['Name'][i])
plt.title("🎯 Student-Wise Performance Across Subjects")
plt.xlabel("Subjects")
plt.ylabel("Marks")
plt.legend()
plt.show()
