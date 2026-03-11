import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import os

# ===============================
# 1. Load & Merge Mental Health Datasets
# ===============================
files = [
    "data/raw/student_depression_dataset.csv",
    "data/raw/depression_dataset_reddit_cleaned.csv",
    "data/raw/Mental-Health-Twitter.csv"
]

dfs = []
for f in files:
    if os.path.exists(f):
        try:
            df = pd.read_csv(f, encoding="utf-8", on_bad_lines="skip")
            print(f"✅ Loaded {f}: {df.shape}")
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")

if not dfs:
    raise FileNotFoundError("❌ No dataset found in data/raw folder.")

df = pd.concat(dfs, ignore_index=True)
print("\nMerged dataset shape:", df.shape)

# ===============================
# 2. Clean & Prepare Data
# ===============================
# Try to detect text column
possible_text_cols = [c for c in df.columns if df[c].dtype == 'object']
text_col = None
for c in possible_text_cols:
    avg_len = df[c].astype(str).str.len().mean()
    if avg_len > 15:
        text_col = c
        break

if not text_col:
    raise ValueError("❌ No text column found automatically.")

# Use the text column
df = df[[text_col]].rename(columns={text_col: "text"})

# If no label exists, make pseudo labels for now
if "label" not in df.columns:
    df["label"] = np.random.choice(["anxiety", "depression", "no_stress"], size=len(df))

# Remove blanks and duplicates
df["text"] = df["text"].astype(str)
df = df[df["text"].str.strip() != ""].drop_duplicates("text")
print("✅ Cleaned dataset shape:", df.shape)

# ===============================
# 3. Train/Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# ===============================
# 4. TF-IDF + Logistic Regression
# ===============================
vectorizer = TfidfVectorizer(max_features=6000, stop_words="english", ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=400)
clf.fit(X_train_tfidf, y_train)

# ===============================
# 5. Evaluation
# ===============================
y_pred = clf.predict(X_test_tfidf)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===============================
# 6. Save Model
# ===============================
os.makedirs("models", exist_ok=True)
joblib.dump({"model": clf, "vectorizer": vectorizer}, "models/fast_model.joblib")
print("\n💾 Model saved to: models/fast_model.joblib")

# ===============================
# 7. Confusion Matrix
# ===============================
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Mental Health Classifier")
plt.tight_layout()

os.makedirs("models", exist_ok=True)
plt.savefig("models/confusion_matrix.png")
print("📊 Saved confusion matrix: models/confusion_matrix.png")
plt.show()
