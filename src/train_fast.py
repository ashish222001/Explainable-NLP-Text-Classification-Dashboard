"""
Fast Emotion Detection (uses HuggingFace 'emotion' dataset)
Runtime: ~1 minute on CPU
"""

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib, os

# -------------------------
# 1️⃣ Load ready-made dataset
# -------------------------
print("📥 Loading 'emotion' dataset from Hugging Face ...")
dataset = load_dataset("emotion")
df = pd.DataFrame(dataset["train"])
print("✅ Loaded:", df.shape)
print(df.head())

# Keep only text + label
df = df[["text", "label"]]
label_map = dataset["train"].features["label"].names
df["label"] = df["label"].apply(lambda x: label_map[x])

# Reduce to 3 key classes for simplicity
map3 = {
    "sadness": "depression",
    "joy": "no_stress",
    "anger": "anxiety",
    "fear": "anxiety",
    "love": "no_stress",
    "surprise": "no_stress",
}
df["label"] = df["label"].map(map3)

# -------------------------
# 2️⃣ Split data
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

# -------------------------
# 3️⃣ Vectorize + Train
# -------------------------
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\n🚀 Training Logistic Regression ...")
clf = LogisticRegression(max_iter=300)
clf.fit(X_train_tfidf, y_train)

# -------------------------
# 4️⃣ Evaluate
# -------------------------
y_pred = clf.predict(X_test_tfidf)
print("\n✅ Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------
# 5️⃣ Save model
# -------------------------
os.makedirs("models", exist_ok=True)
joblib.dump({"model": clf, "vectorizer": tfidf}, "models/fast_model.joblib")
print("\n💾 Model saved to models/fast_model.joblib ✅")
