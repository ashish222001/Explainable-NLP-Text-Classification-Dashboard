import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from data_prep import load_and_clean_data

def train_model(train_csv, model_path='models/model.joblib'):
    df = load_and_clean_data(train_csv)
    X = df['clean_text']
    y = df['label']

    # Split for quick evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Pipeline: TF-IDF + Logistic Regression
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=500, class_weight='balanced'))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\n✅ Classification Report:\n")
    print(classification_report(y_test, preds))

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    plt.show()

    # Save model
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n🎉 Model saved to: {model_path}")

if __name__ == "__main__":
    train_model("data/train.csv")
