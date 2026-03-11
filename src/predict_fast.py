import joblib
model = joblib.load("models/fast_model.joblib")
clf, tfidf = model["model"], model["vectorizer"]

while True:
    text = input("\nEnter text (or 'q' to quit): ")
    if text.lower() == "q": break
    vec = tfidf.transform([text])
    pred = clf.predict(vec)[0]
    print("🧠 Prediction:", pred)
