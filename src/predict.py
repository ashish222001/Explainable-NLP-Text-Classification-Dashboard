import joblib
import sys

# Load model
model = joblib.load("models/model.joblib")

def predict_text(text):
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    return prediction, dict(zip(model.classes_, probabilities))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter text to analyze: ")
    pred, probs = predict_text(text)
    print(f"\nPrediction: {pred}")
    print("Probabilities:")
    for cls, p in probs.items():
        print(f"  {cls}: {p:.3f}")
