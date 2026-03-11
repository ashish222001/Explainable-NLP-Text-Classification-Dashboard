import streamlit as st
import joblib

# Load the trained model
model = joblib.load("models/model.joblib")

st.set_page_config(page_title="Mental Health Detector", layout="wide")
st.title("🧠 Mental Health Detection from Social Media Posts")

st.markdown("""
Type or paste a short social media post below.  
The model will predict the **mental health condition** category.
""")

user_input = st.text_area("Enter a post:", height=150)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("⚠️ Please enter text before analyzing.")
    else:
        prediction = model.predict([user_input])[0]
        probabilities = model.predict_proba([user_input])[0]

        st.subheader(f"Prediction: **{prediction.upper()}**")
        st.markdown("### Probabilities:")
        for label, prob in zip(model.classes_, probabilities):
            st.write(f"- {label}: **{prob:.3f}**")

st.caption("Project by Ashish • Model: Logistic Regression + TF-IDF")
