import streamlit as st
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import os
from lime.lime_text import LimeTextExplainer
import plotly.express as px
import shap

# --------------------------
# Streamlit Page Config
# --------------------------
st.set_page_config(
    page_title="🧠 Advanced Mental Health Detection App",
    page_icon="🧠",
    layout="wide"
)

# 🌙 Custom Theme
st.markdown("""
    <style>
    body, .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTabs [data-baseweb="tab"] {
        color: #f0f0f0;
        background-color: #1c1f26;
        border-radius: 8px;
        margin-right: 5px;
        padding: 6px 12px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2b2f3a;
        color: #74c69d;
        font-weight: bold;
    }
    .stButton > button {
        color: white;
        background-color: #4CAF50;
        border-radius: 8px;
        border: none;
        padding: 0.6em 1.2em;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Advanced Mental Health Detection Dashboard")
st.write(
    "AI-based detection of **Anxiety**, **Depression**, or **No Stress**, "
    "mapped to **real-world mental disorders** (MDD, GAD, OCD, PTSD, etc.) with explainability (SHAP / LIME)."
)

# --------------------------
# Load Model
# --------------------------
@st.cache_resource
def load_model():
    model_path = "models/fast_model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model_data = joblib.load(model_path)
    return model_data["model"], model_data["vectorizer"]

try:
    clf, tfidf = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model load error: {e}")
    st.stop()

def predict_proba_fn(texts):
    texts = [str(t) for t in texts]
    X = tfidf.transform(texts)
    return clf.predict_proba(X)

# --------------------------
# Tabs
# --------------------------
tab1, tab2, tab3 = st.tabs([
    "💬 Single Text Prediction",
    "📂 Bulk CSV Analysis",
    "🔍 Explainability (SHAP / LIME)"
])

# =====================================================
# TAB 1: Single Text Prediction
# =====================================================
with tab1:
    st.subheader("🗣️ Enter your thoughts or message:")
    user_input = st.text_area("Type your text below:", height=140,
        placeholder="e.g., I feel anxious about my future...")

    crisis_keywords = ["die", "suicide", "kill myself", "hopeless", "worthless", "end my life"]

    if st.button("🔍 Analyze Mental Health"):
        if not user_input.strip():
            st.warning("⚠️ Please enter some text first.")
        else:
            text_lower = user_input.lower()
            if any(k in text_lower for k in crisis_keywords):
                st.error("🚨 Crisis alert detected! Please seek immediate professional help.")

            X = tfidf.transform([user_input])
            pred = clf.predict(X)[0]
            probs = clf.predict_proba(X)[0]
            classes = clf.classes_

            st.markdown(f"## 🧭 Primary Emotion: **{pred.upper()}**")

            # --- Emotion confidence bar ---
            conf_df = pd.DataFrame({
                "Class": classes,
                "Confidence (%)": np.round(probs * 100, 2)
            }).sort_values(by="Confidence (%)", ascending=True)
            fig = px.bar(conf_df, x="Confidence (%)", y="Class", orientation="h",
                         color="Class", color_discrete_sequence=px.colors.qualitative.Pastel,
                         template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)

            # --- Clinical Disorder Mapping ---
            disorder_mapping = {
                "depression": {
                    "Major Depressive Disorder (MDD)": 0.7,
                    "Bipolar Disorder": 0.1,
                    "Post-Traumatic Stress Disorder (PTSD)": 0.1,
                    "Schizophrenia": 0.05,
                    "Attention-Deficit/Hyperactivity Disorder (ADHD)": 0.05,
                },
                "anxiety": {
                    "Generalized Anxiety Disorder (GAD)": 0.4,
                    "Obsessive-Compulsive Disorder (OCD)": 0.25,
                    "Social Anxiety Disorder": 0.2,
                    "Post-Traumatic Stress Disorder (PTSD)": 0.1,
                    "Attention-Deficit/Hyperactivity Disorder (ADHD)": 0.05,
                },
                "no_stress": {
                    "Healthy / No Disorder": 0.7,
                    "Mild Stress": 0.1,
                    "Adjustment Disorder": 0.1,
                    "Minor Anxiety": 0.05,
                    "Other": 0.05,
                }
            }

            disorders = disorder_mapping.get(pred.lower(), {})
            disorder_df = pd.DataFrame(list(disorders.items()),
                                       columns=["Disorder", "Probability"])
            st.markdown("### 🧩 Estimated Disorder Probabilities")
            fig_d = px.bar(disorder_df, x="Probability", y="Disorder",
                           orientation="h", color="Probability",
                           color_continuous_scale="Reds", range_x=[0,1],
                           template="plotly_dark", height=400)
            fig_d.update_layout(xaxis_title="Probability", yaxis_title="Disorder")
            st.plotly_chart(fig_d, use_container_width=True)

            likely = disorder_df[disorder_df["Probability"] > 0.3]["Disorder"].tolist()
            if likely:
                st.warning(f"⚠️ Possible disorders based on your message: **{', '.join(likely)}**")
            else:
                st.success("✅ No strong indicators of clinical-level distress detected.")

            # --- Word Cloud ---
            st.markdown("### ☁️ Word Cloud")
            wc = WordCloud(width=800, height=300, background_color="#0e1117",
                           colormap="Pastel1").generate(user_input)
            fig_wc, ax_wc = plt.subplots(figsize=(8,3))
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc)

# =====================================================
# TAB 2: Bulk CSV Upload and Analysis
# =====================================================
with tab2:
    st.subheader("📂 Upload a CSV file for analysis")
    st.write("CSV must contain a **'text'** column.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "text" not in df.columns:
                st.error("❌ CSV must have a 'text' column.")
            else:
                X = tfidf.transform(df["text"].astype(str))
                df["prediction"] = clf.predict(X)
                st.success(f"✅ Processed {len(df)} entries.")
                st.dataframe(df.head(10))

                counts = df["prediction"].value_counts()
                fig2 = px.bar(x=counts.index, y=counts.values, color=counts.index,
                              color_discrete_sequence=px.colors.qualitative.Pastel,
                              template="plotly_dark", height=400)
                st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# =====================================================
# TAB 3: Explainability (SHAP & LIME)
# =====================================================
with tab3:
    st.subheader("🔍 Explain Model Decisions with SHAP & LIME")
    explain_text = st.text_area("Enter text for explanation:",
                                "I feel anxious and nervous about my job.", height=100)
    focus_mode = st.checkbox("🖥️ Focus Mode (expand charts)", value=True)

    if st.button("🧩 Generate Explanation"):
        if not explain_text.strip():
            st.warning("⚠️ Please enter some text.")
        else:
            st.info("Generating SHAP and LIME explanations... please wait ⏳")

            # --- SHAP ---
            try:
                st.markdown("### 🧬 SHAP Explanation")
                background = tfidf.transform(["I feel happy today", "I feel tired", "I am anxious"])
                explainer = shap.LinearExplainer(clf, background, feature_perturbation="interventional")
                X_input = tfidf.transform([explain_text])
                shap_values = explainer.shap_values(X_input)
                if isinstance(shap_values, list):
                    shap_values = shap_values[np.argmax(clf.predict_proba(X_input))]
                shap_values = np.array(shap_values).flatten()[: len(tfidf.get_feature_names_out())]
                feature_names = np.array(tfidf.get_feature_names_out())
                top_idx = np.argsort(np.abs(shap_values))[::-1][:8]
                shap_df = pd.DataFrame({
                    "Word": feature_names[top_idx][::-1],
                    "Impact": shap_values[top_idx][::-1],
                    "Contribution": ["Positive" if s>0 else "Negative" for s in shap_values[top_idx][::-1]]
                })
                fig3 = px.bar(shap_df, x="Impact", y="Word", orientation="h",
                              color="Contribution",
                              color_discrete_map={"Positive":"#90EE90","Negative":"#FF6B6B"},
                              title="Top Word Contributions (SHAP)",
                              template="plotly_dark", height=400 if focus_mode else 300)
                st.plotly_chart(fig3, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ SHAP explanation failed: {e}")

            # --- LIME ---
            try:
                st.markdown("### 🧠 LIME Explanation")
                explainer_lime = LimeTextExplainer(class_names=list(clf.classes_))
                exp = explainer_lime.explain_instance(explain_text, predict_proba_fn, num_features=8)
                lime_df = pd.DataFrame(exp.as_list(), columns=["Word","Weight"])
                lime_df["Contribution"] = np.where(lime_df["Weight"]>0,"Positive","Negative")
                fig_lime = px.bar(lime_df, x="Weight", y="Word", orientation="h",
                                  color="Contribution",
                                  color_discrete_map={"Positive":"#A7C7E7","Negative":"#F28C8C"},
                                  title="Top Word Contributions (LIME)",
                                  template="plotly_dark",
                                  height=400 if focus_mode else 300)
                st.plotly_chart(fig_lime, use_container_width=True)
                st.markdown("### ✨ Highlighted Text Analysis")
                highlighted = explain_text
                for word, weight in exp.as_list():
                    color = "#80ed99" if weight>0 else "#ff6b6b"
                    highlighted = highlighted.replace(
                        word, f"<mark style='background-color:{color}; color:black; padding:2px 4px; border-radius:4px;'>{word}</mark>")
                st.markdown(f"<div style='font-size:1.1rem; line-height:1.8;'>{highlighted}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ LIME explanation failed: {e}")
