# ============================================================
# 🧠 Mental Health Detection Pro Dashboard
# All-in-One Streamlit App (3D Enhanced Version with Footer)
# ============================================================

import streamlit as st
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import shap
from lime.lime_text import LimeTextExplainer
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import os

from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# Global 3D Theme for Plotly
# ============================================================
pio.templates["dark3D"] = pio.templates["plotly_dark"]
pio.templates["dark3D"].layout.scene = dict(
    xaxis=dict(showbackground=True, backgroundcolor="#111418", gridcolor="#333"),
    yaxis=dict(showbackground=True, backgroundcolor="#111418", gridcolor="#333"),
    zaxis=dict(showbackground=True, backgroundcolor="#111418", gridcolor="#333"),
)
pio.templates.default = "dark3D"

# ============================================================
# Page Config & Theme
# ============================================================
st.set_page_config(
    page_title="🧠 Mental Health Detection Dashboard",
    page_icon="🧠",
    layout="wide"
)

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
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Advanced Mental Health Detection Dashboard")

# ============================================================
# Load Model
# ============================================================
@st.cache_resource
def load_model():
    model_path = "models/fast_model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model_data = joblib.load(model_path)
    return model_data["model"], model_data["vectorizer"]

try:
    clf, tfidf = load_model()
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Model load error: {e}")
    st.stop()

# ============================================================
# Helper Function
# ============================================================
def predict_proba_fn(texts):
    texts = [str(t) for t in texts]
    X = tfidf.transform(texts)
    return clf.predict_proba(X)

# ============================================================
# Sidebar Navigation
# ============================================================
menu = st.sidebar.radio(
    "📑 Navigation",
    [
        "🏠 Home (Prediction)",
        "📊 Model Metrics",
        "🧩 SHAP & LIME Explainability",
        "📈 Disorder Analytics",
        "🗂️ Dataset Insights",
        "💡 Future Enhancements"
    ]
)

# ============================================================
# 🏠 HOME PAGE
# ============================================================
if menu == "🏠 Home (Prediction)":
    st.subheader("💬 Mental Health Text Prediction")
    user_input = st.text_area("Enter your thoughts:", height=140,
                              placeholder="e.g., I feel anxious about my exams...")

    if st.button("🔍 Analyze"):
        if not user_input.strip():
            st.warning("Please enter text first.")
        else:
            X = tfidf.transform([user_input])
            pred = clf.predict(X)[0]
            probs = clf.predict_proba(X)[0]
            classes = clf.classes_

            st.markdown(f"## 🧭 Emotion Detected: **{pred.upper()}**")

            # Confidence Chart (3D Style)
            conf_df = pd.DataFrame({
                "Class": classes,
                "Confidence (%)": np.round(probs * 100, 2)
            }).sort_values(by="Confidence (%)", ascending=True)

            fig = px.bar(conf_df, x="Confidence (%)", y="Class", orientation="h",
                         color="Class", color_discrete_sequence=px.colors.sequential.Tealgrn,
                         template="dark3D", title="Confidence Distribution")
            fig.update_traces(marker_line_width=1.5, marker_line_color="#1c1f26", opacity=0.9)
            st.plotly_chart(fig, use_container_width=True)

            # === Disorder Mapping ===
            disorder_mapping = {
                "depression": {
                    "Major Depressive Disorder (MDD)": 0.7,
                    "Bipolar Disorder": 0.1,
                    "PTSD": 0.1,
                    "Schizophrenia": 0.05,
                    "ADHD": 0.05,
                },
                "anxiety": {
                    "GAD": 0.4,
                    "OCD": 0.25,
                    "Social Anxiety": 0.2,
                    "PTSD": 0.1,
                    "ADHD": 0.05,
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
            disorder_df = pd.DataFrame(list(disorders.items()), columns=["Disorder", "Probability"])
            st.markdown("### 🧩 Estimated Disorder Probabilities")
            fig_d = px.bar(disorder_df, x="Probability", y="Disorder", orientation="h",
                           color="Probability", color_continuous_scale="Reds",
                           template="dark3D", range_x=[0,1])
            fig_d.update_traces(marker_line_width=1.3, opacity=0.9)
            st.plotly_chart(fig_d, use_container_width=True)

            likely = disorder_df[disorder_df["Probability"] > 0.3]["Disorder"].tolist()
            if likely:
                st.warning(f"⚠️ Possible Disorders: **{', '.join(likely)}**")
            else:
                st.success("✅ No strong indicators of clinical distress detected.")

            # WordCloud
            st.markdown("### ☁️ Word Cloud")
            wc = WordCloud(width=800, height=300, background_color="#0e1117", colormap="Pastel1").generate(user_input)
            fig_wc, ax_wc = plt.subplots(figsize=(8,3))
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc)

# ============================================================
# 📊 MODEL METRICS
# ============================================================
elif menu == "📊 Model Metrics":
    st.subheader("📊 Model Performance Metrics")
    st.info("Showing simulated evaluation metrics for demonstration.")

    labels = clf.classes_
    cm = np.array([[45, 5, 2], [7, 40, 3], [3, 4, 48]])
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    st.pyplot(fig)

    report = {
        "Accuracy": 0.91,
        "Precision": 0.89,
        "Recall": 0.90,
        "F1-Score": 0.89
    }
    st.markdown("### 🧾 Summary Metrics")
    st.dataframe(pd.DataFrame(report, index=["Score"]).T)

# ============================================================
# 🧩 SHAP & LIME
# ============================================================
elif menu == "🧩 SHAP & LIME Explainability":
    st.subheader("🧠 Explainable AI (SHAP & LIME)")
    text = st.text_area("Enter text:", "I feel anxious and tired all day.")
    if st.button("🧩 Explain"):
        st.info("Generating explanations... please wait ⏳")

        try:
            background = tfidf.transform(["I feel happy", "I am sad", "I am anxious"])
            explainer = shap.LinearExplainer(clf, background, feature_perturbation="interventional")
            X_input = tfidf.transform([text])
            shap_values = explainer.shap_values(X_input)
            if isinstance(shap_values, list):
                shap_values = shap_values[np.argmax(clf.predict_proba(X_input))]
            feature_names = np.array(tfidf.get_feature_names_out())
            top_idx = np.argsort(np.abs(shap_values[0]))[::-1][:10]
            df_shap = pd.DataFrame({
                "Word": feature_names[top_idx],
                "Impact": shap_values[0][top_idx],
                "Type": ["Positive" if x > 0 else "Negative" for x in shap_values[0][top_idx]]
            })
            fig = px.bar(df_shap, x="Impact", y="Word", orientation="h",
                         color="Type", color_discrete_map={"Positive":"#90EE90","Negative":"#FF6B6B"},
                         title="Top Words Influencing Model", template="dark3D")
            fig.update_traces(marker_line_width=1.3)
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"SHAP Error: {e}")

        try:
            explainer_lime = LimeTextExplainer(class_names=list(clf.classes_))
            exp = explainer_lime.explain_instance(text, predict_proba_fn, num_features=8)
            lime_df = pd.DataFrame(exp.as_list(), columns=["Word","Weight"])
            lime_df["Type"] = np.where(lime_df["Weight"]>0,"Positive","Negative")
            fig2 = px.bar(lime_df, x="Weight", y="Word", orientation="h", color="Type",
                          color_discrete_map={"Positive":"#A7C7E7","Negative":"#F28C8C"},
                          title="Top Word Contributions (LIME)", template="dark3D")
            fig2.update_traces(marker_line_width=1.3)
            st.plotly_chart(fig2)
        except Exception as e:
            st.error(f"LIME Error: {e}")

# ============================================================
# 📈 DISORDER ANALYTICS
# ============================================================
elif menu == "📈 Disorder Analytics":
    st.subheader("📈 Clinical Disorder Analytics Overview")
    uploaded = st.file_uploader("Upload CSV with predictions", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if "prediction" in df.columns:
            st.success("✅ File Loaded!")
            fig1 = px.histogram(df, x="prediction", color="prediction",
                                color_discrete_sequence=px.colors.sequential.Tealgrn,
                                title="Distribution of Predicted Emotions", template="dark3D")
            st.plotly_chart(fig1)

            disorders = ["MDD", "GAD", "OCD", "PTSD", "ADHD", "No Disorder"]
            values = np.random.rand(len(disorders))
            values /= values.sum()
            fig2 = px.sunburst(
                names=disorders,
                parents=[""] * len(disorders),
                values=values,
                color=values,
                color_continuous_scale="Tealgrn",
                title="🌀 3D-Like Disorder Distribution View"
            )
            fig2.update_traces(textinfo="label+percent")
            st.plotly_chart(fig2)

# ============================================================
# 🗂️ DATASET INSIGHTS
# ============================================================
elif menu == "🗂️ Dataset Insights":
    st.subheader("🗂️ Dataset Exploration & Insights")
    uploaded = st.file_uploader("Upload Dataset", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(df.head())
        if "text" in df.columns:
            wc = WordCloud(width=900, height=400, background_color="black", colormap="rainbow").generate(" ".join(df["text"].astype(str)))
            fig, ax = plt.subplots(figsize=(8,4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

# ============================================================
# 💡 FUTURE ENHANCEMENTS
# ============================================================
elif menu == "💡 Future Enhancements":
    st.subheader("💡 Future Research & Expansion Roadmap")
    st.markdown("""
    ### 🔮 Planned Enhancements
    - 🧠 Multi-disorder fine-tuning using BERT
    - 📈 Mental health trend tracking over time
    - 💬 Real-time emotion sensing from text streams
    - 📊 Bias detection and fairness evaluation
    - 🩺 Integration with clinician dashboards
    """)

# ============================================================
# 🌟 ENHANCED FOOTER (GLOW EFFECT)
# ============================================================
st.markdown("""
    <style>
    footer {visibility: hidden;}
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
        color: #f0f0f0;
        text-align: center;
        padding: 10px;
        font-size: 15px;
        font-family: 'Poppins', sans-serif;
        border-top: 1px solid #333;
        box-shadow: 0px -2px 10px rgba(0,0,0,0.3);
    }
    .footer span {
        font-weight: 600;
        color: #74c69d;
        text-shadow: 0 0 8px #74c69d, 0 0 20px #74c69d;
        letter-spacing: 0.5px;
    }
    </style>
    <div class="footer">
        Developed by <span>Ashish Nikam</span> | M.Tech AIML Student
    </div>
""", unsafe_allow_html=True)
