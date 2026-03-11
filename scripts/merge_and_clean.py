import pandas as pd
import re
from pathlib import Path

# Paths
RAW_PATH = Path("data/raw/Mental-Health-Twitter.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = PROCESSED_DIR / "merged.csv"

# Load data
print("📥 Loading dataset...")
df = pd.read_csv(RAW_PATH)
print("✅ Loaded:", df.shape)

# --- Inspect columns to choose text & label ---
print("Columns:", df.columns.tolist())
print(df.head(3))

# --- Cleaning helpers ---
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)       # remove URLs
    text = re.sub(r"@\w+", "", text)                 # remove mentions
    text = re.sub(r"#\w+", "", text)                 # remove hashtags
    text = re.sub(r"[^A-Za-z0-9\s.,!?']", " ", text) # keep letters, digits, punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Choose the right columns ---
# Adjust below if your dataset uses different column names
if 'Question' in df.columns and 'Depression' in df.columns:
    df = df[['Question', 'Depression']]
    df = df.rename(columns={'Question': 'text', 'Depression': 'label'})
elif 'Text' in df.columns and 'Label' in df.columns:
    df = df[['Text', 'Label']]
    df = df.rename(columns={'Text': 'text', 'Label': 'label'})
else:
    # fallback - automatically choose first string-like columns
    text_col = df.select_dtypes(include='object').columns[0]
    label_col = df.select_dtypes(include='object').columns[-1]
    df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})

# --- Clean text ---
print("🧹 Cleaning text...")
df['text'] = df['text'].apply(clean_text)
df = df.dropna(subset=['text', 'label'])
df = df[df['text'].str.len() > 5]

# --- Normalize labels ---
df['label'] = df['label'].astype(str).str.lower().str.strip()
df['label'] = df['label'].replace({
    'yes': 'depression',
    'no': 'no_stress',
    'true': 'depression',
    'false': 'no_stress',
    'anxiety': 'anxiety'
})

# --- Remove duplicates ---
before = len(df)
df = df.drop_duplicates(subset=['text'])
after = len(df)
print(f"🧾 Removed {before - after} duplicate rows.")

# --- Save cleaned dataset ---
df.to_csv(OUTPUT_PATH, index=False)
print(f"💾 Cleaned dataset saved to: {OUTPUT_PATH}")
print("Final shape:", df.shape)
print(df['label'].value_counts())
