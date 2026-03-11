import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already available
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean text by removing links, punctuation, and stopwords.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [word for word in text.split() if word not in STOPWORDS and len(word) > 2]
    return " ".join(tokens)

def load_and_clean_data(path):
    """
    Load dataset and apply text cleaning.
    """
    df = pd.read_csv(path)
    df.dropna(subset=['text', 'label'], inplace=True)
    df['clean_text'] = df['text'].apply(clean_text)
    return df
