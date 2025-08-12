import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from argparse import ArgumentParser

# Download NLTK resources (only first run)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Default paths
DEFAULT_INPUT = "data/raw_comments.csv"
DEFAULT_OUTPUT = "data/clean_comments.csv"

# Argument parser (optional overrides)
parser = ArgumentParser(description="Preprocess Reddit comments")
parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to raw comments CSV (default: data/raw_comments.csv)")
parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to save cleaned comments (default: data/clean_comments.csv)")
args = parser.parse_args()

# Check if input file exists
if not os.path.exists(args.input):
    raise FileNotFoundError(f"❌ Input file not found: {args.input}\nRun fetch_reddit.py first.")

# Load data
df = pd.read_csv(args.input)

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)         # keep only letters
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

# Clean comments
df["clean_body"] = df["body"].apply(clean_text)

# Optional: add a date column from timestamp
if "created_utc" in df.columns:
    df["date"] = pd.to_datetime(df["created_utc"]).dt.date

# Save cleaned data
os.makedirs(os.path.dirname(args.output), exist_ok=True)
df.to_csv(args.output, index=False)

print(f"✅ Preprocessed {len(df)} comments.")
print(f"Saved cleaned comments to {args.output}")
print(df.head())
