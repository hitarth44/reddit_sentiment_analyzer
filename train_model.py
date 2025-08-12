# train_model.py
import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load cleaned comments data
df = pd.read_csv("data/clean_comments.csv")

# Apply VADER sentiment scoring
analyzer = SentimentIntensityAnalyzer()
df['vader_compound'] = df['clean_body'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

# Label function
def label(c):
    if c >= 0.05:
        return "positive"
    if c <= -0.05:
        return "negative"
    return "neutral"

# Apply labels
df['label'] = df['vader_compound'].apply(label)

# Filter out very short comments
df = df[df['clean_body'].str.split().str.len() >= 2]

# Split data
X = df['clean_body']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build pipeline (removed multi_class, added class_weight='balanced')
pipe = make_pipeline(
    TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english'),
    LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')
)

# Train
pipe.fit(X_train, y_train)

# Predict & evaluate
y_pred = pipe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save trained model
os.makedirs("models", exist_ok=True)
joblib.dump(pipe, "models/sentiment_model.joblib")
print("[+] Saved trained model to models/sentiment_model.joblib")
