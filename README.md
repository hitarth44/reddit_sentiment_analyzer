# Stock & Reddit Sentiment Analyzer

This project analyzes Reddit discussions about stocks, identifies the most mentioned tickers, and measures the sentiment toward them.  
The included training script uses **VADER sentiment scoring** to label comments, and then trains a **TF-IDF + Logistic Regression** model to predict sentiment.

---

## ✨ Features

- Reads pre-cleaned Reddit comment data from CSV.
- Uses **VADER SentimentIntensityAnalyzer** to assign labels:
  - `positive` if compound ≥ 0.05
  - `negative` if compound ≤ -0.05
  - `neutral` otherwise
- Trains a TF-IDF + Logistic Regression pipeline.
- Uses `class_weight='balanced'` to handle class imbalance.
- Saves the trained model for future predictions.

---

## 📦 Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
