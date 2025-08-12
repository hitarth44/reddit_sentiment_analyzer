# Stock & Reddit Sentiment Analyzer

Reddit Stock Sentiment Analyzer — Built a sentiment classification pipeline for Reddit stock discussions using Python, scikit-learn, TF-IDF vectorization, and VADER sentiment analysis. Implemented balanced Logistic Regression for imbalanced datasets, achieved ~68% accuracy. Deployed model to predict sentiment from new user input. Used Git for version control and hosted project on GitHub.
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
