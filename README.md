# Reddit Stock Sentiment Analyzer

This project analyzes Reddit discussions about stocks, identifies the most mentioned tickers, and measures sentiment toward them.  
It includes:

- A training script (`train_model.py`) that uses VADER sentiment scoring to label comments and trains a TF-IDF + Logistic Regression classifier.
- An exploratory data analysis notebook (`sentiment_analysis.ipynb`) with visualizations and insights.

---

## ✨ Features

- Reads pre-cleaned Reddit comment data from CSV.
- Uses **VADER SentimentIntensityAnalyzer** to assign sentiment labels:
  - `positive` if compound ≥ 0.05
  - `negative` if compound ≤ -0.05
  - `neutral` otherwise
- Trains a TF-IDF + Logistic Regression model with `class_weight='balanced'` to handle imbalance.
- Saves trained model for later predictions.
- Provides EDA and visualizations in a Jupyter notebook.

---

## 📦 Installation

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Prepare Your Data

Place your cleaned comments CSV in the `data/` folder.  
The file **must** contain a `clean_body` column with preprocessed text.

---

### 2. Train the Model

```bash
python train_model.py
```

---

### 3. Explore the Data

The `sentiment_analysis.ipynb` notebook:

- Shows dataset preview & structure.
- Visualizes sentiment class distribution.
- Plots sentiment trends over time.
- Generates word clouds for each sentiment.

Run it:

```bash
jupyter notebook sentiment_analysis.ipynb
```

---

### 4. Predict with the Trained Model

```python
import joblib

model = joblib.load("models/sentiment_model.joblib")
pred = model.predict(["This stock is going to the moon!"])
print(pred)  # Example: ['positive']
```

---

## 📊 Example Output

```
Accuracy: 0.684
              precision    recall  f1-score   support
negative       0.58       0.38      0.46        58
neutral        0.50       0.38      0.43        72
positive       0.74       0.87      0.80       218
```

---

## 📜 License

MIT License — Original work © 2021 [asad70], Modified & extended © 2025 [hitarth44]
