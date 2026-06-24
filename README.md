# Phishing Email Detection System
**B207 Cyber Security | Gisma University of Applied Sciences**
**Author: Baljinder Singh**

Dataset: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

---

## What It Does

This tool analyses the text of an email and classifies it as **PHISHING** or **LEGITIMATE** using a machine learning model trained on real phishing email data. It combines TF-IDF text features with 13 hand-crafted cybersecurity indicators (URL count, urgency keywords, IP-based links, etc.) fed into a Random Forest classifier.

Every analysis result is saved to a local SQLite database for audit and reporting.

---

## Files

```
phishing-detector/
├── detector.py        # Core: feature extraction, ML model, SQLite database
├── train.py           # Train the model on the dataset
├── analyse.py         # CLI: analyse emails, view history
├── setup.sh           # One-command setup and training
├── requirements.txt   # Python dependencies
└── data/
    ├── phishing_email.csv   # Kaggle dataset (download separately — see below)
    ├── results.db           # SQLite database (auto-created)
    └── model.pkl            # Trained model (auto-created)
```

---

## Quick Start (One Command)

```bash
chmod +x setup.sh
./setup.sh
```

This will automatically:
1. Check Python 3.8+
2. Create a virtual environment
3. Install all dependencies
4. Initialise the database
5. Train the model

---

## Dataset Setup (Recommended)

For best accuracy, use the real Kaggle dataset:

1. Go to: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
2. Download `phishing_email.csv`
3. Place it at: `data/phishing_email.csv`
4. Run `./setup.sh` or just `python train.py`

> If the CSV is not present, the system automatically uses a synthetic fallback dataset and clearly logs this. The project still runs fully end-to-end.

---

## Usage

### Analyse a single email (inline text)
```bash
python analyse.py --text "URGENT: Your PayPal account has been suspended. Click http://192.168.1.1/verify NOW!"
```

### Analyse an email from a file
```bash
python analyse.py --file path/to/email.txt
```

### Interactive mode (paste multiple emails one by one)
```bash
python analyse.py --interactive
# Type or paste email text, press Enter twice to submit
# Type 'quit' to exit
```

### View analysis history (from SQLite database)
```bash
python analyse.py --history
python analyse.py --history --limit 50
```

### Retrain the model
```bash
python train.py
python train.py --csv data/phishing_email.csv
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| scikit-learn | >=1.3.0 | TF-IDF vectoriser and Random Forest classifier |
| pandas | >=2.0.0 | Dataset loading and preprocessing |
| numpy | >=1.24.0 | Numerical feature matrix operations |
| scipy | >=1.11.0 | Sparse matrix horizontal stacking (hstack) |

---

## Model Performance (on Kaggle dataset)

| Metric | Score |
|---|---|
| Accuracy | ~97% |
| Precision | ~96% |
| Recall | ~98% |
| F1 Score | ~97% |
