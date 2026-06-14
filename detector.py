import re
import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


_URL = re.compile(r"http[s]?://\S+", re.I)
_URGENT = re.compile(r"\b(urgent|verify|suspend|click here|free|login|password|bank)\b", re.I)

def extract_basic_features(text: str) -> dict:
    """Extract baseline numeric signals from raw email text."""
    urls = _URL.findall(text)
    words = text.split()
    n_words = max(len(words), 1)

    return {
        "num_urls": len(urls),
        "num_urgent": len(_URGENT.findall(text)),
        "urgent_ratio": len(_URGENT.findall(text)) / n_words,
        "text_length": len(text),
    }

class PhishingDetectorBaseline:
    """
    Early prototype using a standard TF-IDF text matrix 
    to train a baseline Random Forest classifier.
    """
    def __init__(self):
        self.vec = TfidfVectorizer(max_features=2000, stop_words='english')
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.trained = False

    def fit(self, texts, labels):
        log.info("Splitting dataset and extracting text vectors...")
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

       
        tfidf_train = self.vec.fit_transform(X_train)
        tfidf_test = self.vec.transform(X_test)

        log.info("Training baseline Random Forest...")
        self.clf.fit(tfidf_train, y_train)
        self.trained = True

        
        predictions = self.clf.predict(tfidf_test)
        print("\n=== Baseline Classification Report ===")
        print(classification_report(y_test, predictions))

    def predict(self, text: str):
        if not self.trained:
            raise RuntimeError("Model must be trained before predicting.")
        tfidf_v = self.vec.transform([text])
        prob = self.clf.predict_proba(tfidf_v)[0][1]
        label = "PHISHING" if prob >= 0.5 else "LEGITIMATE"
        return label, float(prob)


if __name__ == "__main__":

    sample_texts = [
        "URGENT: Click here to verify your banking credentials immediately!",
        "Hey Baljinder, are you available for a project synchronization call tomorrow?",
        "Get a free voucher now by clicking this link!",
        "Please review the attached project schedule before our team synchronization."
    ]
    sample_labels = [1, 0, 1, 0]

    detector = PhishingDetectorBaseline()
    detector.fit(sample_texts, sample_labels)
    
    test_email = "URGENT update your password here"
    verdict, confidence = detector.predict(test_email)
    print(f"\nTest Verdict: {verdict} ({confidence*100:.2f}%)")
