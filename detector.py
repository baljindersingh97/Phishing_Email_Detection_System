
import re, os, sqlite3, pickle, logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report)
from scipy.sparse import hstack, csr_matrix

BASE    = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "data", "results.db")
MODEL   = os.path.join(BASE, "data", "model.pkl")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_URL     = re.compile(r"http[s]?://\S+", re.I)
_IP_URL  = re.compile(r"http[s]?://\d{1,3}(\.\d{1,3}){3}", re.I)
_URGENT  = re.compile(
    r"\b(urgent|verify|suspend|click here|free|winner|prize|limited|"
    r"offer|confirm|update|login|password|bank|credit|paypal|"
    r"act now|immediately|security alert|account|validate)\b", re.I)
_CAPS    = re.compile(r"\b[A-Z]{4,}\b")
_HTML    = re.compile(r"<[^>]+>")
_DOLLAR  = re.compile(r"\$[\d,]+")



def extract_features(text: str) -> dict:
    urls     = _URL.findall(text)
    words    = text.split()
    n_words  = max(len(words), 1)

    return {
        "num_urls":          len(urls),
        "has_ip_url":        int(bool(_IP_URL.search(text))),
        "avg_url_len":       sum(len(u) for u in urls) / max(len(urls), 1),
        "num_urgent":        len(_URGENT.findall(text)),
        "urgent_ratio":      len(_URGENT.findall(text)) / n_words,
        "num_html_tags":     len(_HTML.findall(text)),
        "num_caps":          len(_CAPS.findall(text)),
        "num_exclaim":       text.count("!"),
        "num_dollars":       len(_DOLLAR.findall(text)),
        "has_iframe":        int("<iframe" in text.lower()),
        "has_form":          int("<form"   in text.lower()),
        "text_length":       len(text),
        "num_words":         n_words,
    }


def _feat_matrix(texts):
    keys = sorted(extract_features("").keys())
    rows = [[extract_features(t)[k] for k in keys] for t in texts]
    return csr_matrix(np.array(rows, dtype=float)), keys


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT,
            subject     TEXT,
            snippet     TEXT,
            verdict     TEXT,
            confidence  REAL,
            num_urls    INTEGER,
            num_urgent  INTEGER,
            has_ip_url  INTEGER,
            txt_length  INTEGER
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            trained_at  TEXT,
            accuracy    REAL,
            precision_v REAL,
            recall_v    REAL,
            f1_v        REAL,
            n_samples   INTEGER
        )""")
    con.commit(); con.close()
    log.info("DB ready  →  %s", DB_PATH)


def save_result(subject, text, verdict, confidence, feats) -> int:
    con = sqlite3.connect(DB_PATH)
    cur = con.execute("""
        INSERT INTO results (ts,subject,snippet,verdict,confidence,
                             num_urls,num_urgent,has_ip_url,txt_length)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (datetime.utcnow().isoformat(), subject[:200], text[:300],
         verdict, round(confidence, 4),
         feats["num_urls"], feats["num_urgent"],
         feats["has_ip_url"], feats["text_length"]))
    rid = cur.lastrowid
    con.commit(); con.close()
    return rid


def save_metrics(m, n):
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT INTO metrics (trained_at,accuracy,precision_v,recall_v,f1_v,n_samples)
        VALUES (?,?,?,?,?,?)""",
        (datetime.utcnow().isoformat(),
         m["accuracy"], m["precision"], m["recall"], m["f1"], n))
    con.commit(); con.close()


def get_history(limit=20):
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("""
        SELECT id,ts,subject,verdict,confidence,num_urls,num_urgent
        FROM results ORDER BY id DESC LIMIT ?""", (limit,)).fetchall()
    con.close()
    return rows



class PhishingDetector:

    def __init__(self):
        self.vec = TfidfVectorizer(
            max_features=8000, ngram_range=(1, 2),
            sublinear_tf=True, min_df=2,
            token_pattern=r"\b[a-zA-Z]\w+\b")
        self.clf = RandomForestClassifier(
            n_estimators=200, max_depth=25,
            class_weight="balanced", random_state=42, n_jobs=-1)
        self._feat_keys = None
        self.trained    = False

    def fit(self, texts, labels):
        Xtr, Xte, ytr, yte = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels)

        log.info("Fitting TF-IDF on %d samples …", len(Xtr))
        tfidf_tr = self.vec.fit_transform(Xtr)
        tfidf_te = self.vec.transform(Xte)

        hand_tr, self._feat_keys = _feat_matrix(Xtr)
        hand_te, _               = _feat_matrix(Xte)

        log.info("Training Random Forest …")
        self.clf.fit(hstack([tfidf_tr, hand_tr]), ytr)
        self.trained = True

        ypred = self.clf.predict(hstack([tfidf_te, hand_te]))
        m = dict(
            accuracy  = round(accuracy_score(yte, ypred), 4),
            precision = round(precision_score(yte, ypred, zero_division=0), 4),
            recall    = round(recall_score(yte, ypred, zero_division=0), 4),
            f1        = round(f1_score(yte, ypred, zero_division=0), 4),
        )
        log.info("Results → %s", m)
        print(classification_report(yte, ypred, target_names=["Legit","Phish"]))
        return m

    def predict(self, text: str):
        if not self.trained:
            raise RuntimeError("Call fit() or load() first.")
        feats = extract_features(text)
        hand  = csr_matrix([[feats[k] for k in self._feat_keys]])
        tfidf = self.vec.transform([text])
        prob  = self.clf.predict_proba(hstack([tfidf, hand]))[0][1]
        label = "PHISHING" if prob >= 0.5 else "LEGITIMATE"
        return label, float(prob)

    def save(self):
        os.makedirs(os.path.dirname(MODEL), exist_ok=True)
        with open(MODEL, "wb") as f: pickle.dump(self, f)
        log.info("Model saved → %s", MODEL)

    @staticmethod
    def load():
        with open(MODEL, "rb") as f: return pickle.load(f)
