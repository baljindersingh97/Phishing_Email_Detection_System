import re, os, sqlite3, pickle, logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix

# ── paths ──────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "data", "results.db")
MODEL   = os.path.join(BASE, "data", "model.pkl")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── regex patterns ─────────────────────────────────────────────────────────
_URL     = re.compile(r"http[s]?://\S+", re.I)
_IP_URL  = re.compile(r"http[s]?://\d{1,3}(\.\d{1,3}){3}", re.I)
_CAPS    = re.compile(r"\b[A-Z]{4,}\b")
_HTML    = re.compile(r"<[^>]+>")
_DOLLAR  = re.compile(r"\$[\d,]+")


# ── feature extraction ─────────────────────────────────────────────────────

def extract_features(text: str) -> dict:
    """Return a dict of interpretable numeric features for one email."""
    urls     = _URL.findall(text)
    words    = text.split()
    n_words  = max(len(words), 1)

    return {
        "num_urls":          len(urls),
        "has_ip_url":        int(bool(_IP_URL.search(text))),
        "avg_url_len":       sum(len(u) for u in urls) / max(len(urls), 1),
        "num_urgent":        len(_URGENT.findall(text)),
        "num_exclaim":       text.count("!"),
        "num_dollars":       len(_DOLLAR.findall(text)),
        "has_form":          int("<form"   in text.lower()),
        "text_length":       len(text),
        "num_words":         n_words,
    }


def _feat_matrix(texts):
    """Stack feature dicts into a sparse matrix (rows = emails)."""
    keys = sorted(extract_features("").keys())
    rows = [[extract_features(t)[k] for k in keys] for t in texts]
    return csr_matrix(np.array(rows, dtype=float)), keys


# ── database ───────────────────────────────────────────────────────────────

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS results (
            ts          TEXT,
            subject     TEXT,
            snippet     TEXT
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            trained_at  TEXT,
            accuracy    REAL,
            precision_v REAL,
            recall_v    REAL,
        )""")
    con.commit(); con.close()
    log.info("DB ready  →  %s", DB_PATH)


def save_result(subject, text, verdict, confidence, feats) -> int:
    con = sqlite3.connect(DB_PATH)
    cur = con.execute("""
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
