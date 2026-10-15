"""Train the Fake News classifier: TF-IDF + Logistic Regression.

Dataset priority:
  1. data/Fake.csv + data/True.csv   (Kaggle "Fake and Real News Dataset", columns: title, text)
  2. data/news_dataset.csv           (bundled synthetic starter data; columns: title, text, label)

Run:  python train_model.py
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import database as db
from preprocess import clean_text

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "fake_news_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def load_dataset():
    """Return (DataFrame with columns text,label[0=real,1=fake], source_name)."""
    fake_p, true_p = DATA_DIR / "Fake.csv", DATA_DIR / "True.csv"
    if fake_p.exists() and true_p.exists():
        fake, true = pd.read_csv(fake_p), pd.read_csv(true_p)
        fake["label"], true["label"] = "FAKE", "REAL"
        df = pd.concat([fake, true], ignore_index=True)
        source = "Kaggle Fake.csv + True.csv"
    else:
        df = pd.read_csv(DATA_DIR / "news_dataset.csv")
        source = "Bundled synthetic sample (replace with a real dataset!)"
    df["title"] = df.get("title", "").fillna("")
    df["text"] = df["text"].fillna("")
    df["content"] = (df["title"] + " " + df["text"]).str.strip()
    df["y"] = (df["label"].astype(str).str.upper() == "FAKE").astype(int)
    df = df[df["content"].str.len() > 0].drop_duplicates(subset="content").reset_index(drop=True)
    return df, source


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=2,
                                  max_features=30000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=2.0)),
    ])


def train():
    df, source = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["y"], test_size=0.2, random_state=42, stratify=df["y"])

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    names = vec.get_feature_names_out()
    order = np.argsort(clf.coef_[0])
    top_fake = [(names[i], float(clf.coef_[0][i])) for i in order[::-1][:15]]
    top_real = [(names[i], float(clf.coef_[0][i])) for i in order[:15]]

    metrics = {
        "dataset_source": source,
        "n_total": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "top_fake_words": top_fake, "top_real_words": top_real,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    db.replace_news_records(df[["title", "text"]].assign(
        label=np.where(df["y"] == 1, "FAKE", "REAL"),
        category=df["category"] if "category" in df else None))
    return metrics


if __name__ == "__main__":
    m = train()
    print(f"Dataset : {m['dataset_source']} ({m['n_total']} rows)")
    print(f"Accuracy: {m['accuracy']:.3f} | Precision: {m['precision']:.3f} | "
          f"Recall: {m['recall']:.3f} | F1: {m['f1']:.3f}")
    print("Confusion matrix [[TN, FP],[FN, TP]]:", m["confusion_matrix"])
    print("Model saved to", MODEL_PATH)
