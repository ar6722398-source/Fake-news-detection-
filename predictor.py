"""Load the trained model and predict with word-level explanations."""
import json

import joblib

import train_model

_FEATURE_NAMES = {}


def load_model():
    """Load the saved pipeline; (re)train automatically if missing or incompatible."""
    if not train_model.MODEL_PATH.exists():
        train_model.train()
    try:
        return joblib.load(train_model.MODEL_PATH)
    except Exception:
        train_model.train()
        return joblib.load(train_model.MODEL_PATH)


def load_metrics():
    if not train_model.METRICS_PATH.exists():
        train_model.train()
    return json.loads(train_model.METRICS_PATH.read_text())


def label_from_prob(p: float) -> str:
    if p >= 0.60:
        return "Potentially Fake"
    if p <= 0.40:
        return "Likely Real"
    return "Uncertain"


def predict(pipe, text: str) -> dict:
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    X = vec.transform([text])
    fake_prob = float(clf.predict_proba(X)[0][list(clf.classes_).index(1)])
    key = id(vec)
    if key not in _FEATURE_NAMES:
        _FEATURE_NAMES[key] = vec.get_feature_names_out()
    names = _FEATURE_NAMES[key]

    contrib = sorted(((names[i], float(v * clf.coef_[0][i])) for i, v in zip(X.indices, X.data)),
                     key=lambda t: t[1])
    towards_real = [(w, round(c, 3)) for w, c in contrib[:6] if c < 0]
    towards_fake = [(w, round(c, 3)) for w, c in contrib[::-1][:6] if c > 0]
    known = X.nnz >= 5  # too few known words -> not enough evidence
    label = label_from_prob(fake_prob) if known else "Uncertain"
    return {
        "label": label,
        "fake_prob": fake_prob if known else 0.5,
        "confidence": max(fake_prob, 1 - fake_prob) if known else 0.5,
        "towards_fake": towards_fake,
        "towards_real": towards_real,
        "known_words": int(X.nnz),
    }
