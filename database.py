"""SQLite storage for news records and prediction results."""
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "fake_news.db"


def _conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with closing(_conn()) as con, con:
        con.execute("""CREATE TABLE IF NOT EXISTS news_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT, text TEXT, label TEXT, category TEXT)""")
        con.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT (datetime('now','localtime')),
            input_text TEXT, label TEXT, fake_prob REAL,
            user_feedback TEXT)""")


def replace_news_records(df: pd.DataFrame, limit: int = 5000):
    """Store (a sample of) the training dataset as news records."""
    init_db()
    if len(df) > limit:
        df = df.sample(limit, random_state=1)
    cols = [c for c in ["title", "text", "label", "category"] if c in df.columns]
    out = df[cols].copy()
    for c in ["title", "text", "label", "category"]:
        if c not in out:
            out[c] = None
    with closing(_conn()) as con, con:
        con.execute("DELETE FROM news_records")
        out[["title", "text", "label", "category"]].to_sql("news_records", con, if_exists="append", index=False)


def news_label_counts() -> pd.DataFrame:
    init_db()
    with closing(_conn()) as con:
        return pd.read_sql_query("SELECT label, COUNT(*) AS count FROM news_records GROUP BY label", con)


def save_prediction(text: str, label: str, fake_prob: float) -> int:
    init_db()
    with closing(_conn()) as con, con:
        cur = con.execute("INSERT INTO predictions (input_text, label, fake_prob) VALUES (?,?,?)",
                          (text, label, float(fake_prob)))
        return cur.lastrowid


def set_feedback(pred_id: int, feedback: str):
    with closing(_conn()) as con, con:
        con.execute("UPDATE predictions SET user_feedback=? WHERE id=?", (feedback, pred_id))


def get_predictions(limit: int = 500) -> pd.DataFrame:
    init_db()
    with closing(_conn()) as con:
        return pd.read_sql_query(
            "SELECT id, created_at, label, fake_prob, user_feedback, input_text "
            "FROM predictions ORDER BY id DESC LIMIT ?", con, params=(limit,))


def prediction_stats() -> dict:
    df = get_predictions(100000)
    return {
        "total": len(df),
        "fake": int((df["label"] == "Potentially Fake").sum()),
        "real": int((df["label"] == "Likely Real").sum()),
        "uncertain": int((df["label"] == "Uncertain").sum()),
    }
