"""Text preprocessing (NLP) + rule-based style signals for the Fake News Awareness System."""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
SPACE_RE = re.compile(r"\s+")

# Small extra stop-word list on top of scikit-learn's English list
EXTRA_STOP = {"said", "say", "says", "would", "also", "one", "reuters"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) | EXTRA_STOP


def clean_text(text) -> str:
    """NLP cleaning used by the TF-IDF vectorizer.

    Steps: lowercase -> remove URLs/HTML -> keep style cues ('!' and '?')
    as tokens -> remove digits/punctuation -> remove stop words.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\(reuters\)", " ", text)       # avoid dataset leakage
    text = URL_RE.sub(" urltoken ", text)
    text = HTML_RE.sub(" ", text)
    text = text.replace("!", " exclamationmark ").replace("?", " questionmark ")
    text = NON_ALPHA_RE.sub(" ", text)
    tokens = [t for t in SPACE_RE.split(text) if len(t) > 2 and t not in STOP_WORDS]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Rule-based "red flag" detector (shown next to the ML prediction for awareness)
# ---------------------------------------------------------------------------
CLICKBAIT_PHRASES = [
    "you won't believe", "shocking", "they don't want you to know", "share before",
    "before they delete", "doctors hate", "miracle", "exposed", "wake up",
    "mainstream media", "cover-up", "cover up", "going viral", "forward this",
    "must share", "100%", "one weird trick", "leaked", "secret plan",
    "do your own research",
]
ATTRIBUTION_WORDS = ["according to", "said", "told", "reported", "announced", "stated", "confirmed"]


def style_signals(text: str) -> list:
    """Return a list of human-readable red flags found in the text."""
    flags = []
    if not isinstance(text, str) or not text.strip():
        return flags
    lower = text.lower()

    n_excl = text.count("!")
    if n_excl >= 3:
        flags.append(f"Excessive exclamation marks ({n_excl}) - emotional, attention-grabbing tone.")

    words = re.findall(r"[A-Za-z]{3,}", text)
    if words:
        caps_ratio = sum(1 for w in words if w.isupper()) / len(words)
        if caps_ratio > 0.15:
            flags.append(f"Many ALL-CAPS words ({caps_ratio:.0%} of words) - a common sensationalism trick.")

    found = [p for p in CLICKBAIT_PHRASES if p in lower]
    if found:
        flags.append("Clickbait / pressure phrases found: " + ", ".join(f"'{p}'" for p in found[:5]) + ".")

    if len(text.split()) >= 40 and not any(a in lower for a in ATTRIBUTION_WORDS):
        flags.append("No clear attribution (no 'according to', 'said', 'announced'...) - who is the source?")

    if re.search(r"(anonymous|unnamed) source|insiders? (say|claim|reveal)|sources say", lower):
        flags.append("Relies on anonymous or unnamed sources.")
    return flags
