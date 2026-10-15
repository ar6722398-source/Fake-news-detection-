"""Fake News Awareness & Detection System - Streamlit app.

Run:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import database as db
import predictor
import train_model
from preprocess import style_signals

FAKE_EXAMPLE = (
    "SHOCKING!!! Government HIDING the truth about miracle cure - doctors hate this one weird trick! "
    "Insiders say the mainstream media refuses to report it. Share this with everyone you know "
    "before they delete it!!!"
)
REAL_EXAMPLE = (
    "The central bank said on Friday that it kept its benchmark interest rate unchanged after reviewing "
    "inflation data released by the statistics ministry. A spokesperson told reporters that the decision "
    "followed a two-day policy meeting and that the figures would be reviewed again next quarter. Analysts "
    "noted that retail inflation had eased slightly compared with the previous month, according to official data."
)

FACT_CHECK_SITES = [
    ("PIB Fact Check (India, government claims)", "https://factcheck.pib.gov.in"),
    ("Alt News (India)", "https://www.altnews.in"),
    ("BOOM (India)", "https://www.boomlive.in"),
    ("Snopes", "https://www.snopes.com"),
    ("FactCheck.org", "https://www.factcheck.org"),
    ("Reuters Fact Check", "https://www.reuters.com/fact-check"),
    ("Google Fact Check Explorer", "https://toolbox.google.com/factcheck/explorer"),
]

QUIZ = [
    ("SHOCKING: Doctors HATE this 1 weird trick that cures everything!!!", "Suspicious",
     "Clickbait wording, ALL CAPS, exclamation marks and a miracle claim are classic red flags."),
    ("Health ministry publishes monthly dengue report; cases down 8% from last month, officials said.", "Likely reliable",
     "Named institution, specific figure and attribution. (Still verify on the official website!)"),
    ("Share before it gets deleted: government secretly cancelling all exams tomorrow.", "Suspicious",
     "Urgency + 'share before it is deleted' + no source. Check the official notice first."),
    ("Researchers publish peer-reviewed coral reef study; authors note it covers only two sites.", "Likely reliable",
     "Credible source type and the authors openly mention limitations."),
    ("WhatsApp forward: drinking hot water every 30 minutes kills the virus - forward to 10 people.", "Suspicious",
     "Forward-chain health claim with no evidence. Verify with WHO / health ministry / fact-checkers."),
]


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model (first run trains it, please wait)...")
def get_model():
    return predictor.load_model()


def get_metrics():
    return predictor.load_metrics()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home():
    st.title("📰 Fake News Awareness & Detection System")
    st.write(
        "Paste a news article, headline or WhatsApp forward and get an AI-based "
        "**Real / Potentially Fake** prediction - plus the red flags and verification tips "
        "that help you judge the news yourself."
    )
    m = get_metrics()
    stats = db.prediction_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Articles in dataset", f"{m['n_total']:,}")
    c2.metric("Model accuracy (test set)", f"{m['accuracy']:.1%}")
    c3.metric("Checks made so far", stats["total"])

    st.subheader("How it works")
    st.markdown(
        "1. **Data collection** - local labelled dataset (real vs fake)\n"
        "2. **Pre-processing / NLP** - lowercasing, removing noise and stop words\n"
        "3. **TF-IDF** - converts text to weighted word features (1-2 word phrases)\n"
        "4. **Logistic Regression** - predicts the probability of the news being fake\n"
        "5. **Result + awareness** - prediction, influential words, red flags and verification guide\n"
        "6. **SQLite** - stores records and every prediction for the dashboard"
    )
    if "synthetic" in m["dataset_source"].lower():
        st.warning(
            "The app is currently trained on a small **synthetic starter dataset**, so accuracy looks "
            "unrealistically high. Add a real dataset (see README) and retrain from the "
            "*Model Performance* page for meaningful results."
        )
    st.info(
        "⚠️ This tool gives a statistical estimate based on writing patterns. It does **not** verify facts. "
        "Always cross-check important news with trusted fact-checkers."
    )


def _load_example(text):
    st.session_state["news_input"] = text


def _give_feedback(pred_id, value):
    db.set_feedback(pred_id, value)
    st.session_state["feedback_given"] = pred_id


def page_detect():
    st.title("🔍 Check a News Article")
    model = get_model()

    b1, b2, _ = st.columns([1, 1, 2])
    b1.button("Try fake-style example", on_click=_load_example, args=(FAKE_EXAMPLE,))
    b2.button("Try real-style example", on_click=_load_example, args=(REAL_EXAMPLE,))

    st.text_area("Paste the news text here (headline + body works best)", height=220, key="news_input")

    if st.button("Analyze", type="primary"):
        text = st.session_state.get("news_input", "").strip()
        if len(text.split()) < 15:
            st.warning("Please enter at least ~15 words so the model has enough context.")
        else:
            res = predictor.predict(model, text)
            res["signals"] = style_signals(text)
            res["id"] = db.save_prediction(text, res["label"], res["fake_prob"])
            st.session_state["last_result"] = res
            st.session_state.pop("feedback_given", None)

    res = st.session_state.get("last_result")
    if not res:
        return

    st.divider()
    pct = f"{res['confidence']:.0%}"
    if res["label"] == "Potentially Fake":
        st.error(f"🚩 Potentially Fake  -  {pct} confidence")
    elif res["label"] == "Likely Real":
        st.success(f"✅ Likely Real  -  {pct} confidence")
    else:
        st.warning("🤔 Uncertain - the model does not have enough evidence. Verify manually.")
    st.progress(min(max(res["fake_prob"], 0.0), 1.0), text=f"Fake probability: {res['fake_prob']:.0%}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Words pushing towards FAKE**")
        if res["towards_fake"]:
            st.dataframe(pd.DataFrame(res["towards_fake"], columns=["word / phrase", "influence"]),
                         hide_index=True, use_container_width=True)
        else:
            st.caption("None found.")
    with right:
        st.markdown("**Words pushing towards REAL**")
        if res["towards_real"]:
            st.dataframe(pd.DataFrame(res["towards_real"], columns=["word / phrase", "influence"]),
                         hide_index=True, use_container_width=True)
        else:
            st.caption("None found.")

    st.markdown("**🚩 Red flags in the writing style**")
    if res["signals"]:
        for s in res["signals"]:
            st.markdown(f"- {s}")
    else:
        st.caption("No obvious sensational-writing red flags found. That does not prove the news is true.")

    with st.expander("What should I do next? (verification checklist)"):
        st.markdown(
            "- Search the headline on a trusted news site or a fact-checker.\n"
            "- Check who published it and when. Is the date recent and the source named?\n"
            "- Look for the original source, official statement or data.\n"
            "- Reverse-search any image before believing it.\n"
            "- Do not forward it until you have verified it."
        )

    if st.session_state.get("feedback_given") == res["id"]:
        st.success("Thanks for your feedback!")
    else:
        st.caption("Was this prediction correct?")
        f1, f2, _ = st.columns([1, 1, 4])
        f1.button("👍 Yes", key="fb_yes", on_click=_give_feedback, args=(res["id"], "correct"))
        f2.button("👎 No", key="fb_no", on_click=_give_feedback, args=(res["id"], "incorrect"))


def page_dashboard():
    st.title("📊 Awareness Dashboard")
    stats = db.prediction_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total checks", stats["total"])
    c2.metric("Potentially fake", stats["fake"])
    c3.metric("Likely real", stats["real"])
    c4.metric("Uncertain", stats["uncertain"])

    df = db.get_predictions(5000)
    if df.empty:
        st.info("No checks yet - analyze some news on the *Check News* page and come back.")
    else:
        st.subheader("Results of all checks")
        st.bar_chart(df["label"].value_counts())
        df["day"] = pd.to_datetime(df["created_at"]).dt.date
        daily = df.pivot_table(index="day", columns="label", values="id", aggfunc="count", fill_value=0)
        st.subheader("Checks per day")
        st.bar_chart(daily)

    st.subheader("Training dataset balance")
    counts = db.news_label_counts()
    if not counts.empty:
        st.bar_chart(counts.set_index("label"))

    m = get_metrics()
    st.subheader("Words most associated with each class")
    a, b = st.columns(2)
    a.markdown("**Fake news**")
    a.bar_chart(pd.DataFrame(m["top_fake_words"], columns=["word", "weight"]).set_index("word"))
    b.markdown("**Real news**")
    b.bar_chart(pd.DataFrame(m["top_real_words"], columns=["word", "weight"]).set_index("word").abs())


def page_awareness():
    st.title("🎓 Fake News Awareness Guide")
    t1, t2, t3, t4 = st.tabs(["🚩 Red flags", "✅ How to verify", "🔗 Fact-check sites", "🧠 Quiz"])

    with t1:
        st.markdown(
            "**Common signs of fake or misleading news**\n"
            "- Sensational headline, ALL CAPS, many exclamation marks\n"
            "- 'Share before it gets deleted' / 'forward to 10 people' pressure\n"
            "- No author, no date, no named source ('insiders say', 'a source revealed')\n"
            "- Claims of a miracle cure or a secret that 'they don't want you to know'\n"
            "- Emotional content designed to make you angry or scared\n"
            "- Unknown website, odd URL, spelling and grammar mistakes\n"
            "- Old photo or video reused for a new event"
        )
    with t2:
        st.markdown(
            "**5-step verification routine**\n"
            "1. **Stop** - do not react or forward immediately.\n"
            "2. **Source** - who published it? Is it a known, accountable outlet?\n"
            "3. **Search** - do other trusted outlets report the same thing?\n"
            "4. **Check media** - reverse image search (Google Lens / TinEye), check video context.\n"
            "5. **Fact-check** - look the claim up on a fact-checking site below."
        )
    with t3:
        for name, url in FACT_CHECK_SITES:
            st.markdown(f"- [{name}]({url})")
    with t4:
        options = ["Likely reliable", "Suspicious"]
        with st.form("quiz_form"):
            answers = [st.radio(q[0], options, index=None, key=f"quiz_{i}") for i, q in enumerate(QUIZ)]
            submitted = st.form_submit_button("Check my answers")
        if submitted:
            score = 0
            for (headline, correct, why), ans in zip(QUIZ, answers):
                ok = ans == correct
                score += ok
                (st.success if ok else st.error)(f"{'Correct' if ok else 'Not quite'} - {correct}. {why}")
            st.subheader(f"Your score: {score}/{len(QUIZ)}")


def page_performance():
    st.title("📈 Model Performance")
    m = get_metrics()
    st.caption(f"Dataset: {m['dataset_source']} | Trained: {m['trained_at']} | "
               f"Train rows: {m['n_train']:,} | Test rows: {m['n_test']:,}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']:.1%}")
    c2.metric("Precision", f"{m['precision']:.1%}")
    c3.metric("Recall", f"{m['recall']:.1%}")
    c4.metric("F1-score", f"{m['f1']:.1%}")

    fig, ax = plt.subplots(figsize=(4, 3.4))
    cm = m["confusion_matrix"]
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Real", "Fake"])
    ax.set_yticks([0, 1], ["Real", "Fake"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > max(map(max, cm)) / 2 else "black")
    ax.set_title("Confusion matrix (test set)")
    st.pyplot(fig)

    st.markdown(
        "Precision = of the news flagged fake, how much really was fake. "
        "Recall = of all the fake news, how much was caught."
    )
    if st.button("🔄 Retrain model from data/ folder"):
        with st.spinner("Training..."):
            train_model.train()
        st.cache_resource.clear()
        st.rerun()


def page_history():
    st.title("🗂️ Prediction History")
    df = db.get_predictions(500)
    if df.empty:
        st.info("No predictions stored yet.")
        return
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.download_button("Download as CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="prediction_history.csv", mime="text/csv")
    fb = df["user_feedback"].value_counts()
    if not fb.empty:
        st.caption("User feedback: " + ", ".join(f"{k}: {v}" for k, v in fb.items()))


PAGES = {
    "🏠 Home": page_home,
    "🔍 Check News": page_detect,
    "📊 Dashboard": page_dashboard,
    "🎓 Awareness Guide": page_awareness,
    "📈 Model Performance": page_performance,
    "🗂️ History": page_history,
}


def main():
    st.set_page_config(page_title="Fake News Awareness System", page_icon="📰", layout="wide")
    db.init_db()
    st.sidebar.title("Fake News Awareness")
    choice = st.sidebar.radio("Go to", list(PAGES))
    PAGES[choice]()
    st.sidebar.caption("Mini Project 2026-27 | Dept. of Computer Applications")


if __name__ == "__main__":
    main()
