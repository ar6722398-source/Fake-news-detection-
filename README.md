# Fake News Awareness & Detection System

Mini Project 2026-27 | Department of Computer Applications
Python + Streamlit + Scikit-learn (TF-IDF + Logistic Regression) + SQLite

## Run it (3 commands)
```bash
pip install -r requirements.txt
streamlit run app.py
```
On the first run the model trains automatically (takes a few seconds on the bundled data).
Optional manual training: `python train_model.py`

## IMPORTANT: use a real dataset
The bundled `data/news_dataset.csv` is a small **synthetic** starter set (made by
`generate_sample_dataset.py`), so it shows ~100% accuracy - that is NOT a real result.

For your final submission:
1. Download the Kaggle "Fake and Real News Dataset" (ISOT) -> `Fake.csv` and `True.csv`.
2. Put both files inside the `data/` folder.
3. Run `python train_model.py` (or press *Retrain* on the Model Performance page).
Expect roughly 95-99% on that dataset, and report the real numbers in your documentation.
Any custom CSV also works if it has `title`, `text`, `label` (REAL/FAKE) columns.

## Project structure
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: Home, Check News, Dashboard, Awareness Guide, Model Performance, History |
| `preprocess.py` | NLP cleaning + rule-based red-flag detector |
| `train_model.py` | Loads data, trains TF-IDF + Logistic Regression, evaluates, saves model + metrics |
| `predictor.py` | Loads model, predicts, explains influential words |
| `database.py` | SQLite tables: `news_records`, `predictions` (+ user feedback) |
| `generate_sample_dataset.py` | Creates the synthetic starter dataset |
| `data/`, `models/` | Dataset, SQLite DB, saved model and metrics |

## Mapping to the proposal deliverables
- Trained ML model and local dataset -> `models/`, `data/`
- SQLite database for records and predictions -> `data/fake_news.db`
- User-friendly web interface -> `app.py`
- Fake news awareness dashboard -> *Dashboard* + *Awareness Guide* pages
- Model testing and performance evaluation -> *Model Performance* page (accuracy, precision, recall, F1, confusion matrix)
- Awareness posters/infographics (Canva) and final report -> to be made by the team

## Limitations (mention these in your report/viva)
- The model learns writing style and word patterns, it does not verify facts.
- It can be wrong on short text, satire, or topics unlike the training data.
- Outputs: *Potentially Fake* (>=60% fake probability), *Likely Real* (<=40%), otherwise *Uncertain*.
