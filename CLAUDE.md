# CLAUDE.md — GoCredi Development Guide

> This file is the single source of truth for AI-assisted development on this project.
> Read it fully before making any code changes. Update it when rules change.

---

## Project Overview

**GoCredi** is a credit card approval prediction web application, built as a university
graduation project at Palestine Ahliya University. It predicts whether a credit card
application will be approved or rejected using a trained machine learning pipeline,
served through a Flask web application.

**Authors:** Othman Muhammad Al-Obayyat, Rahaf Ihsan Adeelah, Saja Shehada

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Flask 3.1.3 (application factory + Blueprints) |
| ML pipeline | scikit-learn 1.6.1 (ImbPipeline: preprocessor → SMOTE → RF) |
| Class balancing | imbalanced-learn 0.12.4 / SMOTE |
| Serialization | joblib 1.5.3 |
| Data handling | pandas 3.0.1, numpy 2.4.2 |
| Frontend | Jinja2 templates, custom CSS, Chart.js, Font Awesome |
| Currency API | er-api.com (live exchange rates on index page) |

---

## Repository Structure

```
GoCreadi/                              ← repo root
├── CLAUDE.md                          ← this file
├── README.md                          ← GitHub portfolio page
├── .gitignore
│
├── credit_card_app/                   ← Flask application (canonical, do not restructure)
│   ├── run.py                         ← entry point
│   ├── config.py                      ← Config class with SECRET_KEY
│   ├── requirements.txt               ← pinned dependencies
│   ├── venv/                          ← local only, gitignored
│   ├── logs/                          ← local only, gitignored
│   │   └── app.log
│   └── app/
│       ├── __init__.py                ← app factory: loads pipeline, logging, error handlers
│       ├── routes.py                  ← all routes via Blueprint named `main`
│       ├── validators.py              ← server-side input validation (validate_form)
│       ├── services/
│       │   └── prediction_service.py  ← load_pipeline(), predict_credit(), _get_top_features()
│       ├── static/
│       │   ├── css/style.css          ← all styling (~688 lines, do not rewrite)
│       │   └── img/                   ← logos, favicon, 10 country flag SVGs
│       └── templates/
│           ├── base.html              ← Jinja2 base template
│           ├── index.html             ← landing page with currency ticker
│           ├── form.html              ← prediction input form (10 fields, Set B)
│           ├── result.html            ← prediction result with Chart.js doughnut + risk + top features
│           ├── about.html             ← model explanation page
│           ├── aboutus.html           ← team page
│           ├── contact.html           ← contact form (JS fetch, no page reload)
│           ├── error.html             ← error display page
│           └── partials/
│               ├── navbar.html        ← reusable navigation component
│               └── ticker.html        ← live currency exchange rate ticker
│
├── model_artifacts/                   ← production model (committed to git)
│   └── pipeline.pkl                   ← 6.4MB — the model the Flask app loads
│
├── notebooks/
│   └── credit_approval_model.ipynb    ← main ML notebook (single source of truth)
│
├── data/                              ← local only, gitignored (too large)
│   ├── application_record.csv         ← 52MB raw application data
│   └── credit_record.csv              ← 15MB credit history data
│
└── archive/                           ← local only, gitignored, never commit
    └── ML/                            ← old notebooks, experiment pkl dirs, old csvs
```

---

## Environment Requirements

> **This is the most important setup constraint. Read before doing anything.**

### Python version: 3.11 required

`model_artifacts/pipeline.pkl` was serialized with **scikit-learn 1.6.1 on Python 3.11**.
Loading it with any other scikit-learn version raises:
```
Can't get attribute '_RemainderColsList' on sklearn.compose._column_transformer
```

### To set up the venv:

```bash
# 1. Install Python 3.11 from python.org (add to PATH)

# 2. From repo root (GoCreadi/):
py -3.11 -m venv credit_card_app/venv

# 3. Activate
credit_card_app\venv\Scripts\activate

# 4. Install pinned deps
pip install -r credit_card_app/requirements.txt

# 5. Run
python credit_card_app/run.py
# → http://127.0.0.1:5000
```

### Alternative: retrain with current Python

If Python 3.11 is not available, retrain the pipeline using the notebook with your
current environment, then update `requirements.txt`:
```bash
pip freeze > credit_card_app/requirements.txt
```

---

## Running the App

Path resolution uses `__file__` in `app/__init__.py` — the app can be launched from
any directory.

```bash
python credit_card_app/run.py
# → http://127.0.0.1:5000
```

---

## Flask App Rules

### Architecture
- Application factory pattern — `create_app()` in `app/__init__.py`
- Routes registered as Blueprint named `main` — do not switch to flat `register_routes(app)`
- ML pipeline loaded **once at startup** into `app.config["PIPELINE"]`
- Never load the pipeline inside a route function

### What lives where
- Prediction logic: `app/services/prediction_service.py`
- Route handlers: `app/routes.py`
- Input validation: `app/validators.py`
- Config values: `config.py` → `Config` class

### Routes

| Route | Method | Returns |
|---|---|---|
| `/` | GET | index.html (landing page) |
| `/form` | GET | form.html (input form) |
| `/predict` | POST | result.html or error.html |
| `/about` | GET | about.html |
| `/aboutus` | GET | aboutus.html |
| `/contact` | GET | contact.html |
| `/contact` | POST | 200 empty (JS-handled) |
| `/health` | GET | `{"status": "ok"}` |
| `/api/predict` | POST | JSON prediction response |

### `predict_credit()` contract

```python
predict_credit(form_data, pipeline) -> dict
```

Returns:
```python
{
    "prediction": int,        # 0 = rejected, 1 = approved
    "acceptance": float,      # e.g. 87.3 (percentage)
    "rejection": float,       # e.g. 12.7 (percentage)
    "risk_level": str,        # "Low" | "Medium" | "High"
    "top_features": list[str] # top 3 feature labels, may be [] if RF not available
}
```

Risk thresholds: acceptance ≥ 70% → Low, ≥ 40% → Medium, < 40% → High.

`result.html` depends on all of these key names. Do not rename them.

### Input features — Set B (10 features, pipeline depends on this exactly)

| HTML form field | DataFrame column | Expected type | Example |
|---|---|---|---|
| `gender` | `CODE_GENDER` | str | `"M"` or `"F"` |
| `own_car` | `FLAG_OWN_CAR` | str | `"Y"` or `"N"` |
| `own_realty` | `FLAG_OWN_REALTY` | str | `"Y"` or `"N"` |
| `education` | `NAME_EDUCATION_TYPE` | str | `"Higher education"` |
| `income_type` | `NAME_INCOME_TYPE` | str | `"Working"` |
| `family_status` | `NAME_FAMILY_STATUS` | str | `"Married"` |
| `income` | `AMT_INCOME_TOTAL` | float | `120000.0` |
| `age` | `AGE` | int | `30` |
| `years_employed` | `YEARS_EMPLOYED` | int | `5` |
| `family_members` | `CNT_FAM_MEMBERS` | int | `3` |

Column names and types must match exactly what the pipeline's ColumnTransformer was
trained on. Changing any name will break the pipeline silently or raise a KeyError.

### Templates
- `result.html` expects: `prediction`, `acceptance`, `rejection`, `risk_level`, `top_features`
- `error.html` expects: `message` (str)
- Do not add inline styles to templates — all styling belongs in `static/css/style.css`
- Do not restructure the `partials/` folder

### Validation (`app/validators.py`)

`validate_form(form_data) -> list[str]` validates all 10 fields.
Returns an empty list if valid. Called by both `/predict` and `/api/predict`.

Numeric bounds: income > 0 and ≤ 10M, age 18–100, years_employed 0–60, family_members 1–20.

### Dependencies
- `Flask==3.1.3` — do not downgrade
- `scikit-learn==1.6.1` — must match the version used when `pipeline.pkl` was saved
- `imbalanced-learn==0.12.4` — required for notebook training (ImbPipeline + SMOTE)
- `joblib==1.5.3` — same constraint as scikit-learn
- If you retrain, update `requirements.txt` with `pip freeze`

---

## ML Notebook Rules

**Active notebook:** `notebooks/credit_approval_model.ipynb`

**Do not run any notebook in `archive/`** — they contain data leakage bugs and broken
cell ordering. Reference only.

### Notebook state (post-cleanup)

The notebook has been cleaned up. The following issues were fixed:
- Scrambled experimental cells (double encoding, data leakage, second train_test_split) — deleted
- Redundant standalone scaler — removed
- Preprocessor double-fit — fixed (only `fit_transform` on training data, `transform` on test)
- Data paths corrected to `"../data/..."` for running from `notebooks/` directory
- ImbPipeline imported from `imblearn.pipeline`

### Feature set: Set B (10 features)

Chosen over Set A (7 features) and Set C (14 features) after controlled comparison.
Set C's OCCUPATION_TYPE had 31% unknowns — excluded.

```python
numerical_cols = ['AMT_INCOME_TOTAL', 'AGE', 'YEARS_EMPLOYED', 'CNT_FAM_MEMBERS']
categorical_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                    'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS']
```

### Correct pipeline construction for deployment

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
])

rf_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Save to exactly this path — this is what the Flask app loads
import joblib, os
os.makedirs("../model_artifacts", exist_ok=True)
joblib.dump(rf_pipeline, "../model_artifacts/pipeline.pkl")
```

**ImbPipeline ensures SMOTE runs only during `fit()`, not during `predict()`.**
**No separate scaler. No separate preprocessor file. One pipeline file only.**

### Training workflow (correct order)

```python
# 1. Split first (before SMOTE, before outlier removal)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Remove outliers from training only
X_train_clean = remove_outliers_iqr(X_train.copy(), numerical_cols)
y_train_clean = y_train.loc[X_train_clean.index]

# 3. Fit the ImbPipeline on training data (SMOTE applied internally)
rf_pipeline.fit(X_train_clean, y_train_clean)

# 4. Evaluate on ORIGINAL test set (no SMOTE — ImbPipeline skips it at predict time)
y_pred = rf_pipeline.predict(X_test)
```

### Evaluation metrics (imbalanced classification)

Use these four metrics — accuracy alone is misleading on imbalanced data:
- ROC-AUC
- Macro F1
- Precision (class 0 = bad credit)
- Recall (class 0 = bad credit)

---

## Known Issues

| # | Issue | Status |
|---|---|---|
| 1 | `/contact` POST crashed with `TemplateNotFound: thank_you.html` | **Fixed** — returns `("", 200)`; JS handles success display |
| 2 | No server-side validation on `/predict` | **Fixed** — `app/validators.py` added |
| 3 | 404/500 handlers returned plain strings | **Fixed** — now render `error.html` |
| 4 | `model_artifacts/` path was relative to working directory | **Fixed** — uses `__file__`-relative absolute path |
| 5 | `logs/` missing after clone | **Fixed** — `.gitkeep` added, `.gitignore` uses `*.log` |

---

## Development Workflow Rules

1. **Incremental changes only.** One logical change per session.
2. **Read before editing.** Never modify a file without reading it first.
3. **Test after every change.** Run `python credit_card_app/run.py` and verify the home page loads.
4. **Do not restructure the Flask app.** Blueprint pattern and folder layout are intentional.
5. **Do not change column names.** The 10 Set B column names in `prediction_service.py` must match the trained pipeline exactly.
6. **Do not add `random_state` to production code.** It belongs only in the notebook.
7. **Do not unpin dependencies.** The pickle-based model is version-sensitive.
8. **Notebook changes do not auto-update the app.** After retraining, manually copy the new `pipeline.pkl` to `model_artifacts/` and restart.
9. **Never commit to `main` directly for non-trivial changes.** Use feature branches.
10. **Archive is read-only.** Do not run, edit, or import from `archive/`.

---

## Strict Do-Not-Break Rules

These are working and correct. Do not touch unless the task explicitly requires it:

- `app/__init__.py` — pipeline loading, logging setup, blueprint registration, error handlers
- `app/services/prediction_service.py` — the 10-feature DataFrame construction and feature importance
- `app/validators.py` — all 10 field validations
- `model_artifacts/pipeline.pkl` — do not overwrite until a new trained model is verified
- `app/templates/result.html` — Chart.js doughnut depends on `acceptance` and `rejection` variables
- `app/templates/base.html` — all other templates inherit from this
- `app/static/css/style.css` — do not rewrite
- `requirements.txt` — pinned versions are intentional

---

## Git Workflow

No git history exists yet. Initialize with:

```bash
# From GoCreadi/ (repo root):
git init
git add .
git status               # verify pipeline.pkl IS staged, data/ is NOT staged
git commit -m "Initial commit: GoCredi Flask app with trained pipeline"
```

**Before pushing to GitHub, verify:**
- `git ls-files model_artifacts/pipeline.pkl` returns the file (not empty)
- `git ls-files data/` returns nothing (data is gitignored)
- `git ls-files archive/` returns nothing (archive is gitignored)
- `git ls-files credit_card_app/venv/` returns nothing (venv is gitignored)

**Branch strategy:**
- `main` — always runnable, always clean
- `feature/model-retrain` — new pipeline after notebook changes
