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
| ML pipeline | scikit-learn 1.6.1 (ColumnTransformer + Pipeline) |
| Serialization | joblib 1.5.3 |
| Data handling | pandas 3.0.1, numpy 2.4.2 |
| Class balancing | imbalanced-learn / SMOTE (notebook only) |
| Frontend | Jinja2 templates, custom CSS, Chart.js, Font Awesome |
| Currency API | er-api.com (live exchange rates on index page) |

---

## Repository Structure

```
GoCreadi/                              ← repo root
├── CLAUDE.md                          ← this file
├── README.md                          ← GitHub portfolio page (to be written)
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
│       ├── services/
│       │   └── prediction_service.py  ← load_pipeline() and predict_credit()
│       ├── static/
│       │   ├── css/style.css          ← all styling (636 lines, do not rewrite)
│       │   └── img/                   ← logos, favicon, 10 country flag SVGs
│       └── templates/
│           ├── base.html              ← Jinja2 base template
│           ├── index.html             ← landing page with currency ticker
│           ├── form.html              ← prediction input form
│           ├── result.html            ← prediction result with Chart.js doughnut
│           ├── about.html             ← model explanation page
│           ├── aboutus.html           ← team page
│           ├── contact.html           ← contact form
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
    └── ML/                            ← 8 old notebooks, 7 experiment pkl dirs, old csvs
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

**Available Python on this machine:** Python 3.13 (system) and Python 3.10 — neither works.
**Python 3.11 must be installed separately** before the app can run.

### Venv status: broken

The existing `credit_card_app/venv/` was created by a different user account
(`C:\Users\ome20\...`). That path no longer exists on this machine. The venv's
`python.exe` is dead. Do not use it.

### To recreate the venv correctly:

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

If Python 3.11 is not available, the pipeline must be retrained with the current
environment (Python 3.13 + scikit-learn 1.8.0) after the notebook cleanup is done.
See the ML Notebook Rules section. After retraining, update `requirements.txt` with:
```bash
pip freeze > credit_card_app/requirements.txt
```

---

## Running the App

The app can now be run from **any directory** — path resolution uses `__file__` in
`app/__init__.py` to locate `model_artifacts/pipeline.pkl` and `logs/` as absolute
paths relative to the source files, not the working directory.

```bash
# From anywhere, as long as the venv is active:
python credit_card_app/run.py
# → http://127.0.0.1:5000
```

---

## Flask App Rules

### Architecture
- The app uses the **application factory pattern** — `create_app()` in `app/__init__.py`
- Routes are registered as a **Blueprint** named `main` — do not switch to a flat `register_routes(app)` pattern
- The ML pipeline is loaded **once at startup** into `app.config["PIPELINE"]`
- Never load the pipeline inside a route function — it would reload on every request

### What lives where
- All prediction logic: `app/services/prediction_service.py` — keep it there
- All route handlers: `app/routes.py` — keep them there
- Config values: `config.py` → `Config` class
- Model path: `"model_artifacts/pipeline.pkl"` in `app/__init__.py` — do not change without updating this file

### `predict_credit()` contract
The function signature is:
```python
predict_credit(form_data, pipeline) -> dict
```
It returns a dict with **exactly these keys**:
```python
{
    "prediction": int,       # 0 = rejected, 1 = approved
    "acceptance": float,     # e.g. 87.3 (percentage)
    "rejection": float       # e.g. 12.7 (percentage)
}
```
`result.html` depends on these exact key names. Do not rename them.

### Input features — exact mapping (pipeline depends on this)
| HTML form field | DataFrame column | Expected type | Example |
|---|---|---|---|
| `gender` | `CODE_GENDER` | str | `"M"` or `"F"` |
| `own_car` | `FLAG_OWN_CAR` | str | `"Y"` or `"N"` |
| `education` | `NAME_EDUCATION_TYPE` | str (categorical) | `"Higher education"` |
| `income` | `AMT_INCOME_TOTAL` | float | `120000.0` |
| `age` | `AGE` | int | `30` |
| `years_employed` | `YEARS_EMPLOYED` | int | `5` |
| `family_members` | `CNT_FAM_MEMBERS` | int | `3` |

The column names and types must match exactly what the pipeline's ColumnTransformer
was trained on. Changing any name will cause a silent prediction failure or crash.

### Templates
- All templates extend `base.html` via Jinja2 inheritance (`{% extends "base.html" %}`)
- `result.html` expects: `prediction` (int), `acceptance` (float), `rejection` (float)
- `error.html` expects: `message` (str)
- Do not add inline styles to templates — all styling belongs in `static/css/style.css`
- Do not restructure the `partials/` folder — `navbar.html` and `ticker.html` are included by `base.html`

### Dependencies
- `Flask==3.1.3` — do not downgrade (3.x has breaking API changes from 2.x)
- `scikit-learn==1.6.1` — **must match the version used when `pipeline.pkl` was saved**. A version mismatch will raise a deserialization error when loading the model.
- `joblib==1.5.3` — same constraint as scikit-learn
- If you retrain and save a new pipeline, update `requirements.txt` to match your current environment versions

---

## ML Notebook Rules

**Active notebook:** `notebooks/credit_approval_model.ipynb`

**Do not run any notebook in `archive/`** — they contain data leakage bugs and
broken cell ordering. They are kept for reference only.

### Known issues that must be fixed before retraining

These are pre-existing problems in the current notebook, carried over from the
experimental version. Fix them before any retraining session:

**1. Scrambled experimental section (approximately cells 52–63)**
These cells are leftover experiments that were never cleaned up:
- `X = df.iloc[:, :-1]` re-assigns `X` from the SMOTE-balanced dataframe (wrong — overwrites the correctly-defined 7-feature X)
- `pd.get_dummies` is called on already-encoded data (double encoding)
- `LabelEncoder.fit_transform` is called on `X_test` directly (data leakage — must only `.transform` test data, never `.fit_transform`)
- A second `train_test_split` is run on the balanced dataframe (test set contains synthetic SMOTE samples, inflating all metrics)
- **Action:** Delete these cells entirely before running any training.

**2. Preprocessor is fit twice (cells ~37 and ~39)**
- Cell ~37: `preprocessor.fit(X_train)` — saves preprocessor to disk at this point
- Cell ~39: `preprocessor.fit_transform(X_train_clean)` — refits on the cleaner outlier-removed data
- The saved `preprocessor.pkl` (from cell 37) is fit on different data than what the model was trained on.
- **Action:** Remove cell ~37's `.fit()` call. Keep only the `fit_transform` in cell ~39.

**3. Redundant standalone scaler**
- A `StandardScaler` is defined and fit separately after the `ColumnTransformer`.
- This is not needed — `StandardScaler` is already inside the `ColumnTransformer` for numerical columns.
- The saved `pipeline.pkl` (cell ~96) incorrectly chains: preprocessor → scaler → model. The scaler step was fit on transformed+SMOTE data, not raw input. This makes the pipeline non-functional for real input.
- **Action:** Remove the standalone `scaler` variable and its `.fit_transform` call. Do not include it in the final Pipeline.

**4. Inflated accuracy metrics (~98% RF, ~95% KNN)**
- These scores come from evaluating on a test set that was split from the SMOTE-balanced dataframe (i.e., the test set contains synthetic samples).
- Real-world accuracy on unseen raw data will be lower (expect ~79–93% depending on model).
- **Action:** Always evaluate on `X_test_transformed` — the hold-out from the original first `train_test_split` (before SMOTE), transformed by the preprocessor only.

### Correct pipeline construction for deployment

This is the target structure every retraining session must produce:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier  # or RandomForestClassifier

numerical_cols = ['AMT_INCOME_TOTAL', 'AGE', 'YEARS_EMPLOYED', 'CNT_FAM_MEMBERS']
categorical_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_EDUCATION_TYPE']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first'), categorical_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', knn)           # classifier trained on preprocessed + resampled data
])

# Save to exactly this path — this is what the Flask app loads
import joblib, os
os.makedirs("model_artifacts", exist_ok=True)
joblib.dump(pipeline, "model_artifacts/pipeline.pkl")
```

**No separate scaler. No separate preprocessor file. One pipeline file only.**

### SMOTE placement rule

SMOTE must only be applied to training data, never before the train-test split:

```python
# Step 1: split first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: remove outliers from training only
X_train_clean = remove_outliers_iqr(X_train.copy(), numerical_cols)
y_train_clean = y_train.loc[X_train_clean.index]

# Step 3: preprocess training data
X_train_transformed = preprocessor.fit_transform(X_train_clean)
X_test_transformed = preprocessor.transform(X_test)   # transform only, never fit

# Step 4: SMOTE on transformed training data only
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train_clean)

# Step 5: train classifier on resampled data
knn.fit(X_train_resampled, y_train_resampled)

# Step 6: evaluate on ORIGINAL test set (no SMOTE)
y_pred = knn.predict(X_test_transformed)
```

---

## Known Bugs (fix before next release)

| # | Bug | File | Line | Fix |
|---|---|---|---|---|
| 1 | `/contact` POST renders `thank_you.html` which does not exist → `TemplateNotFound` crash on every contact form submission | `routes.py` | 38 | Create `templates/thank_you.html` or replace with a redirect |
| 2 | No server-side input validation on `/predict` | `routes.py` | `/predict` route | Add `app/validators.py` from the `GoCreadi-Project Final` reference in `archive/` |
| 3 | 404 and 500 error handlers return plain strings instead of rendered templates | `app/__init__.py` | 32–40 | Return `render_template("error.html", message=...)` |
| 4 | ~~`model_artifacts/` path was relative~~ | `app/__init__.py` | — | **Fixed** — now uses `__file__`-relative absolute path |
| 5 | ~~`js/` folder was empty~~ | `static/js/` | — | **Fixed** — `.gitkeep` added |
| 6 | ~~`logs/` missing after clone~~ | `logs/` | — | **Fixed** — `.gitkeep` added, `.gitignore` uses `*.log` |

---

## Approved Merge Plan (not yet applied)

From `archive/ML/GoCreadi-Project Final` → `credit_card_app`:

| Step | Action | Risk |
|---|---|---|
| 1 | Create `app/validators.py` — server-side field validation | None — new file |
| 2 | Edit `app/routes.py` — add `validate_form()` call before prediction in `/predict` | Low |
| 3 | Create `app/templates/thank_you.html` — fix contact route crash | None — new file |
| 4 | (Optional) Extract `app/model_loader.py` — fix path if done | Low |

Apply these changes **after** notebook cleanup and model retraining are complete.

---

## Development Workflow Rules

These rules apply to all code changes, human or AI-assisted:

1. **Incremental changes only.** One logical change per session. Do not refactor and add features at the same time.
2. **Read before editing.** Never suggest or apply changes to a file without reading its current content first.
3. **Test the app after every change.** Run `python run.py` from the repo root and verify the home page loads before committing.
4. **Do not restructure the Flask app.** The Blueprint pattern, folder layout, and template inheritance are intentional. Do not flatten them.
5. **Do not change column names.** The pipeline was serialized with specific feature names. Changing any of the 7 column names in `prediction_service.py` will break model loading silently.
6. **Do not add `random_state` to production code.** `random_state` belongs only in the notebook. The Flask app calls `pipeline.predict()` — no randomness involved.
7. **Do not unpin dependencies** in `requirements.txt`. The pickle-based model is version-sensitive. Unpinning can break deserialization on any new environment.
8. **Notebook changes do not auto-update the app.** After retraining, you must manually copy the new `pipeline.pkl` to `model_artifacts/` and restart the Flask app.
9. **Never commit to `main` directly for non-trivial changes.** Use feature branches.
10. **Archive is read-only.** Files in `archive/` are for reference only. Do not run, edit, or import from them.

---

## Strict Do-Not-Break Rules

These things work right now. Do not touch them unless the task explicitly requires it:

- `app/__init__.py` — pipeline loading, logging setup, blueprint registration
- `app/services/prediction_service.py` — the 7-feature DataFrame construction
- `model_artifacts/pipeline.pkl` — the serialized model (do not overwrite until a new trained model is verified)
- `app/templates/result.html` — Chart.js doughnut chart depends on `acceptance` and `rejection` variables
- `app/templates/base.html` — all other templates inherit from this
- `app/static/css/style.css` — 636 lines, do not rewrite
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
- `feature/notebook-cleanup` — fixing the notebook before retraining
- `feature/validators` — adding server-side input validation
- `feature/model-retrain` — new pipeline after notebook is fixed
