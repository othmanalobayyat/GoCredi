# GoCredi — Credit Card Approval Prediction

A machine learning web application that predicts credit card approval probability
based on applicant financial data. Built as a graduation project at Palestine Ahliya University.

---

## Overview

GoCredi takes ten applicant inputs and returns an approval decision with a probability
breakdown, a risk tier (Low / Medium / High), and the top three most influential model
factors. The backend is a scikit-learn Random Forest pipeline served through a Flask
web application, with both a web UI and a JSON API endpoint.

**Authors:** Othman Muhammad Al-Obayyat · Rahaf Ihsan Adeelah · Saja Shehada

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Flask 3.1.3 |
| ML pipeline | scikit-learn 1.6.1, imbalanced-learn 0.12.4 |
| Data handling | pandas 3.0.1, numpy 2.4.2 |
| Serialization | joblib 1.5.3 |
| Frontend | Jinja2, custom CSS, Chart.js, Font Awesome |
| Currency ticker | er-api.com (live exchange rates) |

---

## Project Structure

```
GoCreadi/
├── credit_card_app/          # Flask application
│   ├── app/
│   │   ├── __init__.py       # App factory, pipeline loading, logging, error handlers
│   │   ├── routes.py         # URL routes (web + JSON API)
│   │   ├── validators.py     # Server-side input validation
│   │   ├── services/
│   │   │   └── prediction_service.py  # predict_credit(), feature importance
│   │   ├── templates/        # Jinja2 HTML templates
│   │   └── static/           # CSS, images, flag SVGs
│   ├── config.py
│   ├── requirements.txt
│   └── run.py
├── model_artifacts/
│   └── pipeline.pkl          # Trained scikit-learn pipeline (6.4 MB)
├── notebooks/
│   └── credit_approval_model.ipynb  # Full ML training notebook
└── data/                     # Raw CSV files (not committed — ~67 MB)
```

---

## Setup

> **Python version required: 3.11**
> The trained `pipeline.pkl` was serialized with scikit-learn 1.6.1 on Python 3.11.
> Loading it with a different scikit-learn version will raise a deserialization error.
> If Python 3.11 is unavailable, retrain the model using the notebook and update
> `requirements.txt` with your current environment's pinned versions.

```bash
# 1. Create a virtual environment with Python 3.11
py -3.11 -m venv credit_card_app/venv

# 2. Activate it
# Windows:
credit_card_app\venv\Scripts\activate
# macOS/Linux:
source credit_card_app/venv/bin/activate

# 3. Install dependencies
pip install -r credit_card_app/requirements.txt

# 4. Run the app (from any directory)
python credit_card_app/run.py
```

The app will be available at `http://127.0.0.1:5000`

---

## Input Features

The model uses ten features (Set B):

| Field | Description | Type | Values |
|---|---|---|---|
| Gender | Applicant gender | Categorical | M / F |
| Own Car | Car ownership | Categorical | Y / N |
| Own Property | Property ownership | Categorical | Y / N |
| Education | Education level | Categorical | Higher education, Secondary, etc. |
| Income Type | Employment category | Categorical | Working, Pensioner, Student, etc. |
| Family Status | Marital status | Categorical | Married, Single, etc. |
| Annual Income | Total yearly income | Numeric | > 0 |
| Age | Applicant age | Numeric | 18–100 |
| Years Employed | Employment duration | Numeric | 0–60 |
| Family Members | Household size | Numeric | 1–20 |

---

## Result Output

Each prediction returns:

- **Decision** — Approved or Not Approved
- **Approval probability** — displayed as a percentage and doughnut chart
- **Risk tier** — Low (≥70%), Medium (40–69%), High (<40%)
- **Top 3 model factors** — the features with the highest impact on the decision

---

## API

### Health check

```
GET /health
→ {"status": "ok"}
```

### Predict (JSON)

```
POST /api/predict
Content-Type: application/json

{
  "gender": "M",
  "own_car": "Y",
  "own_realty": "N",
  "education": "Higher education",
  "income_type": "Working",
  "family_status": "Married",
  "income": "150000",
  "age": "35",
  "years_employed": "8",
  "family_members": "3"
}
```

Response:

```json
{
  "prediction": 1,
  "risk_level": "Low",
  "acceptance": 84.2,
  "rejection": 15.8,
  "top_features": ["Annual Income", "Years Employed", "Age"]
}
```

- `prediction`: `1` = approved, `0` = not approved
- `acceptance` / `rejection`: probabilities summing to 100
- `top_features`: global model feature importances (Random Forest), not per-prediction SHAP values

---

## ML Notebook

The notebook `notebooks/credit_approval_model.ipynb` contains the full training pipeline:

- Data loading and merging from two CSV sources
- Feature engineering (AGE, YEARS_EMPLOYED converted from raw day counts)
- Outlier removal (IQR method, training set only)
- Preprocessing via ColumnTransformer (StandardScaler + OneHotEncoder with `handle_unknown='ignore'`)
- Class balancing via SMOTE inside an ImbPipeline (applied only during training folds)
- Feature selection experiment comparing 7, 10, and 14 feature sets
- Model comparison (Random Forest vs KNN)
- 5-fold StratifiedKFold cross-validation
- RandomizedSearchCV hyperparameter tuning (n_iter=10)
- Final pipeline export to `model_artifacts/pipeline.pkl`

---

## Data

Raw data files are not committed to this repository (combined ~67 MB).

| File | Description | Source |
|---|---|---|
| `application_record.csv` | 438,557 applicant records | Kaggle |
| `credit_record.csv` | 1,048,575 monthly credit status records | Kaggle |

Place both files in the `data/` directory before running the notebook.

---

## License

University graduation project — Palestine Ahliya University, 2025.
