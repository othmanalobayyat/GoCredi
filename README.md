# GoCredi — Credit Card Approval Prediction

A machine learning web application that predicts credit card approval probability
based on applicant financial data. Built as a graduation project at Palestine Ahliya University.

---

## Overview

GoCredi takes seven applicant inputs and returns an approval decision with a
probability breakdown, displayed as a doughnut chart. The backend is a scikit-learn
pipeline (preprocessing + KNN classifier) served through a Flask web application.

**Authors:** Othman Muhammad Al-Obayyat · Rahaf Ihsan Adeelah · Saja Shehada

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Flask 3.1.3 |
| ML pipeline | scikit-learn 1.6.1 |
| Data handling | pandas 3.0.1, numpy 2.4.2 |
| Serialization | joblib 1.5.3 |
| Frontend | Jinja2, custom CSS, Chart.js, Font Awesome |

---

## Project Structure

```
GoCredi/
├── credit_card_app/          # Flask application
│   ├── app/
│   │   ├── __init__.py       # App factory
│   │   ├── routes.py         # URL routes
│   │   ├── services/         # Prediction logic
│   │   ├── templates/        # Jinja2 HTML templates
│   │   └── static/           # CSS, images, flags
│   ├── config.py
│   ├── requirements.txt
│   └── run.py
├── model_artifacts/
│   └── pipeline.pkl          # Trained scikit-learn pipeline
├── notebooks/
│   └── credit_approval_model.ipynb  # ML training notebook
└── data/                     # Raw data (not committed — too large)
```

---

## Setup

> **Python version required: 3.11**
> The trained `pipeline.pkl` was serialized with scikit-learn 1.6.1 on Python 3.11.
> Loading it with a different Python or scikit-learn version will raise a
> deserialization error. See [Environment Notes](#environment-notes) below.

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

# 4. Run the app (from the repo root)
python credit_card_app/run.py
```

The app will be available at `http://127.0.0.1:5000`

---

## Input Features

The model uses seven features:

| Field | Description | Values |
|---|---|---|
| Gender | Applicant gender | M / F |
| Own Car | Car ownership | Y / N |
| Education | Education level | Higher education, Secondary, etc. |
| Annual Income | Total annual income | Numeric |
| Age | Applicant age in years | Numeric |
| Years Employed | Employment duration | Numeric |
| Family Members | Family size | Numeric |

---

## Environment Notes

### Known compatibility issue

The `model_artifacts/pipeline.pkl` was built with:
- Python 3.11
- scikit-learn 1.6.1
- joblib 1.5.3

If your environment has a different version of scikit-learn, loading the pipeline
will raise an error similar to:

```
Can't get attribute '_RemainderColsList' on sklearn.compose._column_transformer
```

**Resolution:** Use Python 3.11 and install the exact versions in `requirements.txt`.
The notebook `notebooks/credit_approval_model.ipynb` can be used to retrain and
export a new pipeline compatible with your current environment after cleanup.

---

## ML Notebook

The notebook `notebooks/credit_approval_model.ipynb` contains the full training
pipeline:
- Data loading and merging from two CSV sources
- Feature engineering (AGE, YEARS_EMPLOYED from raw day counts)
- Outlier removal (IQR method, training set only)
- Preprocessing via ColumnTransformer (StandardScaler + OneHotEncoder)
- Class balancing via SMOTE
- Model comparison (Random Forest, KNN)
- Pipeline export to `model_artifacts/pipeline.pkl`

> **Note:** The notebook requires cleanup before retraining. See `CLAUDE.md` for details.

---

## Data

Raw data files are not committed to this repository (combined ~67MB).

| File | Description | Source |
|---|---|---|
| `application_record.csv` | 438,557 applicant records | Kaggle |
| `credit_record.csv` | 1,048,575 monthly credit status records | Kaggle |

Place both files in the `data/` directory before running the notebook.

---

## License

University graduation project — Palestine Ahliya University, 2025.
