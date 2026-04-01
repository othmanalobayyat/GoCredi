import joblib
import pandas as pd
from collections import defaultdict

# Maps internal column names to labels shown on the result page
COLUMN_LABELS = {
    "AMT_INCOME_TOTAL": "Annual Income",
    "AGE": "Age",
    "YEARS_EMPLOYED": "Years Employed",
    "CNT_FAM_MEMBERS": "Family Members",
    "CODE_GENDER": "Gender",
    "FLAG_OWN_CAR": "Car Ownership",
    "NAME_EDUCATION_TYPE": "Education Level",
    "FLAG_OWN_REALTY": "Property Ownership",
    "NAME_INCOME_TYPE": "Income Type",
    "NAME_FAMILY_STATUS": "Family Status",
}


def _get_top_features(pipeline, n=3):
    """
    Returns the top-n most important feature labels from the trained RF pipeline.

    How it works:
    1. Retrieve feature names after preprocessing via get_feature_names_out().
       These look like 'num__AGE' or 'cat__NAME_INCOME_TYPE_Pensioner'.
    2. Retrieve feature_importances_ from the RandomForestClassifier step.
    3. OHE expands one categorical column into multiple binary columns —
       sum their importances back to the original column so the ranking
       reflects original features, not individual dummy variables.
    4. Sort and return the top-n as human-readable labels.

    Note: these are global model importances (how much each feature matters
    across all predictions), not SHAP-style per-prediction values.
    """
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["clf"]

        feature_names = preprocessor.get_feature_names_out()
        importances = clf.feature_importances_

        # Collect original categorical column names for reverse-mapping OHE output
        cat_columns = [
            col
            for name, _, cols in preprocessor.transformers_
            if name == "cat"
            for col in cols
        ]

        # Aggregate importance scores per original column
        col_importances = defaultdict(float)
        for fname, imp in zip(feature_names, importances):
            if fname.startswith("num__"):
                orig = fname[5:]                          # strip 'num__'
            elif fname.startswith("cat__"):
                suffix = fname[5:]                        # strip 'cat__'
                orig = next(
                    (col for col in cat_columns if suffix.startswith(col + "_")),
                    suffix,                               # fallback: use as-is
                )
            else:
                orig = fname
            col_importances[orig] += imp

        top_n = sorted(col_importances.items(), key=lambda x: x[1], reverse=True)[:n]
        return [COLUMN_LABELS.get(col, col) for col, _ in top_n]

    except Exception:
        return []


def load_pipeline(path):
    return joblib.load(path)


def predict_credit(form_data, pipeline):
    raw_input = pd.DataFrame([{
        "CODE_GENDER": form_data["gender"],
        "FLAG_OWN_CAR": form_data["own_car"],
        "NAME_EDUCATION_TYPE": form_data["education"],
        "FLAG_OWN_REALTY": form_data["own_realty"],
        "NAME_INCOME_TYPE": form_data["income_type"],
        "NAME_FAMILY_STATUS": form_data["family_status"],
        "AMT_INCOME_TOTAL": float(form_data["income"]),
        "AGE": int(form_data["age"]),
        "YEARS_EMPLOYED": int(form_data["years_employed"]),
        "CNT_FAM_MEMBERS": int(form_data["family_members"]),
    }])

    prediction = pipeline.predict(raw_input)[0]
    proba = pipeline.predict_proba(raw_input)[0]

    acceptance = round(proba[1] * 100, 1)

    if acceptance >= 70:
        risk_level = "Low"
    elif acceptance >= 40:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "prediction": int(prediction),
        "acceptance": acceptance,
        "rejection": round(proba[0] * 100, 1),
        "risk_level": risk_level,
        "top_features": _get_top_features(pipeline),
    }