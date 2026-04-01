import joblib
import pandas as pd


def load_pipeline(path):
    return joblib.load(path)


def predict_credit(form_data, pipeline):
    raw_input = pd.DataFrame([{
        "CODE_GENDER": form_data["gender"],
        "FLAG_OWN_CAR": form_data["own_car"],
        "NAME_EDUCATION_TYPE": form_data["education"],
        "AMT_INCOME_TOTAL": float(form_data["income"]),
        "AGE": int(form_data["age"]),
        "YEARS_EMPLOYED": int(form_data["years_employed"]),
        "CNT_FAM_MEMBERS": int(form_data["family_members"]),
    }])

    prediction = pipeline.predict(raw_input)[0]
    proba = pipeline.predict_proba(raw_input)[0]

    return {
        "prediction": prediction,
        "acceptance": round(proba[1] * 100, 1),
        "rejection": round(proba[0] * 100, 1),
    }