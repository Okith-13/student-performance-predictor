import numpy as np
import pandas as pd
import joblib
import os

# Paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

_reg_model   = None
_clf_model   = None
_scaler      = None
_label_enc   = None


def _load_artifacts():
    global _reg_model, _clf_model, _scaler, _label_enc
    if _reg_model is None:
        _reg_model  = joblib.load(os.path.join(MODEL_DIR, 'regression_model.pkl'))
        _clf_model  = joblib.load(os.path.join(MODEL_DIR, 'classification_model.pkl'))
        _scaler     = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        _label_enc  = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))


def _build_features(data: dict) -> pd.DataFrame:
    """Turn raw input dict into a feature-engineered DataFrame."""
    study_hours        = float(data['study_hours_per_day'])
    attendance_rate    = float(data['attendance_rate'])
    sleep_hours        = float(data['sleep_hours'])
    assignments_done   = float(data['assignments_completed'])
    previous_gpa       = float(data['previous_gpa'])
    extra_hours        = float(data['extracurricular_hours'])
    tutoring           = float(data['tutoring_sessions'])
    parent_edu         = float(data['parent_education_level'])

    # Engineered features (must match preprocess.py)
    study_efficiency   = study_hours * (assignments_done / 100)
    sleep_quality_flag = 1 if 6 <= sleep_hours <= 8 else 0
    engagement_score   = (attendance_rate * 0.4 + assignments_done * 0.4 + tutoring * 2) / 100

    row = {
        'study_hours_per_day':    study_hours,
        'attendance_rate':        attendance_rate,
        'sleep_hours':            sleep_hours,
        'assignments_completed':  assignments_done,
        'previous_gpa':           previous_gpa,
        'extracurricular_hours':  extra_hours,
        'tutoring_sessions':      tutoring,
        'parent_education_level': parent_edu,
        'study_efficiency':       study_efficiency,
        'sleep_quality_flag':     sleep_quality_flag,
        'engagement_score':       engagement_score,
    }

    return pd.DataFrame([row])


def predict(data: dict) -> dict:
    """
    Run inference on a single student record.

    Args:
        data: dict with keys matching the 8 input features.

    Returns:
        dict with 'predicted_score', 'predicted_grade', 'grade_probabilities'
    """
    _load_artifacts()

    df = _build_features(data)
    X_scaled = _scaler.transform(df)

    # Regression
    score = float(np.clip(_reg_model.predict(X_scaled)[0], 0, 100))

    # Classification
    grade_encoded = int(_clf_model.predict(X_scaled)[0])
    grade = _label_enc.inverse_transform([grade_encoded])[0]

    grade_probs = {}
    if hasattr(_clf_model, 'predict_proba'):
        probs = _clf_model.predict_proba(X_scaled)[0]
        for cls, prob in zip(_label_enc.classes_, probs):
            grade_probs[cls] = round(float(prob) * 100, 1)

    # Risk level
    if score >= 80:
        risk = 'Low'
    elif score >= 60:
        risk = 'Medium'
    else:
        risk = 'High'

    return {
        'predicted_score':      round(score, 1),
        'predicted_grade':      grade,
        'grade_probabilities':  grade_probs,
        'risk_level':           risk,
    }


if __name__ == '__main__':
    sample = {
        'study_hours_per_day':    5,
        'attendance_rate':        85,
        'sleep_hours':            7,
        'assignments_completed':  90,
        'previous_gpa':           3.2,
        'extracurricular_hours':  2,
        'tutoring_sessions':      3,
        'parent_education_level': 2,
    }
    result = predict(sample)
    print("Prediction result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
