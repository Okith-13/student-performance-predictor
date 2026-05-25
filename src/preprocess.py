import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os


FEATURE_COLS = [
    'study_hours_per_day',
    'attendance_rate',
    'sleep_hours',
    'assignments_completed',
    'previous_gpa',
    'extracurricular_hours',
    'tutoring_sessions',
    'parent_education_level',
    # Engineered features (added below)
    'study_efficiency',
    'sleep_quality_flag',
    'engagement_score',
]

REGRESSION_TARGET = 'exam_score'
CLASSIFICATION_TARGET = 'grade'


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that improve predictive power."""
    df = df.copy()

    # Study efficiency: study hours weighted by assignment completion
    df['study_efficiency'] = df['study_hours_per_day'] * (df['assignments_completed'] / 100)

    # Sleep quality flag: 1 if sleeping 6–8 hrs (optimal), else 0
    df['sleep_quality_flag'] = df['sleep_hours'].apply(lambda x: 1 if 6 <= x <= 8 else 0)

    # Engagement score: blend of attendance + tutoring + assignments
    df['engagement_score'] = (
        df['attendance_rate'] * 0.4 +
        df['assignments_completed'] * 0.4 +
        df['tutoring_sessions'] * 2  # scale up since range is 0-10
    ) / 100

    return df


def load_and_preprocess(data_path: str = 'data/raw/students.csv',
                         save_processed: bool = True):
    """
    Full preprocessing pipeline.
    Returns: X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, scaler, label_encoder
    """
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape}")

    # Feature engineering
    df = engineer_features(df)

    # Encode grade labels
    le = LabelEncoder()
    le.fit(['A', 'B', 'C', 'D', 'F'])
    df['grade_encoded'] = le.transform(df[CLASSIFICATION_TARGET])

    X = df[FEATURE_COLS]
    y_regression = df[REGRESSION_TARGET]
    y_classification = df['grade_encoded']

    # Train-test split (stratified on grade)
    X_train, X_test, \
    y_reg_train, y_reg_test, \
    y_clf_train, y_clf_test = train_test_split(
        X, y_regression, y_classification,
        test_size=0.2, random_state=42, stratify=y_classification
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save_processed:
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(le, 'models/label_encoder.pkl')

        pd.DataFrame(X_train_scaled, columns=FEATURE_COLS).to_csv('data/processed/X_train.csv', index=False)
        pd.DataFrame(X_test_scaled,  columns=FEATURE_COLS).to_csv('data/processed/X_test.csv',  index=False)
        y_reg_train.to_csv('data/processed/y_reg_train.csv', index=False)
        y_reg_test.to_csv('data/processed/y_reg_test.csv',   index=False)
        pd.Series(y_clf_train.values, name='grade_encoded').to_csv('data/processed/y_clf_train.csv', index=False)
        pd.Series(y_clf_test.values,  name='grade_encoded').to_csv('data/processed/y_clf_test.csv',  index=False)

        print("✅ Processed data and artifacts saved to data/processed/ and models/")

    return X_train_scaled, X_test_scaled, y_reg_train, y_reg_test, y_clf_train, y_clf_test, scaler, le


if __name__ == '__main__':
    load_and_preprocess()
