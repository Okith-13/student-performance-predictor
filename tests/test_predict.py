import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


SAMPLE_INPUT = {
    'study_hours_per_day':    5,
    'attendance_rate':        85,
    'sleep_hours':            7,
    'assignments_completed':  90,
    'previous_gpa':           3.2,
    'extracurricular_hours':  2,
    'tutoring_sessions':      3,
    'parent_education_level': 2,
}

LOW_PERFORMER = {
    'study_hours_per_day':    0.5,
    'attendance_rate':        40,
    'sleep_hours':            4,
    'assignments_completed':  20,
    'previous_gpa':           1.0,
    'extracurricular_hours':  15,
    'tutoring_sessions':      0,
    'parent_education_level': 0,
}

HIGH_PERFORMER = {
    'study_hours_per_day':    10,
    'attendance_rate':        98,
    'sleep_hours':            7.5,
    'assignments_completed':  100,
    'previous_gpa':           4.0,
    'extracurricular_hours':  2,
    'tutoring_sessions':      8,
    'parent_education_level': 3,
}


# ─── Feature Engineering Tests ───────────────────────────────────────────────

def test_engineer_features_columns():
    """Engineered DataFrame must contain all expected feature columns."""
    from preprocess import engineer_features, FEATURE_COLS
    import pandas as pd

    df = pd.DataFrame([SAMPLE_INPUT])
    # Add dummy targets to avoid KeyError
    df['exam_score'] = 75
    df['grade'] = 'B'

    df_eng = engineer_features(df)
    for col in ['study_efficiency', 'sleep_quality_flag', 'engagement_score']:
        assert col in df_eng.columns, f"Missing engineered column: {col}"


def test_sleep_quality_flag():
    """sleep_quality_flag should be 1 for 6–8 hrs, else 0."""
    from preprocess import engineer_features
    import pandas as pd

    for hrs, expected in [(5.9, 0), (6.0, 1), (7.5, 1), (8.0, 1), (8.1, 0)]:
        row = {**SAMPLE_INPUT, 'sleep_hours': hrs, 'exam_score': 75, 'grade': 'B'}
        df = engineer_features(pd.DataFrame([row]))
        assert df['sleep_quality_flag'].iloc[0] == expected, f"Failed for sleep_hours={hrs}"


def test_study_efficiency_formula():
    """study_efficiency = study_hours × (assignments / 100)"""
    from preprocess import engineer_features
    import pandas as pd

    row = {**SAMPLE_INPUT, 'study_hours_per_day': 6, 'assignments_completed': 80,
           'exam_score': 75, 'grade': 'B'}
    df = engineer_features(pd.DataFrame([row]))
    expected = 6 * (80 / 100)
    assert abs(df['study_efficiency'].iloc[0] - expected) < 1e-6


# ─── Prediction API Tests (require trained models) ────────────────────────────

def models_available():
    """Helper: check whether trained model files exist."""
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    return (
        os.path.exists(os.path.join(models_dir, 'regression_model.pkl')) and
        os.path.exists(os.path.join(models_dir, 'classification_model.pkl'))
    )


@pytest.mark.skipif(not models_available(), reason="Models not trained yet")
def test_predict_returns_expected_keys():
    from predict import predict
    result = predict(SAMPLE_INPUT)
    for key in ['predicted_score', 'predicted_grade', 'grade_probabilities', 'risk_level']:
        assert key in result


@pytest.mark.skipif(not models_available(), reason="Models not trained yet")
def test_predict_score_in_range():
    from predict import predict
    result = predict(SAMPLE_INPUT)
    assert 0 <= result['predicted_score'] <= 100


@pytest.mark.skipif(not models_available(), reason="Models not trained yet")
def test_predict_valid_grade():
    from predict import predict
    result = predict(SAMPLE_INPUT)
    assert result['predicted_grade'] in ['A', 'B', 'C', 'D', 'F']


@pytest.mark.skipif(not models_available(), reason="Models not trained yet")
def test_high_performer_scores_higher_than_low():
    from predict import predict
    high = predict(HIGH_PERFORMER)['predicted_score']
    low  = predict(LOW_PERFORMER)['predicted_score']
    assert high > low, f"Expected high performer ({high}) > low performer ({low})"


@pytest.mark.skipif(not models_available(), reason="Models not trained yet")
def test_risk_level_values():
    from predict import predict
    for student in [SAMPLE_INPUT, LOW_PERFORMER, HIGH_PERFORMER]:
        result = predict(student)
        assert result['risk_level'] in ['Low', 'Medium', 'High']


@pytest.mark.skipif(not models_available(), reason="Models not trained yet")
def test_grade_probabilities_sum_to_100():
    from predict import predict
    result = predict(SAMPLE_INPUT)
    total = sum(result['grade_probabilities'].values())
    assert abs(total - 100) < 1.0, f"Probabilities sum to {total}, expected ~100"


# ─── Flask API Tests ──────────────────────────────────────────────────────────

@pytest.fixture
def client():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "app", os.path.join(os.path.dirname(__file__), '..', 'app.py'))
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    app_module.app.config['TESTING'] = True
    with app_module.app.test_client() as c:
        yield c


def test_health_endpoint(client):
    res = client.get('/api/health')
    assert res.status_code == 200
    data = res.get_json()
    assert 'status' in data
    assert data['status'] == 'ok'


def test_predict_missing_fields(client):
    res = client.post('/api/predict',
                      json={'study_hours_per_day': 5},
                      content_type='application/json')
    # Either 400 (validation error) or 503 (models not trained)
    assert res.status_code in [400, 503]


@pytest.mark.skipif(not models_available(), reason="Models not trained yet")
def test_full_api_prediction(client):
    res = client.post('/api/predict',
                      json=SAMPLE_INPUT,
                      content_type='application/json')
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    assert 'predicted_score' in data['data']
