from flask import Flask, render_template, request, jsonify
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__)


def get_predictor():
    """Lazy-load predictor (only after models are trained)."""
    try:
        from predict import predict
        return predict
    except Exception as e:
        return None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Main prediction page."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    REST API endpoint for predictions.

    POST /api/predict
    Content-Type: application/json

    Body: {
        "study_hours_per_day": 5,
        "attendance_rate": 85,
        "sleep_hours": 7,
        "assignments_completed": 90,
        "previous_gpa": 3.2,
        "extracurricular_hours": 2,
        "tutoring_sessions": 3,
        "parent_education_level": 2
    }
    """
    predict_fn = get_predictor()
    if predict_fn is None:
        return jsonify({'error': 'Models not trained yet. Run: python src/train.py'}), 503

    try:
        data = request.get_json(force=True)

        required_fields = [
            'study_hours_per_day', 'attendance_rate', 'sleep_hours',
            'assignments_completed', 'previous_gpa', 'extracurricular_hours',
            'tutoring_sessions', 'parent_education_level'
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        result = predict_fn(data)
        return jsonify({'success': True, 'data': result})

    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics')
def api_metrics():
    """Returns saved training metrics."""
    metrics_path = os.path.join(os.path.dirname(__file__), 'models', 'metrics.json')
    if not os.path.exists(metrics_path):
        return jsonify({'error': 'Metrics not found. Run: python src/train.py'}), 404
    with open(metrics_path) as f:
        return jsonify(json.load(f))


@app.route('/api/health')
def health():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    reg_exists = os.path.exists(os.path.join(models_dir, 'regression_model.pkl'))
    clf_exists = os.path.exists(os.path.join(models_dir, 'classification_model.pkl'))
    return jsonify({
        'status': 'ok',
        'models_trained': reg_exists and clf_exists,
        'regression_model': reg_exists,
        'classification_model': clf_exists,
    })


if __name__ == '__main__':
    print("🎓 Student Performance Predictor")
    print("   Visit: http://localhost:5000")
    print("   API  : POST /api/predict")
    app.run(debug=True, host='0.0.0.0', port=5000)
