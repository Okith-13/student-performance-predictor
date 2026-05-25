# Student Performance Predictor

A machine learning project that predicts student exam performance based on study habits, attendance, and other behavioral features — using both **Regression** (predict exact score) and **Classification** (predict grade category).

---

## Project Overview

| Attribute | Details |
|-----------|---------|
| ML Type | Regression + Classification |
| Target (Regression) | Exam Score (0–100) |
| Target (Classification) | Grade (A / B / C / D / F) |
| Dataset | Synthetic + real-world inspired features |
| Models Used | Linear Regression, Random Forest, XGBoost, Logistic Regression |
| Interface | Flask Web App + REST API |

---

## Project Structure

```
student-performance-predictor/
├── data/
│   ├── raw/                    # Raw dataset (CSV)
│   └── processed/              # Cleaned & feature-engineered data
├── models/
│   ├── regression_model.pkl    # Trained regression model
│   └── classification_model.pkl
├── notebooks/
│   └── EDA_and_Modeling.ipynb  # Full analysis notebook
├── src/
│   ├── data_generator.py       # Synthetic dataset generator
│   ├── preprocess.py           # Feature engineering & preprocessing
│   ├── train.py                # Model training pipeline
│   └── predict.py              # Prediction logic
├── static/                     # CSS & JS for web UI
├── templates/                  # HTML templates
├── tests/
│   └── test_predict.py         # Unit tests
├── app.py                      # Flask application
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/student-performance-predictor.git
cd student-performance-predictor
pip install -r requirements.txt
```

### 2. Generate Data & Train Models
```bash
python src/data_generator.py      # Creates data/raw/students.csv
python src/train.py               # Trains and saves models
```

### 3. Run the Web App
```bash
python app.py
# Visit http://localhost:5000
```

### 4. Or Use the API
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "study_hours_per_day": 5,
    "attendance_rate": 85,
    "sleep_hours": 7,
    "assignments_completed": 90,
    "previous_gpa": 3.2,
    "extracurricular_hours": 2,
    "tutoring_sessions": 3,
    "parent_education_level": 2
  }'
```

---

## Features Used

| Feature | Description | Range |
|---------|-------------|-------|
| `study_hours_per_day` | Daily study time in hours | 0–12 |
| `attendance_rate` | % of classes attended | 0–100 |
| `sleep_hours` | Avg nightly sleep | 3–10 |
| `assignments_completed` | % of assignments done | 0–100 |
| `previous_gpa` | GPA from previous semester | 0.0–4.0 |
| `extracurricular_hours` | Weekly extracurricular time | 0–20 |
| `tutoring_sessions` | Tutoring sessions per month | 0–10 |
| `parent_education_level` | Parent's education (0=None, 3=Postgrad) | 0–3 |

---

## Model Performance

| Model | Task | R² / Accuracy |
|-------|------|----------------|
| Linear Regression | Regression | ~0.82 |
| Random Forest | Regression | ~0.91 |
| Logistic Regression | Classification | ~79% |
| Random Forest | Classification | ~88% |

---

## Key Insights (from EDA)

- **Study hours** and **attendance rate** are the top 2 predictors
- Students sleeping 6–8 hours consistently score higher
- Assignment completion rate has a near-linear correlation with scores
- Previous GPA is the strongest single predictor for classification

---

## Tech Stack

- **Python 3.10+**
- **scikit-learn** — ML models
- **XGBoost** — Gradient boosting
- **pandas / numpy** — Data processing
- **matplotlib / seaborn** — Visualization
- **Flask** — Web server & API
- **Jupyter** — Exploration notebook

---