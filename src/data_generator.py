import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_SAMPLES = 2000

np.random.seed(RANDOM_SEED)


def generate_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:

    # --- Core features ---
    study_hours = np.clip(np.random.normal(4.5, 2.0, n_samples), 0, 12)
    attendance_rate = np.clip(np.random.normal(78, 15, n_samples), 0, 100)
    sleep_hours = np.clip(np.random.normal(6.8, 1.2, n_samples), 3, 10)
    assignments_completed = np.clip(np.random.normal(75, 20, n_samples), 0, 100)
    previous_gpa = np.clip(np.random.normal(2.8, 0.7, n_samples), 0.0, 4.0)
    extracurricular_hours = np.clip(np.random.exponential(3, n_samples), 0, 20)
    tutoring_sessions = np.random.choice(range(0, 11), n_samples,
                                          p=[0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01, 0.01])
    parent_education_level = np.random.choice([0, 1, 2, 3], n_samples,
                                               p=[0.1, 0.35, 0.35, 0.20])

    # --- Compute exam score with realistic weights + noise ---
    score = (
        study_hours          * 3.5 +
        attendance_rate      * 0.25 +
        sleep_hours          * 1.8 +
        assignments_completed * 0.20 +
        previous_gpa         * 8.0 +
        tutoring_sessions    * 1.2 +
        parent_education_level * 1.5 -
        np.clip(extracurricular_hours - 5, 0, None) * 0.5 +  # penalty for too many extra hours
        np.random.normal(0, 5, n_samples)  # noise
    )

    # Normalise to 0–100
    score = (score - score.min()) / (score.max() - score.min()) * 100
    score = np.clip(score, 0, 100).round(1)

    # --- Grade classification ---
    def score_to_grade(s):
        if s >= 90: return 'A'
        elif s >= 80: return 'B'
        elif s >= 65: return 'C'
        elif s >= 50: return 'D'
        else: return 'F'

    grade = [score_to_grade(s) for s in score]

    df = pd.DataFrame({
        'study_hours_per_day':    study_hours.round(1),
        'attendance_rate':        attendance_rate.round(1),
        'sleep_hours':            sleep_hours.round(1),
        'assignments_completed':  assignments_completed.round(1),
        'previous_gpa':           previous_gpa.round(2),
        'extracurricular_hours':  extracurricular_hours.round(1),
        'tutoring_sessions':      tutoring_sessions,
        'parent_education_level': parent_education_level,
        'exam_score':             score,
        'grade':                  grade,
    })

    return df


if __name__ == '__main__':
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    df = generate_dataset()
    output_path = 'data/raw/students.csv'
    df.to_csv(output_path, index=False)

    print(f"✅ Dataset saved to {output_path}")
    print(f"   Shape  : {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\nGrade distribution:\n{df['grade'].value_counts().sort_index()}")
    print(f"\nScore stats:\n{df['exam_score'].describe().round(2)}")
