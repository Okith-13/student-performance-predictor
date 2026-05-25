import numpy as np
import pandas as pd
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not found — skipping XGB models.")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_and_preprocess, FEATURE_COLS


def evaluate_regression(model, X_test, y_test, name="Model"):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    print(f"  {name:30s} | RMSE={rmse:.2f} | MAE={mae:.2f} | R²={r2:.4f}")
    return {'name': name, 'rmse': round(rmse, 4), 'mae': round(mae, 4), 'r2': round(r2, 4)}


def evaluate_classification(model, X_test, y_test, le, name="Model"):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"  {name:30s} | Accuracy={acc:.4f}")
    print(classification_report(y_test, preds, target_names=le.classes_))
    return {'name': name, 'accuracy': round(acc, 4)}


def plot_feature_importance(model, feature_names, title, save_path):
    if not hasattr(model, 'feature_importances_'):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(model, X_test, y_test, le, save_path):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix — Random Forest Classifier', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def train_all():
    print("\n📦 Loading & preprocessing data...")
    X_train, X_test, y_reg_train, y_reg_test, \
    y_clf_train, y_clf_test, scaler, le = load_and_preprocess()

    os.makedirs('models', exist_ok=True)
    os.makedirs('static/img', exist_ok=True)

    metrics = {'regression': [], 'classification': []}

    # ─── REGRESSION ──────────────────────────────────────────────────────────
    print("\n📈 Training Regression Models...")

    reg_models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression':  Ridge(alpha=1.0),
        'Random Forest Reg': RandomForestRegressor(n_estimators=150, max_depth=10,
                                                    random_state=42, n_jobs=-1),
    }
    if XGBOOST_AVAILABLE:
        reg_models['XGBoost Regressor'] = XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=42, n_jobs=-1, verbosity=0)

    best_reg_score, best_reg_model, best_reg_name = -np.inf, None, ''
    for name, model in reg_models.items():
        model.fit(X_train, y_reg_train)
        result = evaluate_regression(model, X_test, y_reg_test, name)
        metrics['regression'].append(result)
        if result['r2'] > best_reg_score:
            best_reg_score = result['r2']
            best_reg_model = model
            best_reg_name  = name

    print(f"\n🏆 Best Regression Model: {best_reg_name} (R²={best_reg_score:.4f})")
    joblib.dump(best_reg_model, 'models/regression_model.pkl')

    plot_feature_importance(
        best_reg_model, FEATURE_COLS,
        f'Feature Importance — {best_reg_name}',
        'static/img/regression_feature_importance.png'
    )

    # ─── CLASSIFICATION ───────────────────────────────────────────────────────
    print("\n📊 Training Classification Models...")

    clf_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest Clf':   RandomForestClassifier(n_estimators=150, max_depth=10,
                                                       random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                                           random_state=42),
    }
    if XGBOOST_AVAILABLE:
        clf_models['XGBoost Classifier'] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=42, n_jobs=-1, verbosity=0, use_label_encoder=False,
            eval_metric='mlogloss')

    best_clf_score, best_clf_model, best_clf_name = -np.inf, None, ''
    for name, model in clf_models.items():
        model.fit(X_train, y_clf_train)
        result = evaluate_classification(model, X_test, y_clf_test, le, name)
        metrics['classification'].append(result)
        if result['accuracy'] > best_clf_score:
            best_clf_score = result['accuracy']
            best_clf_model = model
            best_clf_name  = name

    print(f"\n🏆 Best Classification Model: {best_clf_name} (Acc={best_clf_score:.4f})")
    joblib.dump(best_clf_model, 'models/classification_model.pkl')

    plot_feature_importance(
        best_clf_model, FEATURE_COLS,
        f'Feature Importance — {best_clf_name}',
        'static/img/classification_feature_importance.png'
    )
    plot_confusion_matrix(
        best_clf_model, X_test, y_clf_test, le,
        'static/img/confusion_matrix.png'
    )

    # Save metrics summary
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ All models trained and saved.")
    print("   → models/regression_model.pkl")
    print("   → models/classification_model.pkl")
    print("   → models/metrics.json")


if __name__ == '__main__':
    train_all()
