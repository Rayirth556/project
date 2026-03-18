"""
quick_train_31_model.py
=====================
Quick training script for 31-feature model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# Paths
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def train_simple_31_model():
    """Simple training for 31-feature model"""
    print("=" * 60)
    print("QUICK 31-FEATURE MODEL TRAINING")
    print("=" * 60)
    
    # Load processed data
    if not (DATA_DIR / "features.parquet").exists():
        print("Running transaction preprocessing first...")
        from preprocess_transaction_data import TransactionDataPreprocessor
        preprocessor = TransactionDataPreprocessor()
        preprocessor.run_preprocessing(max_clients=1000)
    
    X = pd.read_parquet(DATA_DIR / "features.parquet")
    y = pd.read_parquet(DATA_DIR / "labels.parquet").squeeze()
    
    print(f"Loaded {len(X):,} samples x {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale pos weight: {scale_pos_weight:.3f}")
    
    # Simple XGBoost parameters
    params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Train model
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate
    test_pred = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, test_pred)
    avg_precision = average_precision_score(y_test, test_pred)
    brier = brier_score_loss(y_test, test_pred)
    
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"Test Avg Precision: {avg_precision:.4f}")
    print(f"Test Brier Score: {brier:.4f}")
    
    # Simple calibration (just identity for now)
    calibration = {
        "method": "identity",
        "A": 1.0,
        "B": 0.0
    }
    
    # Feature importance
    feature_importance = {}
    for feature, importance in zip(X.columns, model.feature_importances_):
        feature_importance[feature] = float(importance)
    
    # Save model with all 31 features
    model_path = MODELS_DIR / "pdr_xgb_full_31_features.json"
    model.save_model(str(model_path))
    
    # Save calibration
    cal_path = MODELS_DIR / "pdr_xgb_full_31_calibration.json"
    with open(cal_path, 'w') as f:
        json.dump(calibration, f, indent=2)
    
    # Save feature importance
    imp_path = MODELS_DIR / "feature_importance_full_31.json"
    with open(imp_path, 'w') as f:
        json.dump(feature_importance, f, indent=2)
    
    print(f"\nModel saved: {model_path}")
    print(f"Calibration saved: {cal_path}")
    print(f"Feature importance saved: {imp_path}")
    
    return model_path

if __name__ == "__main__":
    train_simple_31_model()
