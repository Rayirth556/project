"""
train_full_31_feature_model.py
================================
Train model on all 31 features to match the workflow.

This creates a model that works with the complete feature set
including the 3 features we previously removed.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import optuna
from sklearn.calibration import calibration_curve

sys.path.append(str(Path(__file__).parent))
from pdr_pipeline.layer_2_feature_engine import FeatureStoreMSME

# Paths
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def load_processed_data():
    """Load processed features and labels"""
    print("Loading processed data...")
    
    # Use the preprocessed data from our transaction processing
    if not (DATA_DIR / "features.parquet").exists():
        print("Processed data not found. Running transaction preprocessing first...")
        from preprocess_transaction_data import TransactionDataPreprocessor
        preprocessor = TransactionDataPreprocessor()
        preprocessor.run_preprocessing(max_clients=1000)
    
    X = pd.read_parquet(DATA_DIR / "features.parquet")
    y = pd.read_parquet(DATA_DIR / "labels.parquet").squeeze()
    
    print(f"Loaded {len(X):,} samples x {X.shape[1]} features")
    print(f"Default rate: {y.mean():.1%} ({y.sum():,} defaults / {len(y):,} total)")
    
    return X, y

def get_optuna_params(trial):
    """Get XGBoost parameters for Optuna optimization"""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
    }

def train_full_model():
    """Train model on all 31 features"""
    print("=" * 60)
    print("TRAINING FULL 31-FEATURE MODEL")
    print("=" * 60)
    
    # Load data
    X, y = load_processed_data()
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Imbalance: scale_pos_weight={scale_pos_weight:.3f}")
    
    # Hyperparameter optimization
    print("\n[2/4] Hyperparameter search...")
    def objective(trial):
        params = get_optuna_params(trial)
        params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1
        })
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        val_pred = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, val_pred)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, timeout=1800)
    
    best_params = study.best_params
    print(f"Best AUC: {study.best_value:.4f}")
    print(f"Best params: {best_params}")
    
    # Train final model
    print("\n[3/4] Training final model...")
    final_params = best_params.copy()
    final_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1
    })
    
    model = xgb.XGBClassifier(**final_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Evaluation
    print("\n[4/4] Evaluation...")
    test_pred = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, test_pred)
    avg_precision = average_precision_score(y_test, test_pred)
    brier = brier_score_loss(y_test, test_pred)
    
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"Test Avg Precision: {avg_precision:.4f}")
    print(f"Test Brier Score: {brier:.4f}")
    
    # Calibration
    print("\nCalibration: Platt scaling...")
    from sklearn.isotonic import IsotonicRegression
    
    # Use validation set for calibration
    val_pred = model.predict_proba(X_val)[:, 1]
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(val_pred, y_val)
    
    # Save calibration parameters (simplified Platt scaling)
    val_pred_clipped = np.clip(val_pred, 1e-6, 1 - 1e-6)
    logit_val = np.log(val_pred_clipped / (1 - val_pred_clipped))
    logit_y = np.log(y_val / (1 - y_val))
    
    # Simple linear regression for Platt scaling
    A = np.cov(logit_val, logit_y)[0, 1] / np.var(logit_val)
    B = np.mean(logit_y) - A * np.mean(logit_val)
    
    calibration = {
        "method": "platt",
        "A": float(A),
        "B": float(B)
    }
    
    # Feature importance
    feature_importance = {}
    for feature, importance in zip(X.columns, model.feature_importances_):
        feature_importance[feature] = float(importance)
    
    # Save model
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
    
    # Save training report
    report = f"""
FULL 31-FEATURE MODEL TRAINING REPORT
=====================================

Training Data:
- Samples: {len(X):,}
- Features: {X.shape[1]}
- Default Rate: {y.mean():.1%}

Data Split:
- Train: {len(X_train):,}
- Validation: {len(X_val):,}
- Test: {len(X_test):,}

Model Performance:
- Test ROC-AUC: {roc_auc:.4f}
- Test Avg Precision: {avg_precision:.4f}
- Test Brier Score: {brier:.4f}

Hyperparameters:
{json.dumps(best_params, indent=2)}

Top 10 Feature Importances:
"""
    
    # Add top features
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(sorted_features[:10], 1):
        report += f"{i}. {feature}: {importance:.4f}\n"
    
    report_path = MODELS_DIR / "training_report_full_31.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nSuccess! Model saved to: {model_path}")
    print(f"Calibration saved to: {cal_path}")
    print(f"Feature importance saved to: {imp_path}")
    print(f"Training report saved to: {report_path}")
    
    return model, calibration

def main():
    """Main training function"""
    try:
        model, calibration = train_full_model()
        print("\n" + "=" * 60)
        print("FULL 31-FEATURE MODEL TRAINING COMPLETE!")
        print("=" * 60)
        print("Ready for simplified_risk_workflow.py")
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
