"""
cross_validation_clean_model.py
================================
Performs 5-fold cross-validation on the clean model to get realistic performance estimates.
Removes leaked features and provides robust validation metrics.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
FEAT_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
LABEL_PATH = BASE_DIR / "data" / "processed" / "labels.parquet"

# Features to remove
LEAKED_FEATURES = ["academic_background_tier"]
CONSTANT_FEATURES = ["operating_cashflow_survival_flag", "turnover_inflation_spike"]
REMOVED_FEATURES = LEAKED_FEATURES + CONSTANT_FEATURES

def load_and_clean_data():
    """Load and clean data, removing problematic features"""
    if not FEAT_PATH.exists() or not LABEL_PATH.exists():
        raise FileNotFoundError("Preprocessed data not found. Run preprocess_real_world_data.py first")
    
    X = pd.read_parquet(FEAT_PATH)
    y = pd.read_parquet(LABEL_PATH).squeeze()
    
    # Remove problematic features
    X = X.drop(columns=REMOVED_FEATURES, errors='ignore')
    
    # Basic cleaning
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    
    # Winsorize
    for col in X.columns:
        p99 = X[col].quantile(0.99)
        p01 = X[col].quantile(0.01)
        X[col] = X[col].clip(lower=p01, upper=p99)
    
    print(f"Loaded {len(X):,} samples x {X.shape[1]} clean features")
    print(f"Default rate: {y.mean():.1%}")
    print(f"Removed features: {REMOVED_FEATURES}")
    
    return X, y

def get_default_params():
    """Default XGBoost parameters"""
    return {
        "n_estimators": 350,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "verbosity": 0,
        "early_stopping_rounds": 30,
    }

def cross_validate_model(X, y, n_folds=5, random_state=42):
    """Perform stratified k-fold cross-validation"""
    
    print(f"\n{'='*60}")
    print(f"5-FOLD CROSS-VALIDATION")
    print(f"{'='*60}")
    
    # Initialize
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    params = get_default_params()
    
    # Store results
    fold_results = []
    fold_metrics = {
        'roc_auc': [],
        'avg_precision': [],
        'brier_score': [],
        'train_auc': [],
        'val_auc': []
    }
    
    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/{n_folds}")
        print("-" * 30)
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Handle imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Train model
        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predictions
        train_proba = model.predict_proba(X_train)[:, 1]
        val_proba = model.predict_proba(X_val)[:, 1]
        
        # Metrics
        train_auc = roc_auc_score(y_train, train_proba)
        val_auc = roc_auc_score(y_val, val_proba)
        val_ap = average_precision_score(y_val, val_proba)
        val_brier = brier_score_loss(y_val, val_proba)
        
        # Store
        fold_result = {
            'fold': fold,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'train_default_rate': y_train.mean(),
            'val_default_rate': y_val.mean(),
            'train_auc': train_auc,
            'val_auc': val_auc,
            'val_avg_precision': val_ap,
            'val_brier_score': val_brier,
            'scale_pos_weight': scale_pos_weight
        }
        
        fold_results.append(fold_result)
        fold_metrics['roc_auc'].append(val_auc)
        fold_metrics['avg_precision'].append(val_ap)
        fold_metrics['brier_score'].append(val_brier)
        fold_metrics['train_auc'].append(train_auc)
        fold_metrics['val_auc'].append(val_auc)
        
        print(f"  Train AUC: {train_auc:.4f}")
        print(f"  Val AUC:   {val_auc:.4f}")
        print(f"  Val AP:    {val_ap:.4f}")
        print(f"  Val Brier: {val_brier:.4f}")
    
    # Aggregate results
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    
    metrics_summary = {}
    for metric, values in fold_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        metrics_summary[f'{metric}_mean'] = mean_val
        metrics_summary[f'{metric}_std'] = std_val
        print(f"{metric:>15}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Check for overfitting
    mean_train_auc = np.mean(fold_metrics['train_auc'])
    mean_val_auc = np.mean(fold_metrics['val_auc'])
    overfitting_gap = mean_train_auc - mean_val_auc
    
    print(f"\nOverfitting Analysis:")
    print(f"  Mean Train AUC: {mean_train_auc:.4f}")
    print(f"  Mean Val AUC:   {mean_val_auc:.4f}")
    print(f"  Gap:           {overfitting_gap:.4f}")
    
    if overfitting_gap > 0.05:
        print("  ⚠️  WARNING: Potential overfitting detected")
    else:
        print("  OK Good generalization")
    
    return {
        'fold_results': fold_results,
        'metrics_summary': metrics_summary,
        'overfitting_gap': overfitting_gap,
        'n_features': X.shape[1],
        'removed_features': REMOVED_FEATURES
    }

def save_cv_results(results, output_path):
    """Save cross-validation results"""
    output_path.write_text(json.dumps(results, indent=2))
    print(f"OK CV results saved to: {output_path}")

def main():
    """Run cross-validation"""
    try:
        # Load data
        X, y = load_and_clean_data()
        
        # Cross-validation
        results = cross_validate_model(X, y, n_folds=5)
        
        # Save results
        output_path = BASE_DIR / "models" / "cross_validation_results.json"
        save_cv_results(results, output_path)
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Model: Clean XGBoost (no leaked features)")
        print(f"Features: {results['n_features']} (removed {len(results['removed_features'])})")
        print(f"Validation: 5-fold stratified CV")
        print(f"ROC-AUC: {results['metrics_summary']['roc_auc_mean']:.4f} ± {results['metrics_summary']['roc_auc_std']:.4f}")
        print(f"Avg Precision: {results['metrics_summary']['avg_precision_mean']:.4f} ± {results['metrics_summary']['avg_precision_std']:.4f}")
        print(f"Brier Score: {results['metrics_summary']['brier_score_mean']:.4f} ± {results['metrics_summary']['brier_score_std']:.4f}")
        
        if results['overfitting_gap'] < 0.05:
            print("OK Model shows good generalization")
        else:
            print("⚠️  Model may be overfitting")
            
    except Exception as e:
        print(f"Error during cross-validation: {e}")
        raise

if __name__ == "__main__":
    main()
