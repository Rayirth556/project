"""
train_real_world_model.py
==========================
Trains an XGBoost binary classifier on real-world credit default data.

Inputs (from preprocess_real_world_data.py):
    data/processed/features.parquet
    data/processed/labels.parquet

Output:
    models/pdr_xgb_realworld.json       ← XGBoost model (Layer 3 loads this)
    models/feature_importance.json
    models/training_report.txt

Workflow:
    1. Load pre-engineered features (Layer 2 output)
    2. Clean & oversample via SMOTE if imbalanced
    3. Tune with Optuna (fast 30-trial search) or use default HPs
    4. Train final XGBoost on full train set
    5. Evaluate on held-out test set (AUC, AP, Brier Score)
    6. Save model in JSON format (InferenceEngine-compatible)

Run:
    python train_real_world_model.py [--no-tune] [--model-output models/pdr_xgb_realworld.json]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).parent
FEAT_PATH  = BASE_DIR / "data" / "processed" / "features.parquet"
LABEL_PATH = BASE_DIR / "data" / "processed" / "labels.parquet"
MODEL_DIR  = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL_PATH = MODEL_DIR / "pdr_xgb_realworld.json"


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading & Cleaning
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    if not FEAT_PATH.exists() or not LABEL_PATH.exists():
        print("  Error: Preprocessed data not found.")
        print("   Run first:  python preprocess_real_world_data.py")
        sys.exit(1)

    X = pd.read_parquet(FEAT_PATH)
    y = pd.read_parquet(LABEL_PATH).squeeze()

    print(f"  Loaded {len(X):,} samples × {X.shape[1]} features")
    print(f"  Default rate: {y.mean():.1%}  ({y.sum():,} defaults / {len(y):,} total)")
    return X, y


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/NaN with 0, clip extreme outliers."""
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    # Winsorise 99th percentile per column
    for col in X.columns:
        p99 = X[col].quantile(0.99)
        p01 = X[col].quantile(0.01)
        X[col] = X[col].clip(lower=p01, upper=p99)
    return X


# ══════════════════════════════════════════════════════════════════════════════
#  SMOTE Oversampling (optional — triggers when minority class < 20%)
# ══════════════════════════════════════════════════════════════════════════════

def maybe_oversample(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minority_rate = y_train.mean()
    if minority_rate >= 0.20:
        print(f"  Class balance OK ({minority_rate:.1%} positive) — no oversampling needed.")
        return X_train, y_train

    try:
        from imblearn.over_sampling import SMOTE   # noqa: PLC0415
        print(f"  Applying SMOTE (minority rate {minority_rate:.1%}) …")
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"  After SMOTE: {len(X_res):,} samples  (positive rate: {y_res.mean():.1%})")
        return X_res, y_res
    except ImportError:
        print("  Warning:  imbalanced-learn not installed. Skipping SMOTE.")
        print("     Install with:  pip install imbalanced-learn")
        scale = (1 - minority_rate) / minority_rate
        print(f"  Using scale_pos_weight={scale:.2f} in XGBoost instead.")
        return X_train, y_train


# ══════════════════════════════════════════════════════════════════════════════
#  Optuna Hyperparameter Tuning (30-trial fast search)
# ══════════════════════════════════════════════════════════════════════════════

def tune_hyperparams(X_train, y_train, n_trials: int = 30) -> dict:
    try:
        import optuna                         # noqa: PLC0415
        import xgboost as xgb                 # noqa: PLC0415
        from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415
        from sklearn.metrics import roc_auc_score             # noqa: PLC0415

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as e:
        print(f"  Warning:  {e} — skipping Optuna tuning, using defaults.")
        return default_hyperparams()

    print(f"  Running Optuna ({n_trials} trials) …")

    scale_pos = float((y_train == 0).sum() / (y_train == 1).sum()) if y_train.mean() < 0.5 else 1.0

    def objective(trial: optuna.Trial) -> float:
        params = {
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "scale_pos_weight": scale_pos,
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            Xtr, Xval = X_train[tr_idx], X_train[val_idx]
            ytr, yval = y_train[tr_idx], y_train[val_idx]
            model = xgb.XGBClassifier(**params, random_state=42)
            model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
            preds = model.predict_proba(Xval)[:, 1]
            scores.append(roc_auc_score(yval, preds))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["scale_pos_weight"] = scale_pos
    print(f"  Best AUC (CV): {study.best_value:.4f}  |  Params: {best}")
    return best


def default_hyperparams() -> dict:
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
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Training & Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    model_path: Path,
    feature_names: list[str],
) -> dict:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, brier_score_loss,
        classification_report, confusion_matrix,
    )

    X_arr = X.values.astype(np.float32)
    y_arr = y.values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.20, stratify=y_arr, random_state=42
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Oversample training set only
    X_train_res, y_train_res = maybe_oversample(X_train, y_train)

    scale_pos = params.pop("scale_pos_weight", float((y_train_res == 0).sum() / max((y_train_res == 1).sum(), 1)))

    model = xgb.XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        scale_pos_weight=scale_pos,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=30,
    )

    print("  Training final XGBoost model …")
    model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Evaluate
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.35).astype(int)

    auc    = roc_auc_score(y_test, proba)
    ap     = average_precision_score(y_test, proba)
    brier  = brier_score_loss(y_test, proba)

    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=["No Default", "Default"])

    metrics = {
        "roc_auc":  round(auc, 4),
        "avg_precision": round(ap, 4),
        "brier_score": round(brier, 4),
        "n_train": len(X_train_res),
        "n_test": len(X_test),
        "default_rate_test": round(float(y_test.mean()), 4),
        "confusion_matrix": cm.tolist(),
    }

    print(f"\n  ── Evaluation ────────────────────────────────────────────")
    print(f"  ROC-AUC            : {auc:.4f}")
    print(f"  Avg Precision (AP) : {ap:.4f}")
    print(f"  Brier Score        : {brier:.4f}  (lower = better)")
    print(f"\n{report}")

    # Feature importance
    importance = dict(zip(feature_names, model.feature_importances_))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    print("  ── Top 10 Feature Importances ────────────────────────────")
    for feat, imp in list(importance_sorted.items())[:10]:
        bar = "█" * int(imp * 200)
        print(f"  {feat:<40} ({imp:.4f})  {bar}")

    # Save model (Booster JSON — InferenceEngine compatible)
    booster = model.get_booster()
    booster.save_model(str(model_path))
    print(f"\n  Success: Model saved → {model_path}")

    # Save feature importance
    fi_path = model_path.parent / "feature_importance.json"
    fi_path.write_text(json.dumps(importance_sorted, indent=2))

    # Save training report
    report_path = model_path.parent / "training_report.txt"
    with open(report_path, "w") as f:
        f.write(f"PDR Pipeline — Layer 3 Training Report\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Features: {len(feature_names)}\n\n")
        f.write(f"Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nClassification Report:\n{report}\n")
        f.write(f"\nFeature Importances (top 20):\n")
        for feat, imp in list(importance_sorted.items())[:20]:
            f.write(f"  {feat}: {imp:.6f}\n")
    print(f"  Success: Report saved → {report_path}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Layer 3 XGBoost model on real-world credit data.")
    parser.add_argument("--no-tune", action="store_true", help="Skip Optuna and use default hyperparams")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials (default 30)")
    parser.add_argument("--model-output", type=str, default=str(DEFAULT_MODEL_PATH),
                        help="Output path for the trained model JSON")
    args = parser.parse_args()

    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import xgboost as xgb  # noqa: F401
        from sklearn.model_selection import train_test_split  # noqa: F401
    except ImportError as e:
        print(f"Error: Missing dependency: {e}")
        print("   Install:  pip install xgboost scikit-learn")
        sys.exit(1)

    print("\n" + "=" * 62)
    print("  PDR Pipeline — Layer 3 XGBoost Training")
    print("=" * 62)

    print("\n[1/4] Loading preprocessed features …")
    X, y = load_data()
    feature_names = list(X.columns)

    print("\n[2/4] Cleaning features …")
    X = clean_features(X)

    print("\n[3/4] Hyperparameter search …")
    if args.no_tune:
        print("  Using default hyperparameters (--no-tune flag set).")
        params = default_hyperparams()
    else:
        params = tune_hyperparams(X.values.astype(np.float32), y.values.astype(int), n_trials=args.n_trials)

    print("\n[4/4] Training & Evaluation …")
    metrics = train_and_evaluate(X, y, params, model_path, feature_names)

    print("\n" + "=" * 62)
    print("  Success: Training Complete!")
    print(f"  ROC-AUC: {metrics['roc_auc']}  |  Avg Precision: {metrics['avg_precision']}")
    print(f"\n  To run inference with this model:")
    print(f"    from pdr_pipeline.layer_3_inference_engine import InferenceEngine")
    print(f"    engine = InferenceEngine(model_path=r'{model_path}')")
    print(f"    result = engine.predict(feature_vector)")
    print("=" * 62)


if __name__ == "__main__":
    main()
