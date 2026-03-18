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
    2. Clean features
    3. Split into train/val/test (no test-set leakage)
    4. Handle imbalance explicitly (scale_pos_weight OR SMOTE — not both)
    5. Tune with Optuna (optional) using TRAIN only and VAL scoring
    6. Train final XGBoost with early stopping on VAL
    7. Select decision threshold on VAL (objective-driven)
    8. Evaluate once on TEST (AUC, AP, Brier, confusion matrix, thresholded metrics)
    9. Save model in JSON format (InferenceEngine-compatible) + calibration (optional)

Run:
    python train_real_world_model.py [--no-tune] [--model-output models/pdr_xgb_realworld.json]
                                  [--imbalance scale_pos_weight|smote]
                                  [--threshold-strategy youden|f1|fixed --fixed-threshold 0.5]
                                  [--calibrate none|platt]
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
DEFAULT_CALIBRATION_PATH = MODEL_DIR / "pdr_xgb_realworld_calibration.json"
DEFAULT_FEATURE_AUDIT_PATH = MODEL_DIR / "feature_readiness_audit.json"


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

    print(f"  Loaded {len(X):,} samples x {X.shape[1]} features")
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
#  Feature readiness audit (honest production-readiness signal)
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_readiness_audit(X: pd.DataFrame) -> dict:
    """
    Summarize whether each feature is "alive" in the processed dataset.
    This does NOT manufacture signal; it reports distribution/variance realities.
    """
    audit: dict[str, dict] = {}
    for col in X.columns:
        s = pd.to_numeric(X[col], errors="coerce")
        missing = float(s.isna().mean())
        filled = s.fillna(0.0)
        zeros = float((filled == 0).mean())
        nunique = int(filled.nunique(dropna=False))
        std = float(filled.std(ddof=0)) if len(filled) else 0.0
        audit[col] = {
            "missing_rate": round(missing, 6),
            "zero_rate": round(zeros, 6),
            "n_unique": nunique,
            "mean": float(round(filled.mean(), 6)),
            "std": float(round(std, 6)),
            "min": float(round(filled.min(), 6)),
            "p01": float(round(filled.quantile(0.01), 6)),
            "p50": float(round(filled.quantile(0.50), 6)),
            "p99": float(round(filled.quantile(0.99), 6)),
            "max": float(round(filled.max(), 6)),
            "is_effectively_constant": bool((nunique <= 2) or (std == 0.0) or (zeros > 0.995)),
        }
    return {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "features": audit,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Optuna Hyperparameter Tuning (30-trial fast search)
# ══════════════════════════════════════════════════════════════════════════════

def tune_hyperparams(X_train, y_train, X_val, y_val, n_trials: int = 30, scale_pos_weight: float = 1.0) -> dict:
    try:
        import optuna                         # noqa: PLC0415
        import xgboost as xgb                 # noqa: PLC0415
        from sklearn.metrics import roc_auc_score             # noqa: PLC0415

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as e:
        print(f"  Warning:  {e} — skipping Optuna tuning, using defaults.")
        return default_hyperparams()

    print(f"  Running Optuna ({n_trials} trials) ...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "scale_pos_weight": scale_pos_weight,
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
        model = xgb.XGBClassifier(**params, random_state=42, early_stopping_rounds=30)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        return float(roc_auc_score(y_val, preds))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["scale_pos_weight"] = float(scale_pos_weight)
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

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_platt_scaler(val_proba: np.ndarray, y_val: np.ndarray) -> dict:
    """
    Platt scaling on model probabilities.
    We learn A,B such that: p_cal = sigmoid(A * logit(p) + B)
    """
    from sklearn.linear_model import LogisticRegression

    eps = 1e-6
    p = np.clip(val_proba.astype(float), eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)

    lr = LogisticRegression(solver="lbfgs")
    lr.fit(logit, y_val.astype(int))

    A = float(lr.coef_[0][0])
    B = float(lr.intercept_[0])
    return {"method": "platt", "A": A, "B": B}


def apply_platt(cal: dict, proba: np.ndarray) -> np.ndarray:
    eps = 1e-6
    p = np.clip(proba.astype(float), eps, 1 - eps)
    logit = np.log(p / (1 - p))
    z = (cal["A"] * logit) + cal["B"]
    return _sigmoid(z)


def choose_threshold(strategy: str, y_val: np.ndarray, proba_val: np.ndarray, fixed: float = 0.5) -> tuple[float, dict]:
    from sklearn.metrics import f1_score, roc_curve

    if strategy == "fixed":
        thr = float(fixed)
        return thr, {"strategy": "fixed", "threshold": thr}

    if strategy == "youden":
        fpr, tpr, thresholds = roc_curve(y_val, proba_val)
        j = tpr - fpr
        ix = int(np.nanargmax(j))
        thr = float(thresholds[ix])
        return thr, {"strategy": "youden", "threshold": thr, "youden_j": float(j[ix])}

    if strategy == "f1":
        thresholds = np.linspace(0.01, 0.99, 99)
        scores = []
        for t in thresholds:
            scores.append(f1_score(y_val, (proba_val >= t).astype(int), zero_division=0))
        ix = int(np.nanargmax(scores))
        thr = float(thresholds[ix])
        return thr, {"strategy": "f1", "threshold": thr, "f1": float(scores[ix])}

    if strategy == "f05":
        from sklearn.metrics import fbeta_score
        thresholds = np.linspace(0.01, 0.99, 99)
        scores = []
        for t in thresholds:
            scores.append(fbeta_score(y_val, (proba_val >= t).astype(int), beta=0.5, zero_division=0))
        ix = int(np.nanargmax(scores))
        thr = float(thresholds[ix])
        return thr, {"strategy": "f05", "threshold": thr, "f05": float(scores[ix])}

    raise ValueError(f"Unknown threshold strategy: {strategy}")


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    model_path: Path,
    feature_names: list[str],
    *,
    imbalance: str,
    threshold_strategy: str,
    fixed_threshold: float,
    calibrate: str,
    calibration_path: Path,
    feature_audit_path: Path,
) -> dict:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, brier_score_loss,
        classification_report, confusion_matrix, f1_score, precision_score, recall_score,
    )

    X_arr = X.values.astype(np.float32)
    y_arr = y.values.astype(int)

    # Split: train / val / test (no leakage)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_arr, y_arr, test_size=0.15, stratify=y_arr, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176470588, stratify=y_temp, random_state=42
    )  # 0.17647 * 0.85 ~= 0.15
    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    # Feature readiness audit (from processed data, before any resampling)
    feature_audit = build_feature_readiness_audit(pd.DataFrame(X_arr, columns=feature_names))
    feature_audit_path.write_text(json.dumps(feature_audit, indent=2))

    # Imbalance handling is explicit and auditable
    X_train_fit, y_train_fit = X_train, y_train
    scale_pos = float(params.pop("scale_pos_weight", 1.0))
    if imbalance == "scale_pos_weight":
        # Recommend this for tabular + XGBoost: no synthetic samples, just reweight positives
        scale_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        print(f"  Imbalance: scale_pos_weight={scale_pos:.3f} (no SMOTE)")
    elif imbalance == "smote":
        try:
            from imblearn.over_sampling import SMOTE  # noqa: PLC0415
        except ImportError:
            raise ImportError("imbalanced-learn is required for --imbalance smote. Install: pip install imbalanced-learn")
        sm = SMOTE(random_state=42)
        X_train_fit, y_train_fit = sm.fit_resample(X_train, y_train)
        scale_pos = 1.0
        print(f"  Imbalance: SMOTE applied -> {len(X_train_fit):,} samples (scale_pos_weight=1.0)")
    else:
        raise ValueError(f"Unknown imbalance mode: {imbalance}")

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

    print("  Training final XGBoost model ...")
    model.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Threshold selection (on VAL), then single evaluation on TEST
    proba_val = model.predict_proba(X_val)[:, 1]

    calibration: dict | None = None
    if calibrate == "platt":
        calibration = fit_platt_scaler(proba_val, y_val)
        calibration_path.write_text(json.dumps(calibration, indent=2))
        proba_val_used = apply_platt(calibration, proba_val)
        print(f"  Calibration: Platt scaling saved -> {calibration_path}")
    elif calibrate == "none":
        proba_val_used = proba_val
    else:
        raise ValueError(f"Unknown calibration mode: {calibrate}")

    threshold, threshold_details = choose_threshold(
        threshold_strategy, y_val, proba_val_used, fixed=fixed_threshold
    )
    print(f"  Threshold selected on VAL: {threshold:.4f} ({threshold_details['strategy']})")

    proba_test = model.predict_proba(X_test)[:, 1]
    if calibration is not None:
        proba_test_used = apply_platt(calibration, proba_test)
    else:
        proba_test_used = proba_test

    preds = (proba_test_used >= threshold).astype(int)

    auc    = roc_auc_score(y_test, proba_test_used)
    ap     = average_precision_score(y_test, proba_test_used)
    brier  = brier_score_loss(y_test, proba_test_used)

    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=["No Default", "Default"])

    metrics = {
        "roc_auc":  round(auc, 4),
        "avg_precision": round(ap, 4),
        "brier_score": round(brier, 4),
        "n_train": int(len(X_train_fit)),
        "n_val": int(len(X_val)),
        "n_test": len(X_test),
        "default_rate_test": round(float(y_test.mean()), 4),
        "confusion_matrix": cm.tolist(),
        "threshold": float(round(threshold, 6)),
        "threshold_details": threshold_details,
        "precision": float(round(precision_score(y_test, preds, zero_division=0), 4)),
        "recall": float(round(recall_score(y_test, preds, zero_division=0), 4)),
        "f1": float(round(f1_score(y_test, preds, zero_division=0), 4)),
        "imbalance_mode": imbalance,
        "calibration": calibration["method"] if calibration else "none",
    }

    print(f"\n  -- Evaluation --------------------------------------------")
    print(f"  ROC-AUC            : {auc:.4f}")
    print(f"  Avg Precision (AP) : {ap:.4f}")
    print(f"  Brier Score        : {brier:.4f}  (lower = better)")
    print(f"  Threshold          : {threshold:.4f}")
    print(f"  Precision/Recall/F1: {metrics['precision']:.4f} / {metrics['recall']:.4f} / {metrics['f1']:.4f}")
    print(f"\n{report}")

    # Feature importance
    importance = dict(zip(feature_names, [float(x) for x in model.feature_importances_]))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    print("  -- Top 10 Feature Importances ----------------------------")
    for feat, imp in list(importance_sorted.items())[:10]:
        bar = "*" * int(imp * 200)
        print(f"  {feat:<40} ({imp:.4f})  {bar}")

    # Save model (Booster JSON — InferenceEngine compatible)
    booster = model.get_booster()
    booster.save_model(str(model_path))
    print(f"\n  Success: Model saved -> {model_path}")

    # Save feature importance
    fi_path = model_path.parent / "feature_importance.json"
    fi_path.write_text(json.dumps(importance_sorted, indent=2))

    # Save training report
    report_path = model_path.parent / "training_report.txt"
    with open(report_path, "w") as f:
        f.write(f"PDR Pipeline - Layer 3 Training Report\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Calibration Path: {calibration_path if (calibration is not None) else None}\n")
        f.write(f"Feature Readiness Audit: {feature_audit_path}\n")
        f.write(f"Features: {len(feature_names)}\n\n")
        f.write(f"Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nClassification Report:\n{report}\n")
        f.write(f"\nFeature Importances (top 20):\n")
        for feat, imp in list(importance_sorted.items())[:20]:
            f.write(f"  {feat}: {imp:.6f}\n")
    print(f"  Success: Report saved -> {report_path}")

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
    parser.add_argument("--imbalance", choices=["scale_pos_weight", "smote"], default="scale_pos_weight",
                        help="Imbalance handling strategy (default: scale_pos_weight)")
    parser.add_argument("--threshold-strategy", choices=["youden", "f1", "f05", "fixed"], default="f05",
                        help="How to choose classification threshold (chosen on VAL; default: f05)")
    parser.add_argument("--fixed-threshold", type=float, default=0.5,
                        help="Used only when --threshold-strategy fixed")
    parser.add_argument("--calibrate", choices=["none", "platt"], default="platt",
                        help="Optional probability calibration saved alongside the model (default: platt)")
    parser.add_argument("--calibration-output", type=str, default=str(DEFAULT_CALIBRATION_PATH),
                        help="Output path for calibration JSON (if enabled)")
    parser.add_argument("--feature-audit-output", type=str, default=str(DEFAULT_FEATURE_AUDIT_PATH),
                        help="Output path for feature readiness audit JSON")
    args = parser.parse_args()

    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path = Path(args.calibration_output)
    feature_audit_path = Path(args.feature_audit_output)

    try:
        import xgboost as xgb  # noqa: F401
        from sklearn.model_selection import train_test_split  # noqa: F401
    except ImportError as e:
        print(f"Error: Missing dependency: {e}")
        print("   Install:  pip install xgboost scikit-learn")
        sys.exit(1)

    print("\n" + "=" * 62)
    print("  PDR Pipeline - Layer 3 XGBoost Training")
    print("=" * 62)

    print("\n[1/4] Loading preprocessed features ...")
    X, y = load_data()
    feature_names = list(X.columns)

    print("\n[2/4] Cleaning features ...")
    X = clean_features(X)

    print("\n[3/4] Hyperparameter search ...")
    if args.no_tune:
        print("  Using default hyperparameters (--no-tune flag set).")
        params = default_hyperparams()
    else:
        # Tune uses an explicit train/val split (no test leakage)
        from sklearn.model_selection import train_test_split  # noqa: PLC0415
        X_arr = X.values.astype(np.float32)
        y_arr = y.values.astype(int)
        X_temp, X_holdout, y_temp, y_holdout = train_test_split(
            X_arr, y_arr, test_size=0.30, stratify=y_arr, random_state=42
        )
        X_train_t, X_val_t, y_train_t, y_val_t = train_test_split(
            X_temp, y_temp, test_size=0.30, stratify=y_temp, random_state=42
        )
        scale_pos = float((y_train_t == 0).sum() / max((y_train_t == 1).sum(), 1)) if args.imbalance == "scale_pos_weight" else 1.0
        params = tune_hyperparams(X_train_t, y_train_t, X_val_t, y_val_t, n_trials=args.n_trials, scale_pos_weight=scale_pos)

    print("\n[4/4] Training & Evaluation ...")
    metrics = train_and_evaluate(
        X, y, params, model_path, feature_names,
        imbalance=args.imbalance,
        threshold_strategy=args.threshold_strategy,
        fixed_threshold=args.fixed_threshold,
        calibrate=args.calibrate,
        calibration_path=calibration_path,
        feature_audit_path=feature_audit_path,
    )

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
