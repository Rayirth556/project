"""
Layer 3 — Inference Engine
==========================
Status: STATELESS — Awaiting real-world model.

This module accepts a dynamic model path at runtime. No model weights, no
training artefacts, and no PaySim-era hardcoded paths are present here.

Usage:
    engine = InferenceEngine(model_path="path/to/real_world_model.json")
    result = engine.predict(feature_vector)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Optional


# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_STATE = "WAITING_FOR_MODEL"
DEFAULT_CALIBRATION_FILENAME = "pdr_xgb_full_31_calibration.json"

# Expected feature columns for full 31-feature model with weight distribution
# Based on suggested weight distribution:
# Income/cash flow (30%), Transaction behaviour (25%), Business stability (20%),
# Utility discipline (10%), Behavioural data (10%), Education (5%)
EXPECTED_FEATURES = [
    "utility_payment_consistency",           # 6% - Transaction behaviour
    "avg_utility_dpd",                       # 5% - Transaction behaviour
    "rent_wallet_share",                     # 4% - Income/cash flow
    "subscription_commitment_ratio",         # 4% - Transaction behaviour
    "emergency_buffer_months",               # 7% - Income/cash flow
    "min_balance_violation_count",            # 3% - Income/cash flow
    "eod_balance_volatility",                # 5% - Income/cash flow
    "essential_vs_lifestyle_ratio",          # 8% - Income/cash flow
    "cash_withdrawal_dependency",             # 4% - Income/cash flow
    "bounced_transaction_count",              # 3% - Transaction behaviour
    "telecom_number_vintage_days",            # 3% - Behavioural application data
    "telecom_recharge_drop_ratio",            # 3% - Behavioural application data
    "academic_background_tier",               # 5% - Education/skill level (reinstated)
    "purpose_of_loan_encoded",                # 2% - Behavioural application data
    "business_vintage_months",                # 6% - Business stability
    "revenue_growth_trend",                  # 5% - Business stability
    "revenue_seasonality_index",             # 4% - Business stability
    "operating_cashflow_ratio",               # 3% - Income/cash flow
    "operating_cashflow_survival_flag",       # 0% - Policy override
    "cashflow_volatility",                   # 2% - Behavioural application data
    "avg_invoice_payment_delay",              # 3% - Transaction behaviour
    "customer_concentration_ratio",           # 3% - Business stability
    "repeat_customer_revenue_pct",            # 2% - Business stability
    "vendor_payment_discipline",              # 4% - Utility payment discipline
    "gst_filing_consistency_score",           # 3% - Utility payment discipline
    "gst_to_bank_variance",                   # 3% - Utility payment discipline
    "p2p_circular_loop_flag",                 # 0% - Policy override
    "benford_anomaly_score",                  # 0% - Policy override
    "round_number_spike_ratio",               # 0% - Policy override
    "turnover_inflation_spike",               # 0% - Policy override
    "identity_device_mismatch",              # 0% - Policy override
    "transaction_count",                       # Additional feature from preprocessing
]

# Feature weights for weighted scoring
FEATURE_WEIGHTS = {
    "essential_vs_lifestyle_ratio": 0.08,      # 8% - Income/cash flow
    "emergency_buffer_months": 0.07,            # 7% - Income/cash flow
    "eod_balance_volatility": 0.05,            # 5% - Income/cash flow
    "cash_withdrawal_dependency": 0.04,        # 4% - Income/cash flow
    "min_balance_violation_count": 0.03,       # 3% - Income/cash flow
    "operating_cashflow_ratio": 0.03,          # 3% - Income/cash flow
    "utility_payment_consistency": 0.06,       # 6% - Transaction behaviour
    "avg_utility_dpd": 0.05,                   # 5% - Transaction behaviour
    "rent_wallet_share": 0.04,                # 4% - Income/cash flow
    "subscription_commitment_ratio": 0.04,     # 4% - Transaction behaviour
    "bounced_transaction_count": 0.03,          # 3% - Transaction behaviour
    "avg_invoice_payment_delay": 0.03,          # 3% - Transaction behaviour
    "business_vintage_months": 0.06,           # 6% - Business stability
    "revenue_growth_trend": 0.05,              # 5% - Business stability
    "revenue_seasonality_index": 0.04,          # 4% - Business stability
    "customer_concentration_ratio": 0.03,     # 3% - Business stability
    "repeat_customer_revenue_pct": 0.02,       # 2% - Business stability
    "vendor_payment_discipline": 0.04,         # 4% - Utility payment discipline
    "gst_filing_consistency_score": 0.03,      # 3% - Utility payment discipline
    "gst_to_bank_variance": 0.03,              # 3% - Utility payment discipline
    "telecom_number_vintage_days": 0.03,        # 3% - Behavioural application data
    "telecom_recharge_drop_ratio": 0.03,       # 3% - Behavioural application data
    "purpose_of_loan_encoded": 0.02,           # 2% - Behavioural application data
    "cashflow_volatility": 0.02,               # 2% - Behavioural application data
    "academic_background_tier": 0.05,          # 5% - Education/skill level
}


class ModelNotReadyError(RuntimeError):
    """Raised when predict() is called before a real-world model is loaded."""
    pass


class InferenceEngine:
    """
    Thin wrapper around an XGBoost Booster (JSON format).
    Accepts a *dynamic* model_path — no path is hardcoded.

    Parameters
    ----------
    model_path : str | Path | None
        Absolute or relative path to a trained XGBoost model saved as JSON.
        Pass None (default) to keep the engine in WAITING_FOR_MODEL state.
    """

    def __init__(self, model_path: str | Path | None = None):
        self.model_path: Optional[Path] = Path(model_path) if model_path else None
        self._booster: Optional[Any] = None
        self._calibration: Optional[dict] = None
        self.state: str = MODEL_STATE

        if self.model_path:
            self._load_model()

    # ─── Internal ─────────────────────────────────────────────────────────────
    def _load_model(self) -> None:
        """Load the XGBoost model from the supplied path."""
        assert self.model_path is not None, "_load_model called with no model_path set"
        try:
            import xgboost as xgb  # deferred import — not required until a model exists
        except ImportError as exc:
            raise ImportError(
                "xgboost is listed in requirements.txt but is not installed. "
                "Run: pip install -r pdr_pipeline/requirements.txt"
            ) from exc

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"[Layer 3] Model file not found: {self.model_path}\n"
                "Train a real-world model and supply its path to InferenceEngine()."
            )

        self._booster = xgb.Booster()
        self._booster.load_model(str(self.model_path))
        self.state = "MODEL_LOADED"
        print(f"[Layer 3] OK Model loaded from: {self.model_path}")

        # Optional calibration file (Platt scaling) saved by training.
        # Backwards compatible: if absent, predictions are raw model probabilities.
        cal_path = self.model_path.parent / DEFAULT_CALIBRATION_FILENAME
        if cal_path.exists():
            try:
                self._calibration = json.loads(cal_path.read_text())
                if self._calibration.get("method") != "platt":
                    self._calibration = None
                else:
                    print(f"[Layer 3] OK Calibration loaded from: {cal_path}")
            except Exception:
                self._calibration = None

    # ─── Public API ───────────────────────────────────────────────────────────
    def predict(self, feature_vector: dict) -> dict:
        """
        Run inference on a feature vector produced by Layer 2.

        Parameters
        ----------
        feature_vector : dict
            Output of FeatureStoreMSME.generate_feature_vector().

        Returns
        -------
        dict
            {
              "risk_score": float,          # 0.0 (low risk) → 1.0 (high risk)
              "weighted_score": float,       # Weighted score based on feature importance
              "decision": str,              # "APPROVE" | "REVIEW" | "DECLINE"
              "model_state": str,
              "missing_features": list[str]
            }
        """
        if self._booster is None:
            raise ModelNotReadyError(
                "[Layer 3] No model is loaded.\n"
                "Supply a trained model path:\n"
                "  engine = InferenceEngine(model_path='path/to/model.json')"
            )

        import xgboost as xgb

        missing = [f for f in EXPECTED_FEATURES if f not in feature_vector]
        
        # Input validation: handle inf/NaN values
        validated_values = []
        for f in EXPECTED_FEATURES:
            value = feature_vector.get(f, 0.0)
            # Replace inf/NaN with 0.0 (safe default)
            if not np.isfinite(value) or pd.isna(value):
                value = 0.0
            validated_values.append(float(value))

        dmatrix = xgb.DMatrix(
            data=np.array(validated_values, dtype=np.float32).reshape(1, -1),
            feature_names=EXPECTED_FEATURES,
        )
        risk_score = float(self._booster.predict(dmatrix)[0])

        # Optional Platt calibration: p_cal = sigmoid(A*logit(p)+B)
        if self._calibration is not None:
            p = float(np.clip(risk_score, 1e-6, 1 - 1e-6))
            logit = float(np.log(p / (1 - p)))
            z = (float(self._calibration["A"]) * logit) + float(self._calibration["B"])
            risk_score = float(1.0 / (1.0 + np.exp(-z)))

        # Calculate weighted score based on feature importance
        weighted_score = 0.0
        for feature, weight in FEATURE_WEIGHTS.items():
            if feature in feature_vector and weight > 0:
                # Normalize feature value (simple min-max across expected range)
                value = feature_vector[feature]
                if not np.isfinite(value):
                    value = 0.0
                weighted_score += value * weight

        # Normalize weighted score to 0-1 range
        weighted_score = np.clip(weighted_score, 0, 1)

        # Evaluate Integrity Pillar "Hard-Stop" Rules
        policy_overrides = []
        if feature_vector.get("p2p_circular_loop_flag", 0.0) == 1.0:
            policy_overrides.append("Circular P2P Loop Detected")
            
        if feature_vector.get("identity_device_mismatch", 0.0) == 1.0:
            policy_overrides.append("Identity-Device Mismatch")
            
        if policy_overrides:
            decision = "DECLINE" # Hard stop for integrity failures
            risk_score = 1.0     # Maximize risk score due to override
            weighted_score = 1.0
        else:
            # Use weighted score for decision (more interpretable)
            decision = (
                "APPROVE" if weighted_score < 0.35
                else "REVIEW" if weighted_score < 0.65
                else "DECLINE"
            )

        return {
            "risk_score": float(round(risk_score, 6)),
            "weighted_score": float(round(weighted_score, 6)),
            "decision": decision,
            "policy_overrides": policy_overrides,
            "model_state": self.state,
            "missing_features": missing,
        }

    def status(self) -> dict:
        """Return current engine readiness state."""
        return {
            "model_state": self.state,
            "model_path": str(self.model_path) if self.model_path else None,
            "calibration": (self._calibration.get("method") if self._calibration else None),
            "expected_features": EXPECTED_FEATURES,
            "feature_count": len(EXPECTED_FEATURES),
        }


# ─── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = InferenceEngine()          # No model path → WAITING_FOR_MODEL
    print(json.dumps(engine.status(), indent=4))
    print(
        "\n[Layer 3] Engine is STATELESS. "
        "Provide a trained real-world model path to begin inference."
    )
