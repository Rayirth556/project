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
from pathlib import Path
from typing import Any, Optional


# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_STATE = "WAITING_FOR_MODEL"

# Expected feature columns produced by Layer 2 FeatureStoreMSME.
# Update this list when the real-world model is trained.
EXPECTED_FEATURES = [
    "utility_payment_consistency",
    "avg_utility_dpd",
    "rent_wallet_share",
    "subscription_commitment_ratio",
    "emergency_buffer_months",
    "min_balance_violation_count",
    "eod_balance_volatility",
    "essential_vs_lifestyle_ratio",
    "cash_withdrawal_dependency",
    "bounced_transaction_count",
    "telecom_number_vintage_days",
    "telecom_recharge_drop_ratio",
    "academic_background_tier",
    "purpose_of_loan_encoded",
    "business_vintage_months",
    "revenue_growth_trend",
    "revenue_seasonality_index",
    "operating_cashflow_ratio",
    "operating_cashflow_survival_flag",
    "cashflow_volatility",
    "avg_invoice_payment_delay",
    "customer_concentration_ratio",
    "repeat_customer_revenue_pct",
    "vendor_payment_discipline",
    "gst_filing_consistency_score",
    "gst_to_bank_variance",
    "p2p_circular_loop_flag",
    "benford_anomaly_score",
    "round_number_spike_ratio",
    "turnover_inflation_spike",
    "identity_device_mismatch",
]


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
        print(f"[Layer 3] ✓ Model loaded from: {self.model_path}")

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
        values = [feature_vector.get(f, 0.0) for f in EXPECTED_FEATURES]

        dmatrix = xgb.DMatrix(
            data=np.array(values, dtype=np.float32).reshape(1, -1),
            feature_names=EXPECTED_FEATURES,
        )
        risk_score = float(self._booster.predict(dmatrix)[0])

        decision = (
            "APPROVE" if risk_score < 0.35
            else "REVIEW" if risk_score < 0.65
            else "DECLINE"
        )

        return {
            "risk_score": float(round(risk_score, 6)),
            "decision": decision,
            "model_state": self.state,
            "missing_features": missing,
        }

    def status(self) -> dict:
        """Return current engine readiness state."""
        return {
            "model_state": self.state,
            "model_path": str(self.model_path) if self.model_path else None,
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
