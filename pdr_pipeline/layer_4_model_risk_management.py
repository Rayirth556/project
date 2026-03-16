import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# Optional MLFlow integration
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ModelRiskManager:
    """
    Layer 4: Model Risk Management & Governance
    -------------------------------------------
    Scaffold for monitoring, validating, and governing the AI credit-risk model.
    Designed to hook into the Inference Engine (Layer 3) to evaluate inputs and outputs.
    """

    def __init__(self, model_id: str = "unknown_model", version: str = "1.0.0"):
        self.model_id = model_id
        self.version = version
        
        self.metadata: Dict[str, Any] = {}
        self.reference_baseline: Optional[pd.DataFrame] = None
        self.expected_features: List[str] = []
        
        self._setup_logger()

    def _setup_logger(self) -> None:
        self.logger = logging.getLogger(f"ModelRiskManager_{self.model_id}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    # --- 1. Registration & Configuration ---
    
    def register_model_metadata(self, metadata: Dict[str, Any]) -> None:
        """Register details about the model (owner, ticket, training window, etc.)."""
        self.metadata.update(metadata)
        
        if "expected_features" in metadata:
            self.expected_features = metadata["expected_features"]
            
        self.logger.info(f"Registered metadata for model {self.model_id} (v{self.version})")

    def set_reference_baseline(self, baseline_data: pd.DataFrame) -> None:
        """Store the baseline dataset (e.g., training data) used for drift detection later."""
        self.reference_baseline = baseline_data.copy()
        self.logger.info(f"Set reference baseline with {len(self.reference_baseline)} records.")

    # --- 2. Validation & Monitoring Methods ---

    def validate_feature_payload(self, feature_vector: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate incoming inference requests against expected schema/features.
        Called BEFORE prediction.
        """
        result = {
            "status": "PASS",
            "missing_features": [],
            "unexpected_features": [],
            "warnings": []
        }
        
        if not self.expected_features:
            result["status"] = "WARNING_NO_SCHEMA_DEFINED"
            return result

        incoming_keys = set(feature_vector.keys())
        expected_keys = set(self.expected_features)
        
        missing = expected_keys - incoming_keys
        unexpected = incoming_keys - expected_keys
        
        if missing:
            result["missing_features"] = list(missing)
            result["status"] = "FAIL"
            
        if unexpected:
            result["unexpected_features"] = list(unexpected)
            result["warnings"].append(f"Found {len(unexpected)} unexpected features.")
            
        return result

    def evaluate_performance(self, predictions: List[float], labels: List[float], threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate model performance (e.g., Accuracy, ROC-AUC setup) if ground truth is available.
        Called AFTER prediction & label maturation.
        """
        if predictions is None or labels is None:
             return {"error": -1.0, "message": "Predictions and labels cannot be None."}
             
        preds_array = np.array(predictions)
        labels_array = np.array(labels)
        
        if len(preds_array) == 0 or len(labels_array) == 0 or len(preds_array) != len(labels_array):
             return {"error": -1.0, "message": "Predictions and labels must be equal-length lists or arrays."}
             
        # Simple scikit-learn metrics placeholder (implement full metrics later)
        # Using numpy to calculate a quick accuracy based on a boolean threshold for demonstration
        preds_array = np.array(predictions)
        labels_array = np.array(labels)
        
        binary_preds = (preds_array >= threshold).astype(int)
        binary_labels = labels_array.astype(int)
        
        accuracy = float(np.mean(binary_preds == binary_labels))
        
        return {
            "accuracy": round(accuracy, 4),
            "threshold": threshold,
            "samples_evaluated": len(predictions)
        }

    def detect_drift(self, current_data: pd.DataFrame, feature_name: str, bins: int = 10, threshold: float = 0.2) -> Dict[str, Any]:
        """
        Calculate Population Stability Index (PSI) to detect distribution drift for a numeric feature.
        Requires reference baseline.
        """
        if self.reference_baseline is None or feature_name not in self.reference_baseline.columns:
            return {"status": "ERROR", "reason": "No reference baseline or feature missing."}
            
        if feature_name not in current_data.columns:
            return {"status": "ERROR", "reason": f"Feature '{feature_name}' missing in current data."}
            
        ref_data = self.reference_baseline[feature_name].dropna().values
        cur_data = current_data[feature_name].dropna().values
        
        if len(ref_data) == 0 or len(cur_data) == 0:
             return {"status": "ERROR", "reason": "Not enough valid data points to calculate PSI."}
             
        # PSI Calculation using standard bucketing
        # Create bins based on the reference distribution
        _, bin_edges = np.histogram(ref_data, bins=bins)
        
        # Expand outer edges slightly to include min/max
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        ref_counts, _ = np.histogram(ref_data, bins=bin_edges)
        cur_counts, _ = np.histogram(cur_data, bins=bin_edges)
        
        # Convert to percentages and add a small epsilon to avoid division by zero or log(0)
        epsilon = 1e-4
        ref_pct = (ref_counts / len(ref_data)) + epsilon
        cur_pct = (cur_counts / len(cur_data)) + epsilon
        
        # Calculate PSI
        psi_value = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        is_drifting = psi_value > threshold 
        
        return {
            "feature": feature_name,
            "drift_metric": "PSI",
            "metric_value": float(round(psi_value, 4)),
            "threshold": threshold,
            "drift_detected": bool(is_drifting)
        }

    def review_governance(self) -> Dict[str, str]:
        """
        Check if the model satisfies compliance/governance standards.
        """
        required_keys = ["owner", "approval_ticket", "business_unit", "training_window"]
        missing_keys = [k for k in required_keys if k not in self.metadata]
        
        if missing_keys:
            return {
                "status": "NON_COMPLIANT",
                "missing_requirements": ", ".join(missing_keys)
            }
            
        return {"status": "COMPLIANT"}

    # --- 3. Reporting & Tracking ---

    def generate_report(self) -> Dict[str, Any]:
        """Compile a summary report of the Risk Manager's current state."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_id": self.model_id,
            "version": self.version,
            "governance_status": self.review_governance().get("status"),
            "baseline_configured": self.reference_baseline is not None,
            "expected_feature_count": len(self.expected_features),
            "mlflow_integrated": MLFLOW_AVAILABLE
        }

    def log_report_to_mlflow(self) -> None:
        """
        If MLflow is available, log parameters and the generated report.
        """
        if not MLFLOW_AVAILABLE:
            self.logger.warning("MLflow not installed. Cannot log report.")
            return

        try:
            report = self.generate_report()
            with mlflow.start_run(run_name=f"Risk_Report_{self.model_id}"):
                mlflow.log_param("model_id", self.model_id)
                mlflow.log_param("version", self.version)
                mlflow.log_param("governance_status", report["governance_status"])
                # Could log dictionary as JSON artifact
                with open("risk_report.json", "w") as f:
                    json.dump(report, f)
                mlflow.log_artifact("risk_report.json")
            self.logger.info("Successfully logged risk report to MLflow.")
        except Exception as e:
            self.logger.error(f"Failed to log to MLflow: {e}")

    def status(self) -> Dict[str, Any]:
        """Return basic status of the manager itself."""
        return {"manager_state": "ACTIVE", "model_id": self.model_id}


# --- Entrypoint & Scaffold Demonstration ---
if __name__ == "__main__":
    print("====================================")
    print("LAYER 4: MODEL RISK MANAGEMENT")
    print("====================================\n")
    
    manager = ModelRiskManager(model_id="MSME_Credit_Model", version="1.0.0")
    
    # 1. Register Metadata
    manager.register_model_metadata({
        "owner": "DataScience_Team_A",
        "business_unit": "SME_Lending",
        "approval_ticket": "JIRA-9982",
        "training_window": "2023-01 to 2024-01",
        "expected_features": ["f1", "f2", "f3"]
    })
    
    # 2. Check Governance
    gov = manager.review_governance()
    print(f"Governance Check: {gov['status']}\n")
    
    # 3. Validate a payload (Missing f3, unexpected f4)
    payload = {"f1": 0.5, "f2": 1.2, "f4": 9.9}
    val_result = manager.validate_feature_payload(payload)
    print(f"Payload Validation:\n{json.dumps(val_result, indent=2)}\n")
    
    # 4. Set Baseline & Check Drift
    df_base = pd.DataFrame({"f1": [1.0, 1.1, 0.9, 1.05]})
    manager.set_reference_baseline(df_base)
    
    df_current = pd.DataFrame({"f1": [2.5, 2.6, 2.4, 2.7]}) # Clearly drifted
    drift_res = manager.detect_drift(df_current, "f1")
    print(f"Drift Detection on 'f1':\n{json.dumps(drift_res, indent=2)}\n")
    
    # 5. Generate Report
    report = manager.generate_report()
    print(f"Final Risk Report:\n{json.dumps(report, indent=2)}\n")
