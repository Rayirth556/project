import unittest
import numpy as np
import pandas as pd
import sys
import os

# Ensure the root project directory is in the python path for relative imports during tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pdr_pipeline.layer_4_model_risk_management import ModelRiskManager

class TestModelRiskManager(unittest.TestCase):

    def setUp(self):
        self.manager = ModelRiskManager(model_id="Test_Model", version="0.1")

    def test_governance_compliant(self):
        self.manager.register_model_metadata({
            "owner": "test",
            "approval_ticket": "TKT-123",
            "business_unit": "testing",
            "training_window": "2023"
        })
        gov = self.manager.review_governance()
        self.assertEqual(gov["status"], "COMPLIANT")

    def test_governance_non_compliant(self):
        self.manager.register_model_metadata({
            "owner": "test",
            # Missing approval_ticket, etc.
        })
        gov = self.manager.review_governance()
        self.assertEqual(gov["status"], "NON_COMPLIANT")
        self.assertTrue("approval_ticket" in gov["missing_requirements"])

    def test_payload_validation(self):
        self.manager.register_model_metadata({
            "expected_features": ["feat1", "feat2"]
        })
        
        # Test exact match
        res_pass = self.manager.validate_feature_payload({"feat1": 1.0, "feat2": 2.0})
        self.assertEqual(res_pass["status"], "PASS")
        
        # Test missing feature
        res_fail = self.manager.validate_feature_payload({"feat1": 1.0})
        self.assertEqual(res_fail["status"], "FAIL")
        self.assertIn("feat2", res_fail["missing_features"])
        
        # Test unexpected feature
        res_extra = self.manager.validate_feature_payload({"feat1": 1.0, "feat2": 2.0, "featExtra": 3.0})
        # Should still PASS but have warnings
        self.assertEqual(res_extra["status"], "PASS")
        self.assertIn("featExtra", res_extra["unexpected_features"])

    def test_drift_detection(self):
        # Generate enough reference data points for bins (e.g., normal distribution)
        np.random.seed(42)
        df_base = pd.DataFrame({"score": np.random.normal(0.5, 0.1, 1000)})
        self.manager.set_reference_baseline(df_base)
        
        # No drift (same exact data)
        df_cur_stable = df_base.copy()
        res_stable = self.manager.detect_drift(df_cur_stable, "score")
        self.assertFalse(res_stable["drift_detected"])
        self.assertIn("drift_metric", res_stable)
        self.assertEqual(res_stable["drift_metric"], "PSI")
        
        # Severe drift (shifted mean by 3 standard deviations)
        df_cur_drift = pd.DataFrame({"score": np.random.normal(0.8, 0.1, 1000)})
        res_drift = self.manager.detect_drift(df_cur_drift, "score")
        self.assertTrue(res_drift["drift_detected"])
        self.assertGreater(res_drift["metric_value"], 0.2)

    def test_performance_evaluation(self):
        preds = [0.9, 0.1, 0.8, 0.4]
        labels = [1.0, 0.0, 1.0, 1.0] # Last one is a false negative at 0.5 threshold
        
        res = self.manager.evaluate_performance(preds, labels, threshold=0.5)
        # 3 out of 4 correct -> 0.75 accuracy
        self.assertEqual(res["accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
