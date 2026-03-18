"""
solidify_layer_3.py
==================
Comprehensive validation and hardening of Layer 3 Inference Engine.
Ensures production readiness with robust error handling, validation, and monitoring.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Optional, Dict, List
import logging
from datetime import datetime

# Import our clean inference engine
from pdr_pipeline.layer_3_inference_engine import InferenceEngine

class Layer3Validator:
    """Comprehensive validation suite for Layer 3 Inference Engine"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.test_results = []
        
    def _setup_logger(self):
        logger = logging.getLogger("Layer3Validator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def validate_model_loading(self, model_path: str) -> bool:
        """Test model loading and basic functionality"""
        try:
            self.logger.info(f"Testing model loading: {model_path}")
            
            engine = InferenceEngine(model_path)
            status = engine.status()
            
            # Check model state
            assert status['model_state'] == 'MODEL_LOADED', f"Model not loaded: {status['model_state']}"
            assert status['feature_count'] == 28, f"Expected 28 features, got {status['feature_count']}"
            
            self.test_results.append({
                'test': 'model_loading',
                'status': 'PASS',
                'details': f"Model loaded with {status['feature_count']} features"
            })
            
            self.logger.info("✓ Model loading validation passed")
            return True
            
        except Exception as e:
            self.test_results.append({
                'test': 'model_loading',
                'status': 'FAIL',
                'details': str(e)
            })
            self.logger.error(f"✗ Model loading validation failed: {e}")
            return False
    
    def validate_feature_compatibility(self, model_path: str) -> bool:
        """Test feature compatibility between Layer 2 and Layer 3"""
        try:
            self.logger.info("Testing feature compatibility")
            
            engine = InferenceEngine(model_path)
            expected_features = engine.status()['expected_features']
            
            # Create a minimal valid feature vector
            feature_vector = {feat: 0.0 for feat in expected_features}
            
            # Test prediction
            result = engine.predict(feature_vector)
            
            # Check result structure
            required_keys = ['risk_score', 'decision', 'policy_overrides', 'model_state', 'missing_features']
            for key in required_keys:
                assert key in result, f"Missing result key: {key}"
            
            # Check data types and ranges
            assert isinstance(result['risk_score'], (int, float)), "Risk score should be numeric"
            assert 0.0 <= result['risk_score'] <= 1.0, "Risk score should be between 0 and 1"
            assert result['decision'] in ['APPROVE', 'REVIEW', 'DECLINE'], f"Invalid decision: {result['decision']}"
            assert isinstance(result['policy_overrides'], list), "Policy overrides should be a list"
            assert isinstance(result['missing_features'], list), "Missing features should be a list"
            
            self.test_results.append({
                'test': 'feature_compatibility',
                'status': 'PASS',
                'details': f"All {len(expected_features)} features compatible"
            })
            
            self.logger.info("✓ Feature compatibility validation passed")
            return True
            
        except Exception as e:
            self.test_results.append({
                'test': 'feature_compatibility',
                'status': 'FAIL',
                'details': str(e)
            })
            self.logger.error(f"✗ Feature compatibility validation failed: {e}")
            return False
    
    def validate_edge_cases(self, model_path: str) -> bool:
        """Test edge cases and error handling"""
        try:
            self.logger.info("Testing edge cases")
            
            engine = InferenceEngine(model_path)
            expected_features = engine.status()['expected_features']
            
            # Test 1: Empty feature vector
            try:
                result = engine.predict({})
                assert len(result['missing_features']) == len(expected_features), "Should detect all missing features"
                self.logger.info("✓ Empty feature vector handled correctly")
            except Exception as e:
                self.logger.error(f"✗ Empty feature vector failed: {e}")
                raise
            
            # Test 2: Partial feature vector
            try:
                partial_features = {expected_features[0]: 1.0}
                result = engine.predict(partial_features)
                assert len(result['missing_features']) == len(expected_features) - 1, "Should detect missing features"
                self.logger.info("✓ Partial feature vector handled correctly")
            except Exception as e:
                self.logger.error(f"✗ Partial feature vector failed: {e}")
                raise
            
            # Test 3: Invalid values (NaN, inf)
            try:
                invalid_features = {feat: np.nan for feat in expected_features[:5]}
                invalid_features.update({feat: np.inf for feat in expected_features[5:10]})
                invalid_features.update({feat: -np.inf for feat in expected_features[10:15]})
                invalid_features.update({feat: 1.0 for feat in expected_features[15:]})
                
                result = engine.predict(invalid_features)
                # Should handle gracefully without crashing
                assert isinstance(result['risk_score'], (int, float)), "Should return numeric risk score"
                self.logger.info("✓ Invalid values handled correctly")
            except Exception as e:
                self.logger.error(f"✗ Invalid values failed: {e}")
                raise
            
            # Test 4: Policy override scenarios
            try:
                override_features = {feat: 0.0 for feat in expected_features}
                override_features['p2p_circular_loop_flag'] = 1.0
                override_features['identity_device_mismatch'] = 1.0
                
                result = engine.predict(override_features)
                assert result['decision'] == 'DECLINE', "Should decline on policy overrides"
                assert len(result['policy_overrides']) == 2, "Should detect both overrides"
                assert result['risk_score'] == 1.0, "Should max risk score on overrides"
                self.logger.info("✓ Policy overrides handled correctly")
            except Exception as e:
                self.logger.error(f"✗ Policy overrides failed: {e}")
                raise
            
            self.test_results.append({
                'test': 'edge_cases',
                'status': 'PASS',
                'details': "All edge cases handled correctly"
            })
            
            self.logger.info("✓ Edge case validation passed")
            return True
            
        except Exception as e:
            self.test_results.append({
                'test': 'edge_cases',
                'status': 'FAIL',
                'details': str(e)
            })
            self.logger.error(f"✗ Edge case validation failed: {e}")
            return False
    
    def validate_performance_consistency(self, model_path: str, n_tests: int = 100) -> bool:
        """Test performance consistency across multiple predictions"""
        try:
            self.logger.info(f"Testing performance consistency ({n_tests} predictions)")
            
            engine = InferenceEngine(model_path)
            expected_features = engine.status()['expected_features']
            
            predictions = []
            times = []
            
            for i in range(n_tests):
                # Generate random feature vector
                feature_vector = {feat: np.random.uniform(0, 1) for feat in expected_features}
                
                start_time = datetime.now()
                result = engine.predict(feature_vector)
                end_time = datetime.now()
                
                predictions.append(result['risk_score'])
                times.append((end_time - start_time).total_seconds())
            
            # Check consistency
            predictions = np.array(predictions)
            times = np.array(times)
            
            # Risk scores should be in valid range
            assert np.all(predictions >= 0.0) and np.all(predictions <= 1.0), "Risk scores out of range"
            
            # Should have some variation (not all identical)
            assert np.std(predictions) > 0.0, "Predictions should have variation"
            
            # Performance should be reasonable (< 100ms per prediction)
            assert np.mean(times) < 0.1, f"Prediction too slow: {np.mean(times):.3f}s"
            
            self.test_results.append({
                'test': 'performance_consistency',
                'status': 'PASS',
                'details': f"Mean prediction time: {np.mean(times)*1000:.2f}ms, Score std: {np.std(predictions):.4f}"
            })
            
            self.logger.info(f"✓ Performance consistency passed (mean time: {np.mean(times)*1000:.2f}ms)")
            return True
            
        except Exception as e:
            self.test_results.append({
                'test': 'performance_consistency',
                'status': 'FAIL',
                'details': str(e)
            })
            self.logger.error(f"✗ Performance consistency validation failed: {e}")
            return False
    
    def run_comprehensive_validation(self, model_path: str) -> Dict[str, Any]:
        """Run all validation tests"""
        self.logger.info("=" * 60)
        self.logger.info("LAYER 3 COMPREHENSIVE VALIDATION")
        self.logger.info("=" * 60)
        
        # Check model exists
        if not Path(model_path).exists():
            self.logger.error(f"Model not found: {model_path}")
            return {'status': 'FAIL', 'reason': 'Model not found'}
        
        # Run all tests
        tests = [
            self.validate_model_loading,
            self.validate_feature_compatibility,
            self.validate_edge_cases,
            self.validate_performance_consistency
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            if test(model_path):
                passed += 1
        
        # Generate summary
        summary = {
            'model_path': model_path,
            'timestamp': datetime.now().isoformat(),
            'tests_run': total,
            'tests_passed': passed,
            'success_rate': passed / total,
            'overall_status': 'PASS' if passed == total else 'FAIL',
            'detailed_results': self.test_results
        }
        
        self.logger.info("=" * 60)
        self.logger.info("VALIDATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Tests Passed: {passed}/{total}")
        self.logger.info(f"Success Rate: {passed/total:.1%}")
        self.logger.info(f"Overall Status: {summary['overall_status']}")
        
        if summary['overall_status'] == 'PASS':
            self.logger.info("🎉 LAYER 3 IS PRODUCTION READY!")
        else:
            self.logger.error("❌ LAYER 3 NEEDS FIXES BEFORE PRODUCTION")
        
        return summary

def main():
    """Run Layer 3 validation"""
    clean_model_path = "models/pdr_xgb_clean.json"
    
    validator = Layer3Validator()
    results = validator.run_comprehensive_validation(clean_model_path)
    
    # Save results
    results_path = Path("models/layer_3_validation_report.json")
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Validation report saved to: {results_path}")
    
    return results['overall_status'] == 'PASS'

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
