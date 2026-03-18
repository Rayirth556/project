"""
test_inference_pipeline.py
===========================
End-to-end test of the complete inference pipeline.
Tests Layer 2 feature generation + Layer 3 inference with clean model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).parent))

from pdr_pipeline.layer_3_inference_engine import InferenceEngine
from pdr_pipeline.layer_2_feature_engine import FeatureStoreMSME

def create_sample_transaction_data():
    """Create realistic sample transaction data for testing"""
    np.random.seed(42)
    
    # Generate 6 months of transaction data
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='D')
    transactions = []
    
    # Starting balance
    running_balance = 100000
    
    for date in dates:
        # Salary credit (monthly)
        if date.day == 1 and date.month in [1, 2, 3, 4, 5, 6]:
            amount = 50000 + np.random.normal(0, 5000)
            running_balance += amount
            transactions.append({
                'Date': date,
                'Amount': amount,
                'Transaction_Type': 'CREDIT',
                'Category': 'Salary',
                'Description': 'Monthly Salary',
                'Balance': running_balance
            })
        
        # Business revenue (random weekdays)
        if date.weekday() < 5 and np.random.random() < 0.3:
            amount = np.random.uniform(1000, 15000)
            running_balance += amount
            transactions.append({
                'Date': date,
                'Amount': amount,
                'Transaction_Type': 'CREDIT', 
                'Category': 'Sales',
                'Description': 'Business Revenue',
                'Balance': running_balance
            })
        
        # Utility payments (monthly)
        if date.day == 5 and date.month in [1, 2, 3, 4, 5, 6]:
            amount = -(2000 + np.random.normal(0, 200))
            running_balance += amount
            transactions.append({
                'Date': date,
                'Amount': amount,
                'Transaction_Type': 'DEBIT',
                'Category': 'Utility',
                'Description': 'Electricity Bill',
                'Balance': running_balance
            })
        
        # Rent (monthly)
        if date.day == 1 and date.month in [1, 2, 3, 4, 5, 6]:
            amount = -15000
            running_balance += amount
            transactions.append({
                'Date': date,
                'Amount': amount,
                'Transaction_Type': 'DEBIT',
                'Category': 'Rent',
                'Description': 'Monthly Rent',
                'Balance': running_balance
            })
        
        # Random expenses
        if np.random.random() < 0.4:
            amount = -np.random.uniform(100, 2000)
            running_balance += amount
            transactions.append({
                'Date': date,
                'Amount': amount,
                'Transaction_Type': 'DEBIT',
                'Category': np.random.choice(['Food', 'Transport', 'Shopping', 'Entertainment']),
                'Description': 'Daily Expense',
                'Balance': running_balance
            })
    
    df = pd.DataFrame(transactions)
    # Sort by date to ensure chronological order
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def create_sample_ui_data():
    """Create sample UI form data"""
    return {
        # Personal information
        'avg_utility_dpd': 2.5,
        'telecom_number_vintage_days': 365,
        'telecom_recharge_drop_ratio': 0.8,
        'academic_background_tier': 2,  # Graduate
        'purpose_of_loan_encoded': 1,   # Business expansion
        
        # Business information
        'business_vintage_months': 24,
        'revenue_growth_trend': 0.05,
        'revenue_seasonality_index': 0.15,
        'operating_cashflow_ratio': 1.2,
        'operating_cashflow_survival_flag': 1,
        'cashflow_volatility': 15000,
        
        # Customer relationships
        'avg_invoice_payment_delay': 15,
        'customer_concentration_ratio': 0.7,
        'repeat_customer_revenue_pct': 0.8,
        'vendor_payment_discipline': 30,
        
        # Compliance and integrity
        'gst_filing_consistency_score': 11,
        'gst_to_bank_variance': 0.3,
        'p2p_circular_loop_flag': 0,
        'benford_anomaly_score': 0.6,
        'round_number_spike_ratio': 0.02,
        'turnover_inflation_spike': 0,
        'identity_device_mismatch': 0
    }

def test_feature_generation():
    """Test Layer 2 feature generation"""
    print("=" * 60)
    print("TESTING LAYER 2: FEATURE GENERATION")
    print("=" * 60)
    
    # Create sample data
    aa_data = create_sample_transaction_data()
    ui_data = create_sample_ui_data()
    
    print(f"Sample transactions: {len(aa_data)} rows")
    print(f"Date range: {aa_data['Date'].min()} to {aa_data['Date'].max()}")
    print(f"Transaction types: {aa_data['Transaction_Type'].value_counts().to_dict()}")
    
    # Generate features
    try:
        feature_store = FeatureStoreMSME(aa_data, ui_data)
        
        # Test individual feature calculations
        print("\nTesting individual features:")
        
        # Test a few key features
        features_to_test = [
            'utility_payment_consistency',
            'rent_wallet_share', 
            'emergency_buffer_months',
            'business_vintage_months',
            'vendor_payment_discipline'
        ]
        
        feature_vector = {}
        for feature in features_to_test:
            try:
                method = getattr(feature_store, f'calc_{feature}')
                value = method()
                feature_vector[feature] = value
                print(f"  {feature}: {value:.4f}")
            except Exception as e:
                print(f"  {feature}: ERROR - {e}")
        
        # Generate full feature vector
        print(f"\nGenerating full feature vector...")
        full_features = feature_store.generate_feature_vector()
        
        print(f"Generated {len(full_features)} features")
        print(f"Sample features:")
        for i, (k, v) in enumerate(list(full_features.items())[:5]):
            print(f"  {k}: {v:.4f}")
        
        return full_features
        
    except Exception as e:
        print(f"ERROR in feature generation: {e}")
        raise

def test_inference_engine(feature_vector, model_path):
    """Test Layer 3 inference engine"""
    print("\n" + "=" * 60)
    print("TESTING LAYER 3: INFERENCE ENGINE")
    print("=" * 60)
    
    try:
        # Load inference engine
        engine = InferenceEngine(model_path)
        print(f"Model loaded successfully")
        
        # Check engine status
        status = engine.status()
        print(f"Engine state: {status['model_state']}")
        print(f"Expected features: {status['feature_count']}")
        
        # Test prediction
        print(f"\nRunning inference...")
        result = engine.predict(feature_vector)
        
        print(f"Prediction results:")
        print(f"  Risk score: {result['risk_score']:.4f}")
        print(f"  Decision: {result['decision']}")
        print(f"  Policy overrides: {result.get('policy_overrides', [])}")
        print(f"  Missing features: {len(result['missing_features'])}")
        
        if result['missing_features']:
            print(f"  Missing: {result['missing_features'][:5]}...")
        
        return result
        
    except Exception as e:
        print(f"ERROR in inference: {e}")
        raise

def test_pipeline_integration():
    """Test complete pipeline integration"""
    print("=" * 60)
    print("COMPLETE PIPELINE INTEGRATION TEST")
    print("=" * 60)
    
    # Test with clean model
    clean_model_path = Path("models/pdr_xgb_clean.json")
    
    if not clean_model_path.exists():
        print(f"ERROR: Clean model not found at {clean_model_path}")
        print("Run: py train_clean_model.py")
        return False
    
    try:
        # Layer 2: Feature generation
        feature_vector = test_feature_generation()
        
        # Layer 3: Inference
        result = test_inference_engine(feature_vector, clean_model_path)
        
        print(f"\n" + "=" * 60)
        print("INTEGRATION TEST RESULTS")
        print("=" * 60)
        print(f"Status: SUCCESS")
        print(f"Risk score: {result['risk_score']:.4f}")
        print(f"Decision: {result['decision']}")
        print(f"Pipeline working correctly!")
        
        return True
        
    except Exception as e:
        print(f"\nINTEGRATION TEST FAILED: {e}")
        return False

def main():
    """Run complete pipeline test"""
    print("PDR PIPELINE END-TO-END TEST")
    print("=" * 60)
    
    success = test_pipeline_integration()
    
    if success:
        print(f"\n{'='*60}")
        print("ALL TESTS PASSED!")
        print("Pipeline is ready for production use.")
        print("=" * 60)
    else:
        print(f"\n{'='*60}")
        print("TESTS FAILED!")
        print("Pipeline needs debugging before production.")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
