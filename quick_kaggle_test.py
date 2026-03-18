"""
quick_kaggle_test.py
==================
Quick test script for Kaggle dataset with automatic column detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from pdr_pipeline.layer_3_inference_engine import InferenceEngine
from pdr_pipeline.layer_2_feature_engine import FeatureStoreMSME

def auto_detect_columns(df):
    """Automatically detect and map columns"""
    print("Auto-detecting columns...")
    print(f"Available columns: {list(df.columns)}")
    
    # Column mapping patterns
    mappings = {
        'Amount': ['amount', 'value', 'transaction_amount', 'sum'],
        'Date': ['date', 'timestamp', 'transaction_date', 'time'],
        'Transaction_Type': ['type', 'transaction_type', 'direction', 'category'],
        'Category': ['category', 'description', 'merchant', 'narration'],
        'Balance': ['balance', 'account_balance', 'running_balance']
    }
    
    # Create mapping
    column_map = {}
    for target_col, patterns in mappings.items():
        for pattern in patterns:
            matches = [col for col in df.columns if pattern.lower() in col.lower()]
            if matches and target_col not in column_map:
                column_map[target_col] = matches[0]
                print(f"  {target_col} <- {matches[0]}")
                break
    
    return column_map

def quick_test_kaggle(csv_path):
    """Quick test with Kaggle dataset"""
    print("=" * 50)
    print("QUICK KAGGLE DATASET TEST")
    print("=" * 50)
    
    try:
        # Load data
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows")
        
        # Auto-detect columns
        column_map = auto_detect_columns(df)
        
        # Apply mapping
        df = df.rename(columns={v: k for k, v in column_map.items()})
        
        # Basic preprocessing
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        if 'Transaction_Type' in df.columns:
            df['Transaction_Type'] = df['Transaction_Type'].astype(str).str.upper()
            # Simple mapping
            df['Transaction_Type'] = df['Transaction_Type'].apply(
                lambda x: 'CREDIT' if any(word in x.upper() for word in ['CREDIT', 'DEPOSIT', 'INCOME', 'RECEIVE']) 
                else 'DEBIT' if any(word in x.upper() for word in ['DEBIT', 'WITHDRAW', 'PAYMENT', 'SPEND'])
                else 'DEBIT'  # Default
            )
        
        # Add balance if missing
        if 'Balance' not in df.columns and 'Amount' in df.columns:
            df = df.sort_values('Date' if 'Date' in df.columns else df.index)
            df['Balance'] = df['Amount'].cumsum()
        
        # Sample for quick test
        if len(df) > 1000:
            df_sample = df.sample(1000, random_state=42)
        else:
            df_sample = df
        
        print(f"Testing with {len(df_sample)} transactions")
        
        # Create UI data
        ui_data = {
            'avg_utility_dpd': 2.0,
            'telecom_number_vintage_days': 365,
            'telecom_recharge_drop_ratio': 0.8,
            'purpose_of_loan_encoded': 1,
            'business_vintage_months': 24,
            'revenue_growth_trend': 0.05,
            'revenue_seasonality_index': 0.15,
            'operating_cashflow_ratio': 1.2,
            'operating_cashflow_survival_flag': 1,
            'cashflow_volatility': 15000,
            'avg_invoice_payment_delay': 15,
            'customer_concentration_ratio': 0.7,
            'repeat_customer_revenue_pct': 0.8,
            'vendor_payment_discipline': 30,
            'gst_filing_consistency_score': 11,
            'gst_to_bank_variance': 0.3,
            'p2p_circular_loop_flag': 0,
            'benford_anomaly_score': 0.6,
            'round_number_spike_ratio': 0.02,
            'turnover_inflation_spike': 0,
            'identity_device_mismatch': 0
        }
        
        # Test pipeline
        try:
            feature_store = FeatureStoreMSME(df_sample, ui_data)
            feature_vector = feature_store.generate_feature_vector()
            
            engine = InferenceEngine("models/pdr_xgb_clean.json")
            result = engine.predict(feature_vector)
            
            print(f"\nSUCCESS!")
            print(f"Risk Score: {result['risk_score']:.4f}")
            print(f"Decision: {result['decision']}")
            print(f"Features Generated: {len(feature_vector)}")
            print(f"Missing Features: {len(result['missing_features'])}")
            
            return True
            
        except Exception as e:
            print(f"Pipeline Error: {e}")
            return False
            
    except Exception as e:
        print(f"Data Loading Error: {e}")
        return False

def main():
    # Look for common dataset names
    possible_names = [
        'transactions_data.csv',
        'transactions.csv', 
        'financial_data.csv',
        'credit_data.csv'
    ]
    
    dataset_found = False
    for name in possible_names:
        if Path(name).exists():
            print(f"Found dataset: {name}")
            success = quick_test_kaggle(name)
            dataset_found = True
            break
    
    if not dataset_found:
        print("No dataset found!")
        print("\nTo test with Kaggle data:")
        print("1. Download from: https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets")
        print("2. Save as 'transactions_data.csv' in this folder")
        print("3. Run this script again")

if __name__ == "__main__":
    main()
