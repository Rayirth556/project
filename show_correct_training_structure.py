"""
show_correct_training_structure.py
================================
Show the correct training data structure you need.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def show_correct_structure():
    """Show the correct training data structure"""
    print("=" * 80)
    print("CORRECT TRAINING DATA STRUCTURE")
    print("=" * 80)
    
    print("""
🎯 WHAT YOU ACTUALLY NEED FOR TRAINING:

====================================

INPUT DATA STRUCTURE:
===================

For EACH user in your training dataset, you need:

1. TRANSACTION DATA (CSV upload):
   date,client_id,amount,Transaction_Type,Category,Balance
   2023-01-15,user_001,$1500.00,CREDIT,Income,15000.00
   2023-01-16,user_001,$500.00,DEBIT,Rent,14500.00
   2023-01-17,user_001,$100.00,DEBIT,Food,14400.00
   ... (50-500 transactions per user)

2. BACKGROUND DATA (UI form):
   {
     "avg_utility_dpd": 2.5,
     "telecom_number_vintage_days": 365,
     "academic_background_tier": 2,
     "business_vintage_months": 24,
     "purpose_of_loan_encoded": 1,
     ... (18 total fields)
   }

3. LABEL (Real outcome):
   {
     "defaulted": 0  # 0 = no default, 1 = default
   }

TRAINING DATASET FORMAT:
=======================
features.parquet:
├── 5,000 rows (one per user)
├── 31 columns (features)
└── Each row = one complete user profile

labels.parquet:
├── 5,000 rows (one per user)  
├── 1 column (TARGET)
└── Real default outcomes

CURRENT PROBLEM:
===============
You're training on aggregated synthetic data instead of individual user profiles.

SOLUTION:
========
1. Collect real individual user data
2. Each sample = one complete user (transactions + background + outcome)
3. Train model on individual user assessments
""")

def create_sample_structure():
    """Create a sample of the correct structure"""
    print("\n" + "=" * 80)
    print("SAMPLE CORRECT STRUCTURE")
    print("=" * 80)
    
    # Sample user data
    sample_users = []
    
    for i in range(3):
        user_id = f"user_{i+1:03d}"
        
        # Sample transactions
        transactions = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50, freq='D'),
            'client_id': user_id,
            'amount': np.random.uniform(100, 5000, 50),
            'Transaction_Type': np.random.choice(['CREDIT', 'DEBIT'], 50, p=[0.4, 0.6]),
            'Category': np.random.choice(['Income', 'Rent', 'Food', 'Utilities', 'Shopping'], 50),
            'Balance': np.cumsum(np.random.uniform(-1000, 3000, 50))
        })
        
        # Sample background data
        background = {
            'avg_utility_dpd': np.random.uniform(0, 10),
            'telecom_number_vintage_days': np.random.randint(30, 1000),
            'telecom_recharge_drop_ratio': np.random.uniform(0.5, 1.5),
            'academic_background_tier': np.random.randint(1, 4),
            'purpose_of_loan_encoded': np.random.randint(1, 4),
            'business_vintage_months': np.random.randint(3, 120),
            'revenue_growth_trend': np.random.uniform(-0.1, 0.2),
            'revenue_seasonality_index': np.random.uniform(0.1, 0.5),
            'operating_cashflow_ratio': np.random.uniform(0.5, 2.0),
            'operating_cashflow_survival_flag': np.random.randint(0, 2),
            'cashflow_volatility': np.random.uniform(5000, 50000),
            'avg_invoice_payment_delay': np.random.uniform(0, 90),
            'customer_concentration_ratio': np.random.uniform(0.3, 1.0),
            'repeat_customer_revenue_pct': np.random.uniform(0.2, 1.0),
            'vendor_payment_discipline': np.random.uniform(0, 120),
            'gst_filing_consistency_score': np.random.uniform(2, 12),
            'gst_to_bank_variance': np.random.uniform(0.1, 1.0),
            'p2p_circular_loop_flag': np.random.randint(0, 2),
            'benford_anomaly_score': np.random.uniform(0.4, 1.5),
            'round_number_spike_ratio': np.random.uniform(0, 0.2),
            'turnover_inflation_spike': np.random.randint(0, 2),
            'identity_device_mismatch': np.random.randint(0, 2)
        }
        
        # Sample label
        defaulted = np.random.choice([0, 1], p=[0.85, 0.15])
        
        sample_users.append({
            'user_id': user_id,
            'transactions': transactions,
            'background': background,
            'defaulted': defaulted
        })
    
    # Display sample structure
    print("\nSAMPLE USER PROFILES:")
    for user in sample_users:
        print(f"\n{user['user_id']}:")
        print(f"  Transactions: {len(user['transactions'])} rows")
        print(f"  Background: {len(user['background'])} fields")
        print(f"  Defaulted: {user['defaulted']}")
        print(f"  Sample transaction: {user['transactions'].iloc[0].to_dict()}")
        print(f"  Sample background: {dict(list(user['background'].items())[:3])}...")
    
    return sample_users

def show_data_flow_diagram():
    """Show the correct data flow"""
    print("\n" + "=" * 80)
    print("CORRECT DATA FLOW")
    print("=" * 80)
    
    print("""
CORRECT WORKFLOW:
================

1. DATA COLLECTION
   ├── Real users upload transaction files
   ├── Users fill background forms in UI
   └── Track actual default outcomes

2. FEATURE ENGINEERING  
   ├── For each user: process their transactions
   ├── Combine with their background data
   └── Generate 31 features per user

3. MODEL TRAINING
   ├── Input: 5,000 user profiles (31 features each)
   ├── Target: Real default outcomes
   └── Output: Model that can assess individual users

4. INFERENCE
   ├── New user uploads transactions
   ├── New user fills background form
   └── Model predicts default probability

CURRENT MISMATCH:
=================
❌ Training: 1,000 synthetic aggregated samples
❌ Inference: Individual real-time assessment

FIX NEEDED:
===========
✅ Training: Individual user profiles (like inference)
✅ Inference: Individual real-time assessment
""")

if __name__ == "__main__":
    show_correct_structure()
    create_sample_structure()
    show_data_flow_diagram()
    
    print("\n" + "=" * 80)
    print("KEY TAKEAWAY")
    print("=" * 80)
    print("You need to retrain your model on INDIVIDUAL USER DATA,")
    print("not aggregated synthetic data.")
    print("\nEach training sample should be:")
    print("1 user = their transactions + their background + their actual outcome")
    print("=" * 80)
