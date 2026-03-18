"""
check_data_structure.py
=======================
Check the actual structure of our training data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def check_training_data():
    """Check the structure of our training data"""
    print("=" * 60)
    print("TRAINING DATA STRUCTURE ANALYSIS")
    print("=" * 60)
    
    # Load features
    features_path = Path("data/processed/features.parquet")
    if features_path.exists():
        X = pd.read_parquet(features_path)
        print(f"\nFEATURES SHAPE: {X.shape}")
        print(f"FEATURES COLUMNS: {list(X.columns)}")
        print(f"\nSAMPLE FEATURES:")
        print(X.head(3).to_string())
    else:
        print("No features.parquet found!")
    
    # Load labels
    labels_path = Path("data/processed/labels.parquet")
    if labels_path.exists():
        y = pd.read_parquet(labels_path)
        print(f"\nLABELS SHAPE: {y.shape}")
        print(f"LABELS VALUE COUNTS:")
        print(y.value_counts())
        print(f"DEFAULT RATE: {y.squeeze().mean():.1%}")
    else:
        print("No labels.parquet found!")
    
    # Check transaction data
    trans_path = Path("transactions_data.csv")
    if trans_path.exists():
        df = pd.read_csv(trans_path, nrows=5)
        print(f"\nTRANSACTION DATA COLUMNS: {list(df.columns)}")
        print(f"\nSAMPLE TRANSACTIONS:")
        print(df.to_string())
        print(f"\nTOTAL TRANSACTIONS: {pd.read_csv(trans_path).shape[0]:,}")
        print(f"UNIQUE CLIENTS: {pd.read_csv(trans_path)['client_id'].nunique():,}")

def show_data_flow():
    """Show how data flows through our pipeline"""
    print("\n" + "=" * 60)
    print("DATA FLOW ANALYSIS")
    print("=" * 60)
    
    print("""
CURRENT DATA FLOW:
==================

1. RAW TRANSACTION DATA (transactions_data.csv)
   - 13,305,915 transactions
   - 1,219 unique clients
   - Columns: id, date, client_id, card_id, amount, use_chip, merchant_id, etc.
   - Each row = 1 transaction

2. FEATURE GENERATION (preprocess_transaction_data.py)
   - Groups transactions by client_id
   - For each client: generates 31 features from their transaction history
   - Adds synthetic UI data (background info)
   - Result: 1,000 samples x 31 features

3. MODEL TRAINING (quick_train_31_model.py)
   - Trains on 1,000 client samples
   - Each sample = 1 client's aggregated features
   - Target: synthetic default labels (15% default rate)

4. INFERENCE (simplified_risk_workflow.py)
   - Takes: client transactions + UI background data
   - Generates: 31 features
   - Predicts: risk score for that client

PROBLEM IDENTIFIED:
==================
You're right! There's a mismatch:

- Training: 1,000 aggregated client samples
- Inference: Individual client assessment

The model is trained on aggregated data, but you want to assess
individual clients in real-time.

SOLUTION NEEDED:
===============
1. Real individual client transaction data
2. Real background data from UI
3. Real default labels (not synthetic)
""")

if __name__ == "__main__":
    check_training_data()
    show_data_flow()
