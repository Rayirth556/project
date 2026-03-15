import pandas as pd
# Load the data Layer 1 just created
df = pd.read_csv('pdr_pipeline/layer_1_ingestion/ingested_transactions.csv')

# 1. Timeline Check
print(f"Data Range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Total Transactions Fetched: {len(df)}")

# 2. Logic Check for Layer 2 Features
import sys
import os
sys.path.append(os.path.join(os.path.split(__file__)[0], 'pdr_pipeline'))
from layer_2_feature_engine import FeatureStoreMSME

# Add a sample UI profile (Mocking user input)
ui_data = {"gst_declared_revenue": 1000000, "total_members": 4, "earning_members": 1}

fs = FeatureStoreMSME(df, ui_data)
vector = fs.generate_feature_vector()

# Check if specific MSME features are calculating
print(f"Operating Cashflow Ratio: {vector['operating_cashflow_ratio']}")
print(f"Circular Loop Detected: {vector['p2p_circular_loop_flag']}")