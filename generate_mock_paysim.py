import pandas as pd
import numpy as np
import os
import random

def generate_mock_paysim():
    print("Generating Mock PaySim Dataset for Testing...")
    np.random.seed(42)
    random.seed(42)
    
    # We need some users with >= 15 transactions
    high_volume_users = [f"C{i}_HIGH" for i in range(50)]
    low_volume_users = [f"C{i}_LOW" for i in range(200)]
    
    users = high_volume_users * 16 + low_volume_users * 3
    random.shuffle(users)
    
    data = []
    step = 1
    
    for u in users:
        txn_type = random.choice(['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
        amount = round(random.uniform(10.0, 10000.0), 2)
        
        is_fraud = 0
        
        # Make a few high volume users fraud
        if "C10_HIGH" in u or "C20_HIGH" in u:
            is_fraud = 1
            if txn_type in ['TRANSFER', 'CASH_OUT']:
                amount = round(random.uniform(50000.0, 500000.0), 2)
        
        row = {
            'step': step,
            'type': txn_type,
            'amount': amount,
            'nameOrig': u,
            'oldbalanceOrg': round(random.uniform(1000, 100000), 2),
            'newbalanceOrig': round(random.uniform(0, 50000), 2),
            'nameDest': f"M{random.randint(100, 999)}",
            'oldbalanceDest': 0.0,
            'newbalanceDest': amount if txn_type in ['TRANSFER', 'CASH_OUT'] else 0.0,
            'isFraud': is_fraud,
            'isFlaggedFraud': 0
        }
        data.append(row)
        step += random.randint(1, 3)
        
    df = pd.DataFrame(data)
    df.to_csv("PS_20174392719_1491204439457_log.csv", index=False)
    print(f"Mocked PaySim dataset created with {len(df)} rows.")

if __name__ == "__main__":
    generate_mock_paysim()
