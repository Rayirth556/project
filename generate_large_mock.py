import json
import random
from datetime import datetime, timedelta, timezone
import os

def generate_transactions(num_txns=500, days_history=365):
    transactions = []
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_history)
    
    current_balance = 100000.0
    
    # Types of MSME transactions
    revenue_categories = ["SALES-POS", "INVOICE-PAYMENT", "NEFT-CLIENT", "UPI-CUSTOMER"]
    utility_categories = ["ELECTRICITY", "WATER", "INTERNET", "OFFICE-RENT"]
    vendor_categories = ["VENDOR-PAYMENT", "SUPPLIER-RTGS", "RAW-MATERIAL"]
    other_debits = ["CASH-ATM", "EMI-LOAN", "FEE-PENALTY"]
    
    # Generate sequentially
    timestamps = [start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds()))) for _ in range(num_txns)]
    timestamps.sort()
    
    for i, ts in enumerate(timestamps):
        is_credit = random.random() < 0.35 # 35% credits, 65% debits
        
        if is_credit:
            txn_type = "CREDIT"
            amount = round(random.uniform(5000, 150000), 2)
            narration = random.choice(revenue_categories) + f"-{random.randint(1000, 9999)}"
            current_balance += amount
        else:
            txn_type = "DEBIT"
            rand_choice = random.random()
            if rand_choice < 0.4:
                amount = round(random.uniform(10000, 80000), 2)
                narration = random.choice(vendor_categories) + f"-{random.randint(1000, 9999)}"
            elif rand_choice < 0.7:
                amount = round(random.uniform(1000, 5000), 2)
                narration = random.choice(utility_categories) + f"-{random.randint(1000, 9999)}"
            else:
                amount = round(random.uniform(500, 15000), 2)
                narration = random.choice(other_debits) + f"-{random.randint(1000, 9999)}"
            current_balance -= amount
            
        transactions.append({
            "txnId": f"TXN{ts.strftime('%Y%m%d%H%M%S')}{i}",
            "type": txn_type,
            "amount": f"{amount:.2f}",
            "narration": narration,
            "valueDate": ts.strftime('%Y-%m-%d'),
            "transactionTimestamp": ts.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "currentBalance": f"{current_balance:.2f}"
        })
        
    return transactions, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


txns, start_str, end_str = generate_transactions(850, 365)

mock_data = {
  "SessionId": "897be985-7a1b-4749-a55d-f13efcd2eff3",
  "Status": "COMPLETED",
  "FI": [
    {
      "fipId": "SETU-FIP",
      "fipName": "Setu Mock Bank",
      "Accounts": [
        {
          "maskedAccNumber": "XXXX1234",
          "type": "DEPOSIT",
          "linkedAccRef": "link-1234-abcd",
          "Data": {
            "Account": {
              "type": "SAVINGS",
              "accRefNumber": "XXXX1234",
              "Profile": {
                "Holders": {
                  "Holder": [
                    {
                      "name": "MSME Corp Pvt Ltd",
                      "dob": "2015-01-01",
                      "mobile": "9999999999",
                      "pan": "ABCDE1234F",
                      "email": "contact@msmecorp.com"
                    }
                  ]
                }
              },
              "Summary": {
                "currentBalance": txns[-1]["currentBalance"],
                "currency": "INR",
                "balanceDateTime": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                "currentODLimit": "0.00",
                "drawingLimit": "0.00",
                "status": "ACTIVE"
              },
              "Transactions": {
                "startDate": start_str,
                "endDate": end_str,
                "Transaction": txns
              }
            }
          }
        }
      ]
    }
  ]
}

mock_path = r"c:\Users\rayir\OneDrive\Documents\barclays_project\pdr_pipeline\layer_1_ingestion\mock_data\mock_aa_data.json"
with open(mock_path, "w") as f:
    json.dump(mock_data, f, indent=2)

print(f"Generated {len(txns)} realistic MSME transactions to mock_aa_data.json!")
