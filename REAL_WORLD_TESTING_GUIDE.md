# Real-World Data Testing Guide

## Quick Start with Kaggle Dataset

### Step 1: Download the Dataset
```bash
# Option 1: Direct download (if you have Kaggle CLI)
kaggle datasets download -d computingvictor/transactions-fraud-datasets

# Option 2: Manual download
# Visit: https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets
# Click "Download" and extract to project folder
```

### Step 2: Prepare the Data
```bash
# After downloading, extract and find the transactions CSV file
# Common names: transactions_data.csv, transactions.csv, etc.

# Copy to project root
cp path/to/extracted/transactions_data.csv ./
```

### Step 3: Run the Test
```bash
py test_real_world_data.py
```

## What the Test Does

### 1. **Synthetic Realistic Data** ✅ (Always runs)
- Generates 355 realistic transactions over 6 months
- Tests 4 scenarios: Low Risk, Medium Risk, High Risk, Policy Override
- Validates feature generation and inference

### 2. **Kaggle Real Data** 🔄 (If available)
- Loads actual financial transactions
- Maps columns to our expected format
- Tests on real transaction patterns

## Expected Results

### Synthetic Tests (Current Results)
```
Low Risk:     Risk Score = 0.1971, Decision = APPROVE
Medium Risk:  Risk Score = 0.1966, Decision = APPROVE  
High Risk:    Risk Score = 0.2895, Decision = APPROVE
Policy Override: Risk Score = 1.0000, Decision = DECLINE
```

### Real Data Tests
Will show actual risk scores based on real transaction patterns.

## Troubleshooting

### Column Mapping Issues
If the Kaggle dataset has different column names, the test will try to automatically map:
- `amount` → `Amount`
- `date` → `Date`
- `type` → `Transaction_Type`
- `category` → `Category`

### Missing Balance Column
The test automatically calculates running balance if not provided.

### Data Format Issues
The test handles:
- Date format conversion
- Transaction type standardization  
- Missing values
- Invalid data types

## Next Steps

1. **Download the Kaggle dataset** using the instructions above
2. **Run the test** to see real-world performance
3. **Review results** in `models/real_world_test_results.json`
4. **Compare** synthetic vs real data performance

## Alternative Datasets

If the Kaggle dataset doesn't work, try these alternatives:

### European Credit Dataset
- https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
- More structured, easier to map

### Synthetic Financial Data
- https://github.com/vijayvee/financial-fraud-detection
- Pre-cleaned transaction data

### Bank Transaction Data
- https://www.kaggle.com/datasets/ntnu-testimon/bank-transaction-dataset
- Real banking transactions

## Custom Data Testing

To test your own dataset:

1. **Format**: CSV with transaction data
2. **Required columns**: Date, Amount, Transaction_Type (or equivalents)
3. **Save as**: `my_transactions.csv`
4. **Run**: `py test_real_world_data.py`

The framework will automatically handle column mapping and preprocessing!
