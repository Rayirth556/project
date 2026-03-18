import pandas as pd

class Normalizer:
    """
    Formats and normalizes raw data ingested from external APIs.
    """
    def normalize_setu_data(self, raw_data: dict) -> pd.DataFrame:
        """
        Converts raw JSON data into a structured pandas DataFrame.
        """
        # Basic normalization logic Example
        df = pd.json_normalize(raw_data)
        
        # Add timestamp or default formatting if needed
        df['ingested_at'] = pd.Timestamp.now()
        
        return df

def flatten_aa_json(raw_json) -> pd.DataFrame:
    """
    Takes a complex Account Aggregator JSON payload and uses Pandas to flatten it 
    into a clean, normalized DataFrame.
    Returns columns: 'Date', 'Transaction_Type', 'Amount', 'Category', 'Balance'.
    """
    transactions = []
    
    # Recursively find the 'Transaction' list in the complex JSON payload
    def find_transactions(node):
        if isinstance(node, dict):
            for k, v in node.items():
                # Setu/AA payloads commonly use `transaction` (lowercase) inside
                # `account.transactions.transaction`, while some payloads use `Transaction`.
                if isinstance(k, str) and k.lower() == 'transaction' and isinstance(v, list):
                    transactions.extend(v)
                else:
                    find_transactions(v)
        elif isinstance(node, list):
            for item in node:
                find_transactions(item)
                
    find_transactions(raw_json)
    
    # Flatten using pandas
    if transactions:
        df = pd.json_normalize(transactions)
    else:
        df = pd.DataFrame()
        
    expected_cols = ['Date', 'Transaction_Type', 'Amount', 'Category', 'Balance']
    if df.empty:
        return pd.DataFrame(columns=expected_cols)

    # 1. Date
    if 'transactionTimestamp' in df.columns:
        df['Date'] = pd.to_datetime(df['transactionTimestamp'], errors='coerce')
    elif 'date' in df.columns:
        df['Date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['Date'] = pd.NaT

    # 2. Transaction_Type (CREDIT/DEBIT)
    if 'type' in df.columns:
        type_mapping = {'DEBIT': 'DEBIT', 'CREDIT': 'CREDIT', 'DR': 'DEBIT', 'CR': 'CREDIT'}
        df['Transaction_Type'] = df['type'].astype(str).str.upper().map(type_mapping).fillna('UNKNOWN')
    else:
        df['Transaction_Type'] = 'UNKNOWN'

    # 3. Amount
    if 'amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['amount'], errors='coerce')
    else:
        df['Amount'] = 0.0

    # 4. Category
    if 'narration' in df.columns:
        df['Category'] = df['narration']
    else:
        df['Category'] = 'Uncategorized'

    # 5. Balance
    if 'currentBalance' in df.columns:
        df['Balance'] = pd.to_numeric(df['currentBalance'], errors='coerce')
    else:
        df['Balance'] = 0.0

    return df[expected_cols]
