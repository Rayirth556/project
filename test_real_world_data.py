"""
test_real_world_data.py
========================
Testing framework for real-world financial datasets.
Supports Kaggle transactions dataset and custom financial data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Import our pipeline components
sys.path.append(str(Path(__file__).parent))
from pdr_pipeline.layer_3_inference_engine import InferenceEngine
from pdr_pipeline.layer_2_feature_engine import FeatureStoreMSME

class RealWorldDataTester:
    """Test PDR pipeline on real-world financial datasets"""
    
    def __init__(self):
        self.results = []
        self.model_path = "models/pdr_xgb_clean.json"
        
    def load_kaggle_transactions(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess Kaggle transactions dataset"""
        try:
            print(f"Loading Kaggle dataset: {csv_path}")
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} transactions")
            print(f"Columns: {list(df.columns)}")
            
            # Map Kaggle columns to our expected format
            column_mapping = {
                # Common transaction columns - adjust based on actual dataset
                'amount': 'Amount',
                'date': 'Date', 
                'type': 'Transaction_Type',
                'category': 'Category',
                'description': 'Description',
                'balance': 'Balance'
            }
            
            # Apply mapping for columns that exist
            for kaggle_col, our_col in column_mapping.items():
                if kaggle_col in df.columns and our_col not in df.columns:
                    df = df.rename(columns={kaggle_col: our_col})
            
            return df
            
        except Exception as e:
            print(f"Error loading Kaggle dataset: {e}")
            return None
    
    def preprocess_kaggle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess Kaggle data to match our expected format"""
        print("Preprocessing Kaggle data...")
        
        # Ensure required columns exist
        required_cols = ['Date', 'Amount', 'Transaction_Type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            print("Available columns:", list(df.columns))
            
            # Try to infer missing columns
            if 'Date' not in df.columns:
                # Look for date-like columns
                date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols:
                    df = df.rename(columns={date_cols[0]: 'Date'})
                    print(f"Inferred Date column from: {date_cols[0]}")
            
            if 'Amount' not in df.columns:
                # Look for amount-like columns
                amount_cols = [col for col in df.columns if 'amount' in col.lower() or 'value' in col.lower()]
                if amount_cols:
                    df = df.rename(columns={amount_cols[0]: 'Amount'})
                    print(f"Inferred Amount column from: {amount_cols[0]}")
        
        # Convert Date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Standardize Transaction_Type
        if 'Transaction_Type' in df.columns:
            df['Transaction_Type'] = df['Transaction_Type'].astype(str).str.upper()
            # Map common transaction types
            type_mapping = {
                'CREDIT': 'CREDIT',
                'DEBIT': 'DEBIT', 
                'WITHDRAWAL': 'DEBIT',
                'DEPOSIT': 'CREDIT',
                'PAYMENT': 'DEBIT',
                'TRANSFER': 'DEBIT',
                'INCOME': 'CREDIT',
                'EXPENSE': 'DEBIT'
            }
            df['Transaction_Type'] = df['Transaction_Type'].map(type_mapping).fillna(df['Transaction_Type'])
        
        # Add Balance column if missing (calculate running balance)
        if 'Balance' not in df.columns and 'Amount' in df.columns and 'Date' in df.columns:
            print("Calculating running balance...")
            df = df.sort_values('Date')
            df['Balance'] = df['Amount'].cumsum()
        
        print(f"Preprocessed shape: {df.shape}")
        print(f"Final columns: {list(df.columns)}")
        
        return df
    
    def create_sample_ui_data_from_transactions(self, df: pd.DataFrame) -> dict:
        """Create UI data from transaction patterns"""
        print("Creating UI data from transaction patterns...")
        
        ui_data = {
            # Default values - can be enhanced based on data analysis
            'avg_utility_dpd': 2.0,
            'telecom_number_vintage_days': 365,
            'telecom_recharge_drop_ratio': 0.8,
            'purpose_of_loan_encoded': 1,  # Business expansion
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
        
        # Analyze transaction patterns to improve UI data
        if 'Amount' in df.columns and 'Transaction_Type' in df.columns:
            credits = df[df['Transaction_Type'] == 'CREDIT']['Amount']
            debits = df[df['Transaction_Type'] == 'DEBIT']['Amount']
            
            if not credits.empty:
                avg_income = credits.mean()
                if not debits.empty:
                    avg_expenses = abs(debits.mean())
                    ui_data['rent_wallet_share'] = min(avg_expenses / avg_income, 1.0) if avg_income > 0 else 0.0
                    ui_data['subscription_commitment_ratio'] = min(avg_expenses / avg_income * 0.3, 1.0)
        
        return ui_data
    
    def test_single_customer(self, customer_transactions: pd.DataFrame, ui_data: dict) -> dict:
        """Test pipeline on a single customer's data"""
        try:
            print(f"Testing customer with {len(customer_transactions)} transactions...")
            
            # Generate features
            feature_store = FeatureStoreMSME(customer_transactions, ui_data)
            feature_vector = feature_store.generate_feature_vector()
            
            # Run inference
            engine = InferenceEngine(self.model_path)
            result = engine.predict(feature_vector)
            
            return {
                'success': True,
                'transaction_count': len(customer_transactions),
                'feature_count': len(feature_vector),
                'risk_score': result['risk_score'],
                'decision': result['decision'],
                'policy_overrides': result['policy_overrides'],
                'missing_features': result['missing_features']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'transaction_count': len(customer_transactions) if 'customer_transactions' in locals() else 0
            }
    
    def test_kaggle_dataset(self, csv_path: str, sample_size: int = 100):
        """Test pipeline on Kaggle dataset"""
        print("=" * 60)
        print("TESTING ON KAGGLE TRANSACTIONS DATASET")
        print("=" * 60)
        
        # Load and preprocess data
        df = self.load_kaggle_transactions(csv_path)
        if df is None:
            return False
        
        df = self.preprocess_kaggle_data(df)
        
        # Create UI data
        ui_data = self.create_sample_ui_data_from_transactions(df)
        
        # Sample customers (group by some identifier if available)
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df
        
        print(f"Testing on {len(df_sample)} transactions...")
        
        # Test as single customer (can be enhanced for multi-customer)
        result = self.test_single_customer(df_sample, ui_data)
        
        if result['success']:
            print(f"OK SUCCESS: Risk Score = {result['risk_score']:.4f}, Decision = {result['decision']}")
            if result['policy_overrides']:
                print(f"Policy Overrides: {result['policy_overrides']}")
            if result['missing_features']:
                print(f"Missing Features: {len(result['missing_features'])}")
        else:
            print(f"X FAILED: {result['error']}")
        
        self.results.append({
            'dataset': 'Kaggle Transactions',
            'sample_size': len(df_sample),
            'result': result
        })
        
        return result['success']
    
    def test_synthetic_realistic_data(self):
        """Test on synthetic but realistic transaction patterns"""
        print("=" * 60)
        print("TESTING ON SYNTHETIC REALISTIC DATA")
        print("=" * 60)
        
        # Create realistic transaction patterns
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', '2024-06-30', freq='D')
        
        transactions = []
        balance = 50000  # Starting balance
        
        for date in dates:
            # Random daily transactions
            n_transactions = np.random.poisson(2)  # Average 2 transactions per day
            
            for _ in range(n_transactions):
                # Realistic transaction amounts and types
                if np.random.random() < 0.3:  # 30% credits
                    amount = np.random.lognormal(8, 1)  # Log-normal distribution for income
                    transaction_type = 'CREDIT'
                    category = np.random.choice(['Salary', 'Sales', 'Refund', 'Investment'])
                else:  # 70% debits
                    amount = -np.random.lognormal(6, 0.8)  # Smaller log-normal for expenses
                    transaction_type = 'DEBIT'
                    category = np.random.choice(['Food', 'Rent', 'Utilities', 'Shopping', 'Transport'])
                
                balance += amount
                
                transactions.append({
                    'Date': date,
                    'Amount': amount,
                    'Transaction_Type': transaction_type,
                    'Category': category,
                    'Description': f'{category} transaction',
                    'Balance': balance
                })
        
        df = pd.DataFrame(transactions)
        print(f"Generated {len(df)} realistic transactions")
        
        # Create realistic UI data
        ui_data = {
            'avg_utility_dpd': np.random.uniform(0, 10),
            'telecom_number_vintage_days': np.random.randint(30, 1000),
            'telecom_recharge_drop_ratio': np.random.uniform(0.5, 1.5),
            'purpose_of_loan_encoded': np.random.randint(1, 4),
            'business_vintage_months': np.random.randint(6, 60),
            'revenue_growth_trend': np.random.uniform(-0.1, 0.2),
            'revenue_seasonality_index': np.random.uniform(0.1, 0.3),
            'operating_cashflow_ratio': np.random.uniform(0.8, 1.5),
            'operating_cashflow_survival_flag': np.random.randint(0, 2),
            'cashflow_volatility': np.random.uniform(5000, 50000),
            'avg_invoice_payment_delay': np.random.uniform(0, 60),
            'customer_concentration_ratio': np.random.uniform(0.3, 1.0),
            'repeat_customer_revenue_pct': np.random.uniform(0.5, 1.0),
            'vendor_payment_discipline': np.random.uniform(0, 90),
            'gst_filing_consistency_score': np.random.uniform(2, 12),
            'gst_to_bank_variance': np.random.uniform(0.1, 1.0),
            'p2p_circular_loop_flag': np.random.choice([0, 0, 0, 1], p=[0.9, 0.05, 0.04, 0.01]),
            'benford_anomaly_score': np.random.uniform(0.4, 1.2),
            'round_number_spike_ratio': np.random.uniform(0, 0.1),
            'turnover_inflation_spike': 0,
            'identity_device_mismatch': np.random.choice([0, 0, 0, 1], p=[0.95, 0.03, 0.02, 0])
        }
        
        # Test multiple scenarios
        scenarios = ['Low Risk', 'Medium Risk', 'High Risk', 'Policy Override']
        
        for scenario in scenarios:
            print(f"\n--- Testing {scenario} Scenario ---")
            
            # Adjust UI data for scenario
            scenario_ui = ui_data.copy()
            if scenario == 'Low Risk':
                scenario_ui.update({
                    'avg_utility_dpd': 0,
                    'business_vintage_months': 48,
                    'revenue_growth_trend': 0.15,
                    'p2p_circular_loop_flag': 0,
                    'identity_device_mismatch': 0
                })
            elif scenario == 'High Risk':
                scenario_ui.update({
                    'avg_utility_dpd': 15,
                    'business_vintage_months': 6,
                    'revenue_growth_trend': -0.05,
                    'p2p_circular_loop_flag': 0,
                    'identity_device_mismatch': 0
                })
            elif scenario == 'Policy Override':
                scenario_ui.update({
                    'p2p_circular_loop_flag': 1,
                    'identity_device_mismatch': 1
                })
            
            result = self.test_single_customer(df, scenario_ui)
            
            if result['success']:
                print(f"OK {scenario}: Risk Score = {result['risk_score']:.4f}, Decision = {result['decision']}")
                if result['policy_overrides']:
                    print(f"   Overrides: {result['policy_overrides']}")
            else:
                print(f"X {scenario}: {result['error']}")
            
            self.results.append({
                'dataset': f'Synthetic {scenario}',
                'sample_size': len(df),
                'result': result
            })
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("REAL-WORLD DATA TESTING REPORT")
        print("=" * 60)
        
        successful_tests = sum(1 for r in self.results if r['result']['success'])
        total_tests = len(self.results)
        
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Success Rate: {successful_tests/total_tests:.1%}")
        
        print(f"\nDetailed Results:")
        for i, result in enumerate(self.results, 1):
            status = "OK" if result['result']['success'] else "X"
            print(f"{i}. {status} {result['dataset']}")
            if result['result']['success']:
                print(f"   Risk Score: {result['result']['risk_score']:.4f}")
                print(f"   Decision: {result['result']['decision']}")
            else:
                print(f"   Error: {result['result']['error']}")
        
        # Save results
        report_path = Path("models/real_world_test_results.json")
        report_path.write_text(json.dumps(self.results, indent=2, default=str))
        print(f"\nDetailed results saved to: {report_path}")

def main():
    """Main testing function"""
    tester = RealWorldDataTester()
    
    # Test 1: Synthetic realistic data (always available)
    tester.test_synthetic_realistic_data()
    
    # Test 2: Kaggle dataset (if available)
    kaggle_path = "transactions_data.csv"  # Update this path
    if Path(kaggle_path).exists():
        print(f"\nFound Kaggle dataset at {kaggle_path}")
        tester.test_kaggle_dataset(kaggle_path)
    else:
        print(f"\nKaggle dataset not found at {kaggle_path}")
        print("To test with Kaggle data:")
        print("1. Download from: https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets")
        print("2. Save as 'transactions_data.csv' in project root")
        print("3. Run this script again")
    
    # Generate report
    tester.generate_test_report()

if __name__ == "__main__":
    main()
