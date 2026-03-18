"""
simplified_risk_workflow.py
==========================
Complete end-to-end risk assessment workflow.

Workflow:
1. User enters background data via simple UI/form
2. Transaction data comes from transactions_data.csv
3. Model processes both and gives risk score
4. No SETU dependency - works with any transaction format

Input Format for Transaction Data:
- Required: date, client_id, amount (can be string like "$-77.00")
- Optional: merchant_category, description
- We'll auto-detect and map columns
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).parent))

from pdr_pipeline.layer_3_inference_engine import InferenceEngine
from pdr_pipeline.layer_2_feature_engine import FeatureStoreMSME

class SimplifiedRiskWorkflow:
    """Simplified risk assessment without SETU dependency"""
    
    def __init__(self):
        self.transaction_data_path = Path("transactions_data.csv")
        self.model_path = "models/pdr_xgb_full_31_features.json"
        
    def get_user_background_data(self):
        """Simple form to collect user background data"""
        print("=" * 60)
        print("USER BACKGROUND DATA FORM")
        print("=" * 60)
        print("Please enter the following information (press Enter for defaults):")
        
        # Collect user input with defaults
        user_data = {}
        
        # Personal Information
        print("\n--- Personal Information ---")
        user_data['avg_utility_dpd'] = self._get_float_input("Average utility payment delay (days)", 2.0)
        user_data['telecom_number_vintage_days'] = self._get_int_input("Mobile number age (days)", 365)
        user_data['telecom_recharge_drop_ratio'] = self._get_float_input("Mobile recharge consistency (0.5-2.0)", 0.8)
        user_data['academic_background_tier'] = self._get_int_input("Education level (1=High School, 2=Graduate, 3=Postgraduate)", 2)
        
        # Financial Information
        print("\n--- Financial Information ---")
        user_data['purpose_of_loan_encoded'] = self._get_int_input("Loan purpose (1=Business, 2=Personal, 3=Education)", 1)
        user_data['business_vintage_months'] = self._get_int_input("Business operating months", 24)
        user_data['revenue_growth_trend'] = self._get_float_input("Revenue growth rate (-0.1 to 0.2)", 0.05)
        user_data['revenue_seasonality_index'] = self._get_float_input("Revenue seasonality (0.1 to 0.3)", 0.15)
        user_data['operating_cashflow_ratio'] = self._get_float_input("Operating cash flow ratio", 1.2)
        user_data['cashflow_volatility'] = self._get_float_input("Cash flow volatility", 15000)
        
        # Business Information
        print("\n--- Business Information ---")
        user_data['avg_invoice_payment_delay'] = self._get_float_input("Average invoice payment delay (days)", 15)
        user_data['customer_concentration_ratio'] = self._get_float_input("Customer concentration (0.3 to 1.0)", 0.7)
        user_data['repeat_customer_revenue_pct'] = self._get_float_input("Repeat customer revenue percentage", 0.8)
        user_data['vendor_payment_discipline'] = self._get_float_input("Vendor payment discipline score", 30)
        
        # Compliance Information
        print("\n--- Compliance Information ---")
        user_data['gst_filing_consistency_score'] = self._get_float_input("GST filing consistency (2-12)", 11)
        user_data['gst_to_bank_variance'] = self._get_float_input("GST to bank variance", 0.3)
        
        return user_data
    
    def _get_float_input(self, prompt, default):
        """Helper to get float input with default"""
        try:
            value = input(f"{prompt} [{default}]: ").strip()
            return float(value) if value else default
        except ValueError:
            print(f"Using default: {default}")
            return default
    
    def _get_int_input(self, prompt, default):
        """Helper to get int input with default"""
        try:
            value = input(f"{prompt} [{default}]: ").strip()
            return int(value) if value else default
        except ValueError:
            print(f"Using default: {default}")
            return default
    
    def load_and_preprocess_transactions(self, client_id=None):
        """Load and preprocess transaction data for a specific client"""
        print(f"Loading transaction data from {self.transaction_data_path}")
        
        try:
            # Load transaction data
            df = pd.read_csv(self.transaction_data_path)
            print(f"Loaded {len(df):,} transactions for {df['client_id'].nunique():,} clients")
            
            # Preprocess
            df['date'] = pd.to_datetime(df['date'])
            
            # Clean amount column
            if 'amount' in df.columns:
                df['amount_clean'] = df['amount'].astype(str).str.replace('$', '').str.replace(',', '')
                df['amount_clean'] = pd.to_numeric(df['amount_clean'], errors='coerce')
            else:
                # Try other common amount column names
                for col in df.columns:
                    if 'amount' in col.lower() or 'value' in col.lower():
                        df['amount_clean'] = pd.to_numeric(df[col], errors='coerce')
                        break
            
            # Determine transaction type
            df['Transaction_Type'] = df['amount_clean'].apply(lambda x: 'CREDIT' if x > 0 else 'DEBIT')
            
            # Create category from available data
            if 'mcc' in df.columns:
                mcc_mapping = {
                    '5499': 'Food', '5311': 'Retail', '4829': 'Utilities',
                    '5411': 'Groceries', '5541': 'Gas Stations', '5812': 'Restaurants',
                    '5814': 'Fast Food', '4121': 'Transport', '6532': 'Rent',
                    '5999': 'Shopping', '6011': 'Financial', '7216': 'Cleaning'
                }
                df['Category'] = df['mcc'].astype(str).map(mcc_mapping).fillna('Other')
            else:
                df['Category'] = 'Transaction'
            
            # Create running balance
            df = df.sort_values(['client_id', 'date'])
            df['Balance'] = df.groupby('client_id')['amount_clean'].cumsum()
            
            # Create description
            df['Description'] = df['Category'] + ' transaction'
            
            # Rename for compatibility
            df = df.rename(columns={'date': 'Date', 'amount_clean': 'Amount'})
            
            # Filter for specific client if provided
            if client_id:
                df = df[df['client_id'] == client_id]
                print(f"Filtered to {len(df)} transactions for client {client_id}")
            
            print(f"Transaction data prepared successfully")
            print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"Transaction types: {df['Transaction_Type'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            print(f"Error loading transaction data: {e}")
            return None
    
    def assess_risk(self, user_data, transaction_data):
        """Complete risk assessment combining user data and transactions"""
        print("\n" + "=" * 60)
        print("RISK ASSESSMENT IN PROGRESS...")
        print("=" * 60)
        
        try:
            # Generate features from transaction data + user data
            feature_store = FeatureStoreMSME(transaction_data, user_data)
            feature_vector = feature_store.generate_feature_vector()
            
            print(f"Generated {len(feature_vector)} features")
            
            # Run inference
            engine = InferenceEngine(self.model_path)
            result = engine.predict(feature_vector)
            
            return result, feature_vector
            
        except Exception as e:
            print(f"Error during risk assessment: {e}")
            return None, None
    
    def display_results(self, result, feature_vector):
        """Display risk assessment results"""
        if not result:
            print("Risk assessment failed!")
            return
        
        print("\n" + "=" * 60)
        print("RISK ASSESSMENT RESULTS")
        print("=" * 60)
        
        # Main results
        print(f"\nRISK SCORE: {result['risk_score']:.4f}")
        print(f"WEIGHTED SCORE: {result.get('weighted_score', 'N/A')}")
        print(f"DECISION: {result['decision']}")
        
        # Policy overrides
        if result['policy_overrides']:
            print(f"\nPOLICY OVERRIDES:")
            for override in result['policy_overrides']:
                print(f"  - {override}")
        
        # Missing features
        if result['missing_features']:
            print(f"\nMISSING FEATURES: {len(result['missing_features'])}")
            if len(result['missing_features']) <= 5:
                for feature in result['missing_features']:
                    print(f"  - {feature}")
        
        # Feature breakdown (top 10 most impactful)
        print(f"\nTOP 10 FEATURE CONTRIBUTIONS:")
        
        # Calculate feature impacts based on weights
        feature_impacts = []
        for feature, weight in self._get_feature_weights().items():
            if feature in feature_vector and weight > 0:
                value = feature_vector[feature]
                impact = abs(value * weight)
                feature_impacts.append((feature, impact, weight, value))
        
        # Sort by impact
        feature_impacts.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, impact, weight, value) in enumerate(feature_impacts[:10], 1):
            print(f"  {i:2d}. {feature:<30} = {value:8.3f} (weight: {weight:.1%})")
        
        # Risk interpretation
        print(f"\nRISK INTERPRETATION:")
        if result['decision'] == 'APPROVE':
            print("  OK LOW RISK - Recommended for approval")
            print("  OK Standard terms and conditions apply")
        elif result['decision'] == 'REVIEW':
            print("  WARNING MEDIUM RISK - Requires manual review")
            print("  WARNING Consider additional documentation or guarantor")
        else:
            print("  X HIGH RISK - Recommended for decline")
            print("  X High probability of default detected")
    
    def _get_feature_weights(self):
        """Get feature weights for impact calculation"""
        return {
            "essential_vs_lifestyle_ratio": 0.08,
            "emergency_buffer_months": 0.07,
            "eod_balance_volatility": 0.05,
            "cash_withdrawal_dependency": 0.04,
            "min_balance_violation_count": 0.03,
            "operating_cashflow_ratio": 0.03,
            "utility_payment_consistency": 0.06,
            "avg_utility_dpd": 0.05,
            "rent_wallet_share": 0.04,
            "subscription_commitment_ratio": 0.04,
            "bounced_transaction_count": 0.03,
            "avg_invoice_payment_delay": 0.03,
            "business_vintage_months": 0.06,
            "revenue_growth_trend": 0.05,
            "revenue_seasonality_index": 0.04,
            "customer_concentration_ratio": 0.03,
            "repeat_customer_revenue_pct": 0.02,
            "vendor_payment_discipline": 0.04,
            "gst_filing_consistency_score": 0.03,
            "gst_to_bank_variance": 0.03,
            "telecom_number_vintage_days": 0.03,
            "telecom_recharge_drop_ratio": 0.03,
            "purpose_of_loan_encoded": 0.02,
            "cashflow_volatility": 0.02,
            "academic_background_tier": 0.05,
        }
    
    def run_interactive_workflow(self):
        """Run complete interactive workflow"""
        print("BANK SIMPLIFIED RISK ASSESSMENT WORKFLOW")
        print("No SETU dependency - works with any transaction data!")
        
        # Step 1: Get user background data
        user_data = self.get_user_background_data()
        
        # Step 2: Load transaction data
        transaction_data = self.load_and_preprocess_transactions()
        
        if transaction_data is None:
            print("Failed to load transaction data. Exiting.")
            return
        
        # Step 3: Assess risk
        result, feature_vector = self.assess_risk(user_data, transaction_data)
        
        # Step 4: Display results
        self.display_results(result, feature_vector)
        
        # Step 5: Save results
        if result:
            self.save_assessment_result(user_data, result, feature_vector)
    
    def run_batch_workflow(self, client_ids=None, max_clients=10):
        """Run batch assessment for multiple clients"""
        print("BATCH RISK ASSESSMENT")
        
        # Load transaction data
        transaction_data = self.load_and_preprocess_transactions()
        
        if transaction_data is None:
            print("Failed to load transaction data. Exiting.")
            return
        
        # Get unique clients
        unique_clients = transaction_data['client_id'].unique()
        
        if client_ids:
            # Use specified clients
            test_clients = [cid for cid in client_ids if cid in unique_clients]
        else:
            # Sample random clients
            test_clients = np.random.choice(unique_clients, min(max_clients, len(unique_clients)), replace=False)
        
        print(f"Assessing {len(test_clients)} clients...")
        
        results = []
        for client_id in test_clients:
            print(f"\n--- Assessing Client {client_id} ---")
            
            # Filter client transactions
            client_transactions = transaction_data[transaction_data['client_id'] == client_id]
            
            # Use default user data (can be customized)
            default_user_data = {
                'avg_utility_dpd': 2.0,
                'telecom_number_vintage_days': 365,
                'telecom_recharge_drop_ratio': 0.8,
                'academic_background_tier': 2,
                'purpose_of_loan_encoded': 1,
                'business_vintage_months': 24,
                'revenue_growth_trend': 0.05,
                'revenue_seasonality_index': 0.15,
                'operating_cashflow_ratio': 1.2,
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
            
            # Assess risk
            result, feature_vector = self.assess_risk(default_user_data, client_transactions)
            
            if result:
                results.append({
                    'client_id': client_id,
                    'transaction_count': len(client_transactions),
                    'risk_score': result['risk_score'],
                    'weighted_score': result.get('weighted_score', 0),
                    'decision': result['decision'],
                    'policy_overrides': result['policy_overrides']
                })
        
        # Display batch results
        self.display_batch_results(results)
        
        # Save batch results
        if results:  # Only save if we have results
            self.save_batch_results(results)
    
    def display_batch_results(self, results):
        """Display batch assessment results"""
        print("\n" + "=" * 80)
        print("BATCH ASSESSMENT RESULTS")
        print("=" * 80)
        
        if not results:
            print("No successful assessments to display.")
            return
        
        # Summary
        approved = sum(1 for r in results if r['decision'] == 'APPROVE')
        review = sum(1 for r in results if r['decision'] == 'REVIEW')
        declined = sum(1 for r in results if r['decision'] == 'DECLINE')
        
        print(f"\nSUMMARY:")
        print(f"  Total Assessed: {len(results)}")
        print(f"  Approved: {approved} ({approved/len(results):.1%})")
        print(f"  Review: {review} ({review/len(results):.1%})")
        print(f"  Declined: {declined} ({declined/len(results):.1%})")
        print(f"  Average Risk Score: {np.mean([r['risk_score'] for r in results]):.4f}")
        
        # Detailed results
        print(f"\nDETAILED RESULTS:")
        print(f"{'Client ID':<10} {'Transactions':<12} {'Risk Score':<12} {'Weighted':<10} {'Decision':<10}")
        print("-" * 60)
        
        for result in results:
            print(f"{result['client_id']:<10} {result['transaction_count']:<12} "
                  f"{result['risk_score']:<12.4f} {result['weighted_score']:<10.4f} "
                  f"{result['decision']:<10}")
    
    def save_assessment_result(self, user_data, result, feature_vector):
        """Save individual assessment result"""
        assessment_data = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'user_data': user_data,
            'risk_result': result,
            'feature_vector': feature_vector
        }
        
        output_path = Path("risk_assessment_result.json")
        with open(output_path, 'w') as f:
            json.dump(assessment_data, f, indent=2, default=str)
        
        print(f"\nAssessment result saved to: {output_path}")
    
    def save_batch_results(self, results):
        """Save batch assessment results"""
        if results:
            df = pd.DataFrame(results)
            output_path = Path("batch_risk_assessment_results.csv")
            df.to_csv(output_path, index=False)
            print(f"\nBatch results saved to: {output_path}")

def main():
    """Main entry point"""
    workflow = SimplifiedRiskWorkflow()
    
    print("Choose workflow mode:")
    print("1. Interactive (single user assessment)")
    print("2. Batch (multiple clients from transaction data)")
    print("3. Demo (sample client assessment)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        workflow.run_interactive_workflow()
    elif choice == '2':
        workflow.run_batch_workflow()
    elif choice == '3':
        # Demo with sample client
        print("Running demo assessment...")
        workflow.run_batch_workflow(client_ids=[7475327, 561, 1129], max_clients=3)
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
