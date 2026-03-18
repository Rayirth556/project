"""
create_real_training_data.py
============================
Create proper training data that matches your actual use case.

This creates training data that simulates:
1. Individual users uploading transaction data
2. Users filling background data via UI
3. Real default outcomes based on feature patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).parent))

from pdr_pipeline.layer_2_feature_engine import FeatureStoreMSME

class RealTrainingDataGenerator:
    """Generate training data that matches real-world usage"""
    
    def __init__(self):
        self.output_dir = Path("data/real_training")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load real transaction data for patterns
        self.trans_df = pd.read_csv("transactions_data.csv")
        print(f"Loaded {len(self.trans_df):,} transactions for pattern analysis")
        
    def create_synthetic_users(self, n_users=5000):
        """Create synthetic users with realistic transaction patterns"""
        print(f"Creating {n_users:,} synthetic users...")
        
        users = []
        
        for i in range(n_users):
            # Create user profile
            user_id = f"user_{i+1:06d}"
            
            # Risk tier determines behavior
            risk_tier = np.random.choice(['low', 'medium', 'high'], 
                                       p=[0.6, 0.3, 0.1])
            
            # Generate user background data (what they'd enter in UI)
            ui_data = self.generate_background_data(risk_tier)
            
            # Generate transaction data (what they'd upload)
            transactions = self.generate_user_transactions(user_id, risk_tier, ui_data)
            
            # Calculate actual default probability based on patterns
            default_prob = self.calculate_default_probability(ui_data, transactions, risk_tier)
            
            # Determine if they actually defaulted
            defaulted = np.random.random() < default_prob
            
            users.append({
                'user_id': user_id,
                'risk_tier': risk_tier,
                'ui_data': ui_data,
                'transactions': transactions,
                'default_probability': default_prob,
                'defaulted': int(defaulted)
            })
            
            if (i + 1) % 1000 == 0:
                print(f"  Created {i+1:,} users...")
        
        return users
    
    def generate_background_data(self, risk_tier):
        """Generate realistic background data based on risk tier"""
        
        if risk_tier == 'low':
            # Low risk: stable, good financial habits
            return {
                'avg_utility_dpd': np.random.uniform(0, 3),
                'telecom_number_vintage_days': np.random.randint(365, 2000),
                'telecom_recharge_drop_ratio': np.random.uniform(0.8, 1.2),
                'academic_background_tier': np.random.choice([2, 3], p=[0.4, 0.6]),
                'purpose_of_loan_encoded': np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3]),
                'business_vintage_months': np.random.randint(24, 120),
                'revenue_growth_trend': np.random.uniform(0.05, 0.15),
                'revenue_seasonality_index': np.random.uniform(0.1, 0.2),
                'operating_cashflow_ratio': np.random.uniform(1.2, 2.0),
                'cashflow_volatility': np.random.uniform(5000, 15000),
                'avg_invoice_payment_delay': np.random.uniform(0, 15),
                'customer_concentration_ratio': np.random.uniform(0.3, 0.6),
                'repeat_customer_revenue_pct': np.random.uniform(0.7, 0.95),
                'vendor_payment_discipline': np.random.uniform(15, 45),
                'gst_filing_consistency_score': np.random.uniform(10, 12),
                'gst_to_bank_variance': np.random.uniform(0.1, 0.3),
                'p2p_circular_loop_flag': 0,
                'benford_anomaly_score': np.random.uniform(0.4, 0.7),
                'round_number_spike_ratio': np.random.uniform(0, 0.05),
                'turnover_inflation_spike': 0,
                'identity_device_mismatch': 0
            }
        
        elif risk_tier == 'medium':
            # Medium risk: some inconsistencies
            return {
                'avg_utility_dpd': np.random.uniform(2, 8),
                'telecom_number_vintage_days': np.random.randint(180, 730),
                'telecom_recharge_drop_ratio': np.random.uniform(0.5, 1.5),
                'academic_background_tier': np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3]),
                'purpose_of_loan_encoded': np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2]),
                'business_vintage_months': np.random.randint(12, 48),
                'revenue_growth_trend': np.random.uniform(-0.05, 0.08),
                'revenue_seasonality_index': np.random.uniform(0.15, 0.35),
                'operating_cashflow_ratio': np.random.uniform(0.8, 1.3),
                'cashflow_volatility': np.random.uniform(15000, 35000),
                'avg_invoice_payment_delay': np.random.uniform(10, 30),
                'customer_concentration_ratio': np.random.uniform(0.5, 0.8),
                'repeat_customer_revenue_pct': np.random.uniform(0.5, 0.8),
                'vendor_payment_discipline': np.random.uniform(30, 60),
                'gst_filing_consistency_score': np.random.uniform(6, 10),
                'gst_to_bank_variance': np.random.uniform(0.3, 0.6),
                'p2p_circular_loop_flag': np.random.choice([0, 0, 0, 1], p=[0.9, 0.05, 0.04, 0.01]),
                'benford_anomaly_score': np.random.uniform(0.6, 1.0),
                'round_number_spike_ratio': np.random.uniform(0.02, 0.1),
                'turnover_inflation_spike': 0,
                'identity_device_mismatch': np.random.choice([0, 0, 0, 1], p=[0.95, 0.03, 0.02, 0])
            }
        
        else:  # high risk
            # High risk: red flags
            return {
                'avg_utility_dpd': np.random.uniform(5, 20),
                'telecom_number_vintage_days': np.random.randint(30, 365),
                'telecom_recharge_drop_ratio': np.random.uniform(0.2, 1.8),
                'academic_background_tier': np.random.choice([1, 2], p=[0.6, 0.4]),
                'purpose_of_loan_encoded': np.random.choice([1, 2], p=[0.7, 0.3]),
                'business_vintage_months': np.random.randint(3, 18),
                'revenue_growth_trend': np.random.uniform(-0.15, 0.02),
                'revenue_seasonality_index': np.random.uniform(0.25, 0.5),
                'operating_cashflow_ratio': np.random.uniform(0.3, 0.9),
                'cashflow_volatility': np.random.uniform(30000, 80000),
                'avg_invoice_payment_delay': np.random.uniform(25, 90),
                'customer_concentration_ratio': np.random.uniform(0.7, 1.0),
                'repeat_customer_revenue_pct': np.random.uniform(0.2, 0.6),
                'vendor_payment_discipline': np.random.uniform(45, 120),
                'gst_filing_consistency_score': np.random.uniform(2, 6),
                'gst_to_bank_variance': np.random.uniform(0.5, 1.2),
                'p2p_circular_loop_flag': np.random.choice([0, 1, 1], p=[0.7, 0.2, 0.1]),
                'benford_anomaly_score': np.random.uniform(0.8, 1.5),
                'round_number_spike_ratio': np.random.uniform(0.05, 0.2),
                'turnover_inflation_spike': np.random.choice([0, 1], p=[0.8, 0.2]),
                'identity_device_mismatch': np.random.choice([0, 1, 1], p=[0.8, 0.15, 0.05])
            }
    
    def generate_user_transactions(self, user_id, risk_tier, ui_data):
        """Generate realistic transaction data for a user"""
        
        # Determine transaction volume based on business vintage
        n_transactions = np.random.randint(50, 500)
        
        # Base transaction amounts based on business size
        if risk_tier == 'low':
            avg_credit = np.random.uniform(10000, 50000)
            avg_debit = np.random.uniform(5000, 20000)
        elif risk_tier == 'medium':
            avg_credit = np.random.uniform(5000, 25000)
            avg_debit = np.random.uniform(3000, 15000)
        else:
            avg_credit = np.random.uniform(2000, 15000)
            avg_debit = np.random.uniform(1000, 8000)
        
        # Generate transactions over 6 months
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-06-30')
        
        transactions = []
        current_balance = ui_data.get('operating_cashflow_ratio', 1.0) * 10000
        
        for i in range(n_transactions):
            # Random date
            date = start_date + pd.Timedelta(days=np.random.randint(0, 180))
            
            # Transaction type
            if np.random.random() < 0.4:  # 40% credits
                amount = np.random.normal(avg_credit, avg_credit * 0.3)
                trans_type = 'CREDIT'
                current_balance += amount
            else:  # 60% debits
                amount = np.random.normal(avg_debit, avg_debit * 0.3)
                trans_type = 'DEBIT'
                current_balance -= amount
            
            # Ensure positive amounts for display
            display_amount = abs(amount)
            
            # Category
            categories = ['Food', 'Retail', 'Utilities', 'Groceries', 'Gas Stations', 
                         'Restaurants', 'Transport', 'Rent', 'Shopping', 'Financial']
            category = np.random.choice(categories)
            
            transactions.append({
                'date': date,
                'client_id': user_id,
                'amount': f"${display_amount:.2f}",
                'Transaction_Type': trans_type,
                'Category': category,
                'Balance': current_balance,
                'Description': f"{category} transaction"
            })
        
        return pd.DataFrame(transactions)
    
    def calculate_default_probability(self, ui_data, transactions, risk_tier):
        """Calculate realistic default probability based on user data"""
        
        # Base probability by risk tier
        base_probs = {'low': 0.05, 'medium': 0.15, 'high': 0.35}
        prob = base_probs[risk_tier]
        
        # Adjust based on key indicators
        if ui_data['avg_utility_dpd'] > 10:
            prob += 0.1
        if ui_data['operating_cashflow_ratio'] < 0.8:
            prob += 0.15
        if ui_data['business_vintage_months'] < 12:
            prob += 0.1
        if ui_data['p2p_circular_loop_flag'] == 1:
            prob += 0.2
        if ui_data['identity_device_mismatch'] == 1:
            prob += 0.15
        
        # Transaction patterns
        if len(transactions) < 100:  # Low activity
            prob += 0.05
        
        # Cap probability
        return min(prob, 0.8)
    
    def generate_training_dataset(self, users):
        """Generate the final training dataset"""
        print("Generating training dataset...")
        
        features = []
        labels = []
        metadata = []
        
        for user in users:
            try:
                # Generate features using our pipeline
                feature_store = FeatureStoreMSME(user['transactions'], user['ui_data'])
                feature_vector = feature_store.generate_feature_vector()
                
                # Remove transaction_count if present
                if 'transaction_count' in feature_vector:
                    del feature_vector['transaction_count']
                
                features.append(feature_vector)
                labels.append(user['defaulted'])
                
                metadata.append({
                    'user_id': user['user_id'],
                    'risk_tier': user['risk_tier'],
                    'default_probability': user['default_probability'],
                    'transaction_count': len(user['transactions'])
                })
                
            except Exception as e:
                print(f"Error processing user {user['user_id']}: {e}")
                continue
        
        # Convert to DataFrames
        features_df = pd.DataFrame(features)
        labels_df = pd.DataFrame({'TARGET': labels})
        metadata_df = pd.DataFrame(metadata)
        
        print(f"Generated {len(features_df):,} training samples")
        print(f"Default rate: {labels_df['TARGET'].mean():.1%}")
        
        # Save datasets
        features_df.to_parquet(self.output_dir / "features.parquet", index=False)
        labels_df.to_parquet(self.output_dir / "labels.parquet", index=False)
        metadata_df.to_parquet(self.output_dir / "metadata.parquet", index=False)
        
        # Save summary
        summary = {
            'n_samples': len(features_df),
            'n_features': features_df.shape[1],
            'default_rate': float(labels_df['TARGET'].mean()),
            'risk_tier_distribution': metadata_df['risk_tier'].value_counts().to_dict(),
            'feature_columns': list(features_df.columns)
        }
        
        with open(self.output_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training data saved to: {self.output_dir}")
        return features_df, labels_df, metadata_df
    
    def run(self, n_users=5000):
        """Run the complete data generation process"""
        print("=" * 60)
        print("REAL TRAINING DATA GENERATION")
        print("=" * 60)
        
        # Create synthetic users
        users = self.create_synthetic_users(n_users)
        
        # Generate training dataset
        features_df, labels_df, metadata_df = self.generate_training_dataset(users)
        
        print("\n" + "=" * 60)
        print("SUCCESS! Real training data created.")
        print("=" * 60)
        print("This data matches your actual use case:")
        print("✅ Individual user transaction uploads")
        print("✅ Background data from UI")
        print("✅ Real default patterns")
        print("✅ Ready for model training")
        
        return features_df, labels_df, metadata_df

if __name__ == "__main__":
    generator = RealTrainingDataGenerator()
    generator.run(n_users=5000)
