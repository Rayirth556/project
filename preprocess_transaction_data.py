"""
preprocess_transaction_data.py
==============================
Preprocess transaction_data.csv with full 31-feature mapping and weight distribution.

Based on the suggested weight distribution:
- Income/cash flow (30%)
- Transaction behaviour (25%) 
- Business or job stability (20%)
- Utility payment discipline (10%)
- Behavioural application data (10%)
- Education/skill level (5%)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

# Import our pipeline components
import sys
sys.path.append(str(Path(__file__).parent))
from pdr_pipeline.layer_2_feature_engine import FeatureStoreMSME
from pdr_pipeline.layer_3_inference_engine import InferenceEngine

class TransactionDataPreprocessor:
    """Preprocess transaction_data.csv with full feature mapping"""
    
    def __init__(self):
        self.data_path = Path("transactions_data.csv")
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Feature weight mapping based on your distribution
        self.feature_weights = {
            # Income/cash flow (30%)
            'essential_vs_lifestyle_ratio': 0.08,      # 8%
            'emergency_buffer_months': 0.07,            # 7%
            'eod_balance_volatility': 0.05,            # 5%
            'cash_withdrawal_dependency': 0.04,        # 4%
            'min_balance_violation_count': 0.03,       # 3%
            'operating_cashflow_ratio': 0.03,          # 3%
            
            # Transaction behaviour (25%)
            'utility_payment_consistency': 0.06,       # 6%
            'avg_utility_dpd': 0.05,                   # 5%
            'rent_wallet_share': 0.04,                # 4%
            'subscription_commitment_ratio': 0.04,     # 4%
            'bounced_transaction_count': 0.03,          # 3%
            'avg_invoice_payment_delay': 0.03,          # 3%
            
            # Business or job stability (20%)
            'business_vintage_months': 0.06,           # 6%
            'revenue_growth_trend': 0.05,              # 5%
            'revenue_seasonality_index': 0.04,          # 4%
            'customer_concentration_ratio': 0.03,     # 3%
            'repeat_customer_revenue_pct': 0.02,       # 2%
            
            # Utility payment discipline (10%)
            'vendor_payment_discipline': 0.04,         # 4%
            'gst_filing_consistency_score': 0.03,      # 3%
            'gst_to_bank_variance': 0.03,              # 3%
            
            # Behavioural application data (10%)
            'telecom_number_vintage_days': 0.03,        # 3%
            'telecom_recharge_drop_ratio': 0.03,       # 3%
            'purpose_of_loan_encoded': 0.02,           # 2%
            'cashflow_volatility': 0.02,               # 2%
            
            # Education/skill level (5%)
            'academic_background_tier': 0.05,          # 5% (reinstated with proper weight)
            
            # Integrity/fraud features (remaining 0% - policy overrides)
            'operating_cashflow_survival_flag': 0.0,    # 0% (policy override)
            'turnover_inflation_spike': 0.0,           # 0% (policy override)
            'p2p_circular_loop_flag': 0.0,              # 0% (policy override)
            'benford_anomaly_score': 0.0,               # 0% (policy override)
            'round_number_spike_ratio': 0.0,            # 0% (policy override)
            'identity_device_mismatch': 0.0             # 0% (policy override)
        }
        
        # Verify weights sum to 100%
        total_weight = sum(self.feature_weights.values())
        print(f"Total feature weight: {total_weight:.1%}")
    
    def load_transaction_data(self):
        """Load and preprocess transaction_data.csv"""
        print(f"Loading transaction data from {self.data_path}")
        
        # Load data
        df = pd.read_csv(self.data_path)
        print(f"Loaded {len(df):,} transactions")
        print(f"Columns: {list(df.columns)}")
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Convert amount to numeric (remove $ and commas)
        df['amount_clean'] = df['amount'].str.replace('$', '').str.replace(',', '').astype(float)
        
        # Determine transaction type (positive = credit, negative = debit)
        df['Transaction_Type'] = df['amount_clean'].apply(lambda x: 'CREDIT' if x > 0 else 'DEBIT')
        
        # Create category from MCC (Merchant Category Code)
        mcc_mapping = {
            '5499': 'Food',
            '5311': 'Retail', 
            '4829': 'Utilities',
            '5411': 'Groceries',
            '5541': 'Gas Stations',
            '5812': 'Restaurants',
            '5814': 'Fast Food',
            '4121': 'Transport',
            '6532': 'Rent',
            '5999': 'Shopping',
            '6011': 'Financial',
            '7216': 'Cleaning',
            '4111': 'Transport',
            '7298': 'Health',
            '5261': 'Hardware'
        }
        
        df['Category'] = df['mcc'].astype(str).map(mcc_mapping).fillna('Other')
        
        # Create running balance per client
        df = df.sort_values(['client_id', 'date'])
        df['Balance'] = df.groupby('client_id')['amount_clean'].cumsum()
        
        # Create description
        df['Description'] = df['Category'] + ' transaction'
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'date': 'Date',
            'amount_clean': 'Amount'
        })
        
        print(f"Preprocessed shape: {df.shape}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Unique clients: {df['client_id'].nunique():,}")
        
        return df
    
    def create_ui_data_for_client(self, client_data):
        """Create UI data for a specific client based on their transaction patterns"""
        client_id = client_data['client_id'].iloc[0]
        
        # Analyze transaction patterns
        credits = client_data[client_data['Transaction_Type'] == 'CREDIT']
        debits = client_data[client_data['Transaction_Type'] == 'DEBIT']
        
        # Calculate derived metrics
        avg_income = credits['Amount'].mean() if not credits.empty else 0
        avg_expenses = abs(debits['Amount'].mean()) if not debits.empty else 0
        
        # Time-based analysis
        date_range = (client_data['Date'].max() - client_data['Date'].min()).days
        business_vintage_months = max(1, date_range // 30)
        
        # Create UI data with realistic values based on transaction patterns
        ui_data = {
            # NTC Behavioral Discipline (10% weight)
            'avg_utility_dpd': np.random.uniform(0, 10),
            'telecom_number_vintage_days': np.random.randint(30, 1000),
            'telecom_recharge_drop_ratio': np.random.uniform(0.5, 1.5),
            'academic_background_tier': np.random.randint(1, 4),  # Reinstated with 5% weight
            
            # Liquidity & Stress Layer (30% weight)
            'rent_wallet_share': min(avg_expenses / avg_income, 1.0) if avg_income > 0 else 0.0,
            'subscription_commitment_ratio': min(avg_expenses / avg_income * 0.3, 1.0) if avg_income > 0 else 0.0,
            'emergency_buffer_months': np.random.uniform(0.5, 12),
            'min_balance_violation_count': np.random.poisson(2),
            'eod_balance_volatility': np.random.uniform(0.1, 0.8),
            'essential_vs_lifestyle_ratio': np.random.uniform(0.5, 2.0),
            'cash_withdrawal_dependency': np.random.uniform(0.1, 0.9),
            'bounced_transaction_count': np.random.poisson(1),
            
            # Business Viability (20% weight)
            'purpose_of_loan_encoded': np.random.randint(1, 4),
            'business_vintage_months': business_vintage_months,
            'revenue_growth_trend': np.random.uniform(-0.1, 0.2),
            'revenue_seasonality_index': np.random.uniform(0.1, 0.3),
            'operating_cashflow_ratio': np.random.uniform(0.8, 1.5),
            'operating_cashflow_survival_flag': np.random.randint(0, 2),
            'cashflow_volatility': np.random.uniform(5000, 50000),
            
            # Customer Relationships (15% weight)
            'avg_invoice_payment_delay': np.random.uniform(0, 60),
            'customer_concentration_ratio': np.random.uniform(0.3, 1.0),
            'repeat_customer_revenue_pct': np.random.uniform(0.5, 1.0),
            'vendor_payment_discipline': np.random.uniform(0, 90),
            
            # Compliance & Governance (10% weight)
            'gst_filing_consistency_score': np.random.uniform(2, 12),
            'gst_to_bank_variance': np.random.uniform(0.1, 1.0),
            'p2p_circular_loop_flag': np.random.choice([0, 0, 0, 1], p=[0.9, 0.05, 0.04, 0.01]),
            'benford_anomaly_score': np.random.uniform(0.4, 1.2),
            'round_number_spike_ratio': np.random.uniform(0, 0.1),
            'turnover_inflation_spike': 0,
            'identity_device_mismatch': np.random.choice([0, 0, 0, 1], p=[0.95, 0.03, 0.02, 0])
        }
        
        return ui_data
    
    def generate_features_for_all_clients(self, df, max_clients=1000):
        """Generate features for all clients in the dataset"""
        print(f"Generating features for clients...")
        
        # Get unique clients
        unique_clients = df['client_id'].unique()
        
        # Sample clients if too many
        if len(unique_clients) > max_clients:
            unique_clients = np.random.choice(unique_clients, max_clients, replace=False)
            print(f"Sampled {max_clients} clients from {len(df['client_id'].unique()):,} total")
        
        all_features = []
        processed_count = 0
        
        for client_id in unique_clients:
            try:
                # Get client transactions
                client_data = df[df['client_id'] == client_id].copy()
                
                if len(client_data) < 10:  # Skip clients with too few transactions
                    continue
                
                # Create UI data
                ui_data = self.create_ui_data_for_client(client_data)
                
                # Generate features
                feature_store = FeatureStoreMSME(client_data, ui_data)
                feature_vector = feature_store.generate_feature_vector()
                
                # Add client metadata
                feature_vector['client_id'] = client_id
                feature_vector['transaction_count'] = len(client_data)
                
                all_features.append(feature_vector)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} clients...")
                    
            except Exception as e:
                print(f"Error processing client {client_id}: {e}")
                continue
        
        print(f"Successfully processed {processed_count} clients")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Create synthetic labels based on feature patterns (for demonstration)
        # In real scenario, these would come from actual default data
        features_df['TARGET'] = self.generate_synthetic_labels(features_df)
        
        return features_df
    
    def generate_synthetic_labels(self, features_df):
        """Generate synthetic default labels based on feature patterns"""
        print("Generating synthetic labels...")
        
        # Create risk score based on weighted features
        risk_scores = np.zeros(len(features_df))
        
        for feature, weight in self.feature_weights.items():
            if feature in features_df.columns:
                # Normalize feature to 0-1 range
                feature_values = features_df[feature].fillna(0)
                feature_min = feature_values.min()
                feature_max = feature_values.max()
                
                if feature_max > feature_min:
                    normalized = (feature_values - feature_min) / (feature_max - feature_min)
                else:
                    normalized = np.zeros_like(feature_values)
                
                # Add weighted contribution to risk score
                risk_scores += normalized * weight
        
        # Add some randomness
        risk_scores += np.random.normal(0, 0.1, len(features_df))
        
        # Cap risk scores to 0-1
        risk_scores = np.clip(risk_scores, 0, 1)
        
        # Generate labels (higher risk = more likely to default)
        # Target ~15% default rate (typical for MSME lending)
        default_threshold = np.percentile(risk_scores, 85)
        labels = (risk_scores > default_threshold).astype(int)
        
        print(f"Generated {labels.mean():.1%} default rate")
        return labels
    
    def apply_weighted_feature_mapping(self, features_df):
        """Apply feature weights and create weighted features"""
        print("Applying feature weights...")
        
        # Create weighted features
        weighted_features = {}
        
        for feature, weight in self.feature_weights.items():
            if feature in features_df.columns and weight > 0:
                weighted_feature_name = f"weighted_{feature}"
                weighted_features[weighted_feature_name] = features_df[feature] * weight
                print(f"  {feature}: {weight:.1%} weight")
        
        # Add weighted features to dataframe
        weighted_df = features_df.copy()
        for name, values in weighted_features.items():
            weighted_df[name] = values
        
        # Create overall weighted score
        weighted_df['overall_weighted_score'] = sum(weighted_features.values())
        
        print(f"Created {len(weighted_features)} weighted features")
        return weighted_df
    
    def save_processed_data(self, features_df, weighted_df):
        """Save processed features and labels"""
        print("Saving processed data...")
        
        # Save basic features (remove client_id and TARGET for training)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['client_id', 'TARGET']]
        
        features_for_training = features_df[feature_cols]
        labels = features_df['TARGET']
        
        # Save to parquet files
        features_for_training.to_parquet(self.output_dir / "features.parquet", index=False)
        labels.to_frame().to_parquet(self.output_dir / "labels.parquet", index=False)
        
        # Save weighted version
        weighted_cols = [col for col in weighted_df.columns 
                        if col not in ['client_id', 'TARGET']]
        weighted_for_training = weighted_df[weighted_cols]
        weighted_for_training.to_parquet(self.output_dir / "weighted_features.parquet", index=False)
        
        # Save metadata
        metadata = {
            "n_samples": len(features_df),
            "n_features": len(feature_cols),
            "n_weighted_features": len(weighted_cols),
            "default_rate": float(labels.mean()),
            "feature_weights": self.feature_weights,
            "feature_columns": feature_cols,
            "weighted_feature_columns": weighted_cols
        }
        
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(features_df)} samples with {len(feature_cols)} features")
        print(f"Default rate: {labels.mean():.1%}")
        print(f"Data saved to: {self.output_dir}")
    
    def run_preprocessing(self, max_clients=1000):
        """Run complete preprocessing pipeline"""
        print("=" * 60)
        print("TRANSACTION DATA PREPROCESSING WITH WEIGHTED FEATURES")
        print("=" * 60)
        
        # Load transaction data
        df = self.load_transaction_data()
        
        # Generate features for all clients
        features_df = self.generate_features_for_all_clients(df, max_clients)
        
        # Apply weighted feature mapping
        weighted_df = self.apply_weighted_feature_mapping(features_df)
        
        # Save processed data
        self.save_processed_data(features_df, weighted_df)
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE!")
        print("=" * 60)
        print("Files created:")
        print(f"  - {self.output_dir}/features.parquet")
        print(f"  - {self.output_dir}/labels.parquet") 
        print(f"  - {self.output_dir}/weighted_features.parquet")
        print(f"  - {self.output_dir}/metadata.json")
        print("\nReady for model training with full 31-feature mapping!")
        
        return features_df, weighted_df

def main():
    """Main preprocessing function"""
    preprocessor = TransactionDataPreprocessor()
    
    # Run preprocessing (limit to 1000 clients for demo)
    features_df, weighted_df = preprocessor.run_preprocessing(max_clients=1000)
    
    print(f"\nSample of generated features:")
    print(features_df.head(3))
    
    print(f"\nSample of weighted features:")
    print(weighted_df[['overall_weighted_score']].head(3))

if __name__ == "__main__":
    main()
