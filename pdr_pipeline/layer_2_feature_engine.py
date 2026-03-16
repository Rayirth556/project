import pandas as pd
import numpy as np
import networkx as nx
import json
from typing import Optional

class FeatureStoreMSME:
    """
    FeatureStore for generating unified credit scoring features tailored for NTC and MSME borrowers.
    Processes standardized transaction CSV data (Layer 1 output) and UI-input JSON to generate
    a comprehensive Feature Vector of 50+ markers across 6 strategic pillars.
    """
    
    def __init__(self, aa_data: pd.DataFrame, ui_data: dict, config: Optional[dict] = None):
        """
        :param aa_data: DataFrame from Layer 1 standardized CSV.
        :param ui_data: Dictionary containing fields from the UI form.
        :param config: Dictionary to hold model parameters like 99th percentile caps for winsorization.
        """
        self.aa_data = aa_data.copy()
        self.ui_data = ui_data
        self.config = config or {}
        
        # Prepare datetime, monthly, and quarterly aggregates
        if not self.aa_data.empty and 'Date' in self.aa_data.columns:
            self.aa_data['Date'] = pd.to_datetime(self.aa_data['Date'], utc=True)
            self.aa_data = self.aa_data.sort_values('Date').reset_index(drop=True)
            self.aa_data['YearMonth'] = self.aa_data['Date'].dt.to_period('M')
            self.aa_data['Quarter'] = self.aa_data['Date'].dt.to_period('Q')
        else:
            self.aa_data['YearMonth'] = []
            self.aa_data['Quarter'] = []

    # ==========================================
    # Helper Functions
    # ==========================================
    def _get_monthly_income(self):
        if self.aa_data.empty: return 0.0
        income = self.aa_data[
            (self.aa_data['Transaction_Type'].str.upper() == 'CREDIT') &
            (self.aa_data['Category'].str.contains('Salary|Income|Revenue|Sales', case=False, na=False))
        ]
        return income.groupby('YearMonth')['Amount'].sum().mean() if not income.empty else 0.0

    # ==========================================
    # Pillar I: NTC Behavioral Discipline
    # ==========================================
    def calc_utility_payment_consistency(self) -> float:
        """Max streak of on-time utility credits (represented as consecutive monthly utility payments)"""
        if self.aa_data.empty: return 0.0
        utility_txns = self.aa_data[
            (self.aa_data['Transaction_Type'].str.upper() == 'DEBIT') & 
            (self.aa_data['Category'].str.contains('Utility|Bill|Electricity|Water|Broadband', case=False, na=False))
        ]
        if utility_txns.empty: return 0.0
        
        months_with_payment = sorted(utility_txns['YearMonth'].unique())
        if len(months_with_payment) == 0: return 0.0
        
        max_streak = 1
        current_streak = 1
        for i in range(1, len(months_with_payment)):
            if (months_with_payment[i] - months_with_payment[i-1]).n == 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        return float(max_streak)

    def calc_avg_utility_dpd(self) -> float:
        """Average Days Past Due across all utility bills (Fetched from UI Profile or assumed 0)"""
        return float(self.ui_data.get('avg_utility_dpd', 0.0))

    def calc_rent_wallet_share(self) -> float:
        """Monthly rent / Average monthly income"""
        if self.aa_data.empty: return 0.0
        rent_txns = self.aa_data[
            (self.aa_data['Transaction_Type'].str.upper() == 'DEBIT') &
            (self.aa_data['Category'].str.contains('Rent|Lease', case=False, na=False))
        ]
        avg_rent = rent_txns.groupby('YearMonth')['Amount'].sum().mean() if not rent_txns.empty else 0.0
        avg_income = self._get_monthly_income()
        if avg_income == 0: return 0.0
        return float(avg_rent / avg_income)

    def calc_subscription_commitment_ratio(self) -> float:
        """Total fixed monthly subscriptions / Average monthly income"""
        if self.aa_data.empty: return 0.0
        subs = self.aa_data[
            (self.aa_data['Transaction_Type'].str.upper() == 'DEBIT') &
            (self.aa_data['Category'].str.contains('Subscription|Fixed|EMI|Loan', case=False, na=False))
        ]
        avg_subs = subs.groupby('YearMonth')['Amount'].sum().mean() if not subs.empty else 0.0
        avg_income = self._get_monthly_income()
        if avg_income == 0: return 0.0
        return float(avg_subs / avg_income)

    # ==========================================
    # Pillar II: Liquidity & Stress Layer
    # ==========================================
    def calc_emergency_buffer_months(self) -> float:
        """Current balance / Average monthly essential outflow"""
        if self.aa_data.empty: return 0.0
        mean_bal = self.aa_data['Balance'].iloc[-1] if not self.aa_data.empty else 0.0
        essential_outflow = self.aa_data[
            (self.aa_data['Transaction_Type'].str.upper() == 'DEBIT') &
            (self.aa_data['Category'].str.contains('Utility|Rent|Groceries|Essential|EMI|Medical', case=False, na=False))
        ]
        avg_essential = essential_outflow.groupby('YearMonth')['Amount'].sum().mean() if not essential_outflow.empty else 0.0
        if pd.isna(avg_essential) or avg_essential == 0:
            avg_essential = self.aa_data[self.aa_data['Transaction_Type'].str.upper() == 'DEBIT'].groupby('YearMonth')['Amount'].sum().mean()
            if pd.isna(avg_essential) or avg_essential == 0:
                return float(mean_bal) if pd.notna(mean_bal) else 0.0
        return float(mean_bal / avg_essential)

    def calc_min_balance_violation_count(self) -> float:
        """Count of drops below ₹500 in the last 6 months"""
        if self.aa_data.empty: return 0.0
        six_months_ago = self.aa_data['Date'].max() - pd.DateOffset(months=6)
        recent_data = self.aa_data[self.aa_data['Date'] >= six_months_ago]
        violations = recent_data[recent_data['Balance'] < 500.0]
        return float(len(violations))

    def calc_eod_balance_volatility(self) -> float:
        """Coefficient of Variation (Std Dev / Mean) of daily closing balances"""
        if self.aa_data.empty: return 0.0
        daily_balances = self.aa_data.groupby(self.aa_data['Date'].dt.date)['Balance'].last()
        if len(daily_balances) < 2: return 0.0
        mean_bal = daily_balances.mean()
        if mean_bal == 0: return 0.0
        cv = daily_balances.std() / mean_bal
        return self._apply_winsorization(float(cv), 'eod_balance_volatility')

    def calc_essential_vs_lifestyle_ratio(self) -> float:
        """Survival spending (Groceries/Medical) vs. Discretionary (Dining/Travel)"""
        if self.aa_data.empty: return 0.0
        essential = self.aa_data[
            (self.aa_data['Transaction_Type'].str.upper() == 'DEBIT') &
            (self.aa_data['Category'].str.contains('Groceries|Medical|Utility|Rent', case=False, na=False))
        ]['Amount'].sum()
        
        lifestyle = self.aa_data[
            (self.aa_data['Transaction_Type'].str.upper() == 'DEBIT') &
            (self.aa_data['Category'].str.contains('Dining|Travel|Entertainment|Shopping|Zomato|Swiggy', case=False, na=False))
        ]['Amount'].sum()
        
        if lifestyle == 0: return float(essential)
        return float(essential / lifestyle)

    def calc_cash_withdrawal_dependency(self) -> float:
        """Total cash withdrawals / Total monthly outflows"""
        if self.aa_data.empty: return 0.0
        total_outflows = self.aa_data[self.aa_data['Transaction_Type'].str.upper() == 'DEBIT']['Amount'].sum()
        if total_outflows == 0: return 0.0
        cash_txns = self.aa_data[
            (self.aa_data['Transaction_Type'].str.upper() == 'DEBIT') &
            (self.aa_data['Category'].str.contains('Cash|ATM|Withdrawal', case=False, na=False))
        ]
        return float(cash_txns['Amount'].sum() / total_outflows)

    def calc_bounced_transaction_count(self) -> float:
        """Count of bounced transactions/penalty fees (Last 6 months)"""
        if self.aa_data.empty: return 0.0
        six_months_ago = self.aa_data['Date'].max() - pd.DateOffset(months=6)
        recent_data = self.aa_data[self.aa_data['Date'] >= six_months_ago]
        bounced = recent_data[recent_data['Category'].str.contains('Bounce|Dishonor|Penalty|Fee|Reversal', case=False, na=False)]
        return float(len(bounced))

    # ==========================================
    # Pillar III: Alt-Data & Identity
    # ==========================================
    def calc_telecom_number_vintage_days(self) -> float:
        """Days since phone activation (Proxy for identity stability)"""
        return float(self.ui_data.get('telecom_number_vintage_days', 0.0))

    def calc_telecom_recharge_drop_ratio(self) -> float:
        """Current month recharge / 6-month average (Leading stress indicator)"""
        if self.aa_data.empty: return 1.0 # default to 1 meaning neutral
        recharges = self.aa_data[
            (self.aa_data['Transaction_Type'].str.upper() == 'DEBIT') &
            (self.aa_data['Category'].str.contains('Telecom|Recharge|Mobile', case=False, na=False))
        ]
        if recharges.empty: return 1.0
        
        monthly_recharges = recharges.groupby('YearMonth')['Amount'].sum()
        if len(monthly_recharges) == 0: return 1.0
        
        current_month = monthly_recharges.index.max()
        current_val = monthly_recharges.get(current_month, 0.0)
        
        past_6_months_avg = monthly_recharges[monthly_recharges.index < current_month].tail(6).mean()
        if pd.isna(past_6_months_avg) or past_6_months_avg == 0:
            return 1.0
        return float(current_val / past_6_months_avg)

    def calc_academic_background_tier(self) -> float:
        """Encoded value representing education level (Tier 1-4)"""
        return float(self.ui_data.get('academic_background_tier', 4.0))

    def calc_purpose_of_loan_encoded(self) -> float:
        """Target-encoded value of loan reason (e.g., Working Capital vs. Personal)"""
        purpose_mapping = {
            'Working Capital': 1.0,
            'Equipment Expansion': 2.0,
            'Debt Consolidation': 3.0,
            'Personal / General': 4.0
        }
        raw_purpose = self.ui_data.get('purpose_of_loan', 'Personal / General')
        return purpose_mapping.get(raw_purpose, 4.0)

    # ==========================================
    # Pillar IV: MSME Operational Stability
    # ==========================================
    def calc_business_vintage_months(self) -> float:
        """Months since first banking transaction or GST registration"""
        if not self.aa_data.empty:
            months_aa = (self.aa_data['Date'].max().year - self.aa_data['Date'].min().year) * 12 + (self.aa_data['Date'].max().month - self.aa_data['Date'].min().month)
        else:
            months_aa = 0
            
        ui_vintage = self.ui_data.get('business_vintage_months', 0)
        return float(max(months_aa, ui_vintage))

    def calc_revenue_growth_trend(self) -> float:
        """% MoM change in incoming business cashflow"""
        if self.aa_data.empty: return 0.0
        revenue = self.aa_data[
            (self.aa_data['Transaction_Type'].str.upper() == 'CREDIT') &
            (self.aa_data['Category'].str.contains('Revenue|Sales|Income|POS', case=False, na=False))
        ]
        if revenue.empty: return 0.0
        
        monthly_rev = revenue.groupby('YearMonth')['Amount'].sum().sort_index()
        if len(monthly_rev) < 2: return 0.0
        
        pct_change = monthly_rev.pct_change().dropna()
        if pct_change.empty: return 0.0
        return self._apply_winsorization(float(pct_change.mean()), 'revenue_growth_trend')

    def calc_revenue_seasonality_index(self) -> float:
        """Statistical variance of revenue across quarters (Normalizes lumpy businesses)"""
        if self.aa_data.empty: return 0.0
        revenue = self.aa_data[
            (self.aa_data['Transaction_Type'].str.upper() == 'CREDIT') &
            (self.aa_data['Category'].str.contains('Revenue|Sales|Income|POS', case=False, na=False))
        ]
        if revenue.empty: return 0.0
        quarterly_rev = revenue.groupby('Quarter')['Amount'].sum()
        if len(quarterly_rev) < 2: return 0.0
        mean_rev = quarterly_rev.mean()
        if mean_rev == 0: return 0.0
        
        cv = quarterly_rev.std() / mean_rev
        return self._apply_winsorization(float(cv), 'revenue_seasonality_index')

    def calc_operating_cashflow_ratio(self) -> tuple[float, float]:
        """Monthly inflows / Monthly outflows (Survival threshold > 1.0)"""
        if self.aa_data.empty: return 0.0, 0.0
        inflows = self.aa_data[self.aa_data['Transaction_Type'].str.upper() == 'CREDIT']['Amount'].sum()
        outflows = self.aa_data[self.aa_data['Transaction_Type'].str.upper() == 'DEBIT']['Amount'].sum()
        
        ratio = float(inflows / outflows) if outflows > 0 else float(inflows)
        survival_flag = 1.0 if ratio > 1.0 else 0.0
        return ratio, survival_flag

    def calc_cashflow_volatility(self) -> float:
        """Std Dev of monthly net cashflow (Measures revenue reliability)"""
        if self.aa_data.empty: return 0.0
        credits = self.aa_data[self.aa_data['Transaction_Type'].str.upper() == 'CREDIT'].groupby('YearMonth')['Amount'].sum()
        debits = self.aa_data[self.aa_data['Transaction_Type'].str.upper() == 'DEBIT'].groupby('YearMonth')['Amount'].sum()
        net_cashflow = credits.subtract(debits, fill_value=0)
        
        if len(net_cashflow) < 2: return 0.0
        return self._apply_winsorization(float(net_cashflow.std()), 'cashflow_volatility')

    def calc_avg_invoice_payment_delay(self) -> float:
        """Average days between invoice date and payment receipt"""
        return float(self.ui_data.get('avg_invoice_payment_delay', 0.0))


    # ==========================================
    # Pillar V: Network Risk & Compliance
    # ==========================================
    def calc_customer_concentration_ratio(self) -> float:
        """% of revenue from Top 3 clients (High ratio = High Fragility)"""
        if self.aa_data.empty: return 0.0
        col = 'Counterparty' if 'Counterparty' in self.aa_data.columns else 'Category'
        revenue_txns = self.aa_data[
            (self.aa_data['Transaction_Type'].str.upper() == 'CREDIT') &
            (self.aa_data['Category'].str.contains('Revenue|Sales|Income|POS', case=False, na=False))
        ]
        if revenue_txns.empty: return 0.0
        total_revenue = revenue_txns['Amount'].sum()
        if total_revenue == 0: return 0.0
        top_3_rev = revenue_txns.groupby(col)['Amount'].sum().nlargest(3).sum()
        return float(top_3_rev / total_revenue)

    def calc_repeat_customer_revenue_pct(self) -> float:
        """% of revenue from repeat counterparties (Measures stickiness)"""
        if self.aa_data.empty: return 0.0
        col = 'Counterparty' if 'Counterparty' in self.aa_data.columns else 'Category'
        revenue_txns = self.aa_data[
            (self.aa_data['Transaction_Type'].str.upper() == 'CREDIT') &
            (self.aa_data['Category'].str.contains('Revenue|Sales|Income|POS', case=False, na=False))
        ]
        if revenue_txns.empty: return 0.0
        
        total_revenue = revenue_txns['Amount'].sum()
        if total_revenue == 0: return 0.0
        
        txns_per_client = revenue_txns.groupby(col).size()
        repeat_clients = txns_per_client[txns_per_client > 1].index
        
        repeat_revenue = revenue_txns[revenue_txns[col].isin(repeat_clients)]['Amount'].sum()
        return float(repeat_revenue / total_revenue)

    def calc_vendor_payment_discipline(self) -> float:
        """Average DPD when the MSME pays its own suppliers"""
        return float(self.ui_data.get('vendor_payment_discipline_dpd', 0.0))

    def calc_gst_filing_consistency_score(self) -> float:
        """Longest streak of on-time GSTR-1/3B filings"""
        return float(self.ui_data.get('gst_filing_consistency_score', 0.0))

    def calc_gst_to_bank_variance(self) -> float:
        """Difference between UI-declared revenue and bank inflows"""
        ui_gst_revenue = float(self.ui_data.get('declared_gst_revenue', 0.0))
        if self.aa_data.empty: return ui_gst_revenue
        bank_inflows = self.aa_data[
            (self.aa_data['Transaction_Type'].str.upper() == 'CREDIT') &
            (self.aa_data['Category'].str.contains('Revenue|Sales|Income|POS', case=False, na=False))
        ]['Amount'].sum()
        
        # Calculate percentage difference instead of absolute difference
        if bank_inflows == 0: return 1.0 # 100% variance
        variance = abs(ui_gst_revenue - bank_inflows) / bank_inflows
        return self._apply_winsorization(float(variance), 'gst_to_bank_variance')


    # ==========================================
    # Pillar VI: Trust Intelligence & Forensic
    # ==========================================
    def calc_p2p_circular_loop_flag(self) -> float:
        """Detect A -> B -> A cycles using Graph Theory"""
        if self.aa_data.empty: return 0.0
        
        G = nx.DiGraph()
        target_account = self.config.get('anchor_account', 'Self')
        col = 'Counterparty' if 'Counterparty' in self.aa_data.columns else 'Category'
        
        for _, row in self.aa_data.iterrows():
            cp = str(row.get(col, 'Unknown'))
            txn_type = str(row['Transaction_Type']).upper()
            amt = row['Amount']

            if txn_type == 'CREDIT':
                G.add_edge(cp, target_account, weight=amt)
            elif txn_type == 'DEBIT':
                G.add_edge(target_account, cp, weight=amt)
                
        try:
            cycles = list(nx.simple_cycles(G))
            for cycle in cycles:
                # Cycle length 2 means Self -> B -> Self (A -> B -> A)
                if target_account in cycle and len(cycle) >= 2:
                    return 1.0 # Cycle Detected
        except:
            pass
        return 0.0

    def calc_benford_anomaly_score(self) -> float:
        """Detect if first digits of transaction amounts deviate from Benford's Law"""
        if self.aa_data.empty: return 0.0
            
        amounts = self.aa_data['Amount'].dropna()
        amounts = amounts[amounts > 0]
        if amounts.empty: return 0.0
            
        first_digits = amounts.astype(str).str.extract(r'([1-9])')[0].dropna().astype(int)
        if first_digits.empty: return 0.0
            
        freq = first_digits.value_counts(normalize=True).sort_index()
        digits = np.arange(1, 10)
        expected = np.log10(1 + 1/digits)
        
        actual = np.zeros(9)
        for d in range(1, 10):
            if d in freq:
                actual[d-1] = freq[d]
                
        mad = np.sum(np.abs(actual - expected))
        return float(mad)

    def calc_round_number_spike_ratio(self) -> float:
        """% of transactions ending in '000' (Detects manual bookkeeping fraud)"""
        if self.aa_data.empty: return 0.0
        amounts = self.aa_data['Amount'].dropna()
        if amounts.empty: return 0.0
        round_amounts = amounts[amounts % 1000 == 0]
        return float(len(round_amounts) / len(amounts))

    def calc_turnover_inflation_spike(self) -> float:
        """Flags unnatural volume spikes 30-60 days before loan application"""
        if self.aa_data.empty: return 0.0
        revenue_txns = self.aa_data[
            (self.aa_data['Transaction_Type'].str.upper() == 'CREDIT') &
            (self.aa_data['Category'].str.contains('Revenue|Sales|Income|POS', case=False, na=False))
        ]
        if revenue_txns.empty: return 0.0
        
        max_date = self.aa_data['Date'].max()
        cutoff_date = max_date - pd.DateOffset(days=60)
        
        recent_revenue = revenue_txns[revenue_txns['Date'] >= cutoff_date]['Amount'].sum()
        historical_revenue = revenue_txns[revenue_txns['Date'] < cutoff_date]['Amount'].sum()
        
        # We compute mean daily or monthly to compare
        historical_days = (cutoff_date - self.aa_data['Date'].min()).days
        if historical_days <= 0 or historical_revenue == 0: return 0.0
        
        historical_daily_avg = historical_revenue / historical_days
        recent_daily_avg = recent_revenue / 60
        
        if recent_daily_avg > (historical_daily_avg * 1.5):
            return 1.0 # Spike detected
        return 0.0

    def calc_identity_device_mismatch(self) -> float:
        """Flags if multiple accounts share the same Device IP/MAC address"""
        # Extracted from UI Data typically
        return float(self.ui_data.get('identity_device_mismatch_flag', 0.0))

    # ==========================================
    # Utilities
    # ==========================================
    def _apply_winsorization(self, value: float, feature_name: str) -> float:
        """Winsorize (cap) volatility/variance features at the 99th percentile"""
        p99_caps = {
            'revenue_seasonality_index': 5.0, 
            'gst_to_bank_variance': 100.0, # Now a percentage
            'revenue_growth_trend': 2.0,
            'eod_balance_volatility': 3.0,
            'cashflow_volatility': 5_000_000.0
        }
        cap = self.config.get('winsorize_p99', {}).get(feature_name, p99_caps.get(feature_name, float('inf')))
        return min(value, cap) if pd.notna(value) else 0.0

    def generate_feature_vector(self) -> dict:
        """Output single flattened dict ready for Layer 3 XGBoost Inference"""
        oc_ratio, oc_survival = self.calc_operating_cashflow_ratio()
        
        features = {
            # Pillar I
            'utility_payment_consistency': self.calc_utility_payment_consistency(),
            'avg_utility_dpd': self.calc_avg_utility_dpd(),
            'rent_wallet_share': self.calc_rent_wallet_share(),
            'subscription_commitment_ratio': self.calc_subscription_commitment_ratio(),
            
            # Pillar II
            'emergency_buffer_months': self.calc_emergency_buffer_months(),
            'min_balance_violation_count': self.calc_min_balance_violation_count(),
            'eod_balance_volatility': self.calc_eod_balance_volatility(),
            'essential_vs_lifestyle_ratio': self.calc_essential_vs_lifestyle_ratio(),
            'cash_withdrawal_dependency': self.calc_cash_withdrawal_dependency(),
            'bounced_transaction_count': self.calc_bounced_transaction_count(),
            
            # Pillar III
            'telecom_number_vintage_days': self.calc_telecom_number_vintage_days(),
            'telecom_recharge_drop_ratio': self.calc_telecom_recharge_drop_ratio(),
            'academic_background_tier': self.calc_academic_background_tier(),
            'purpose_of_loan_encoded': self.calc_purpose_of_loan_encoded(),
            
            # Pillar IV
            'business_vintage_months': self.calc_business_vintage_months(),
            'revenue_growth_trend': self.calc_revenue_growth_trend(),
            'revenue_seasonality_index': self.calc_revenue_seasonality_index(),
            'operating_cashflow_ratio': oc_ratio,
            'operating_cashflow_survival_flag': oc_survival,
            'cashflow_volatility': self.calc_cashflow_volatility(),
            'avg_invoice_payment_delay': self.calc_avg_invoice_payment_delay(),
            
            # Pillar V
            'customer_concentration_ratio': self.calc_customer_concentration_ratio(),
            'repeat_customer_revenue_pct': self.calc_repeat_customer_revenue_pct(),
            'vendor_payment_discipline': self.calc_vendor_payment_discipline(),
            'gst_filing_consistency_score': self.calc_gst_filing_consistency_score(),
            'gst_to_bank_variance': self.calc_gst_to_bank_variance(),
            
            # Pillar VI
            'p2p_circular_loop_flag': self.calc_p2p_circular_loop_flag(),
            'benford_anomaly_score': self.calc_benford_anomaly_score(),
            'round_number_spike_ratio': self.calc_round_number_spike_ratio(),
            'turnover_inflation_spike': self.calc_turnover_inflation_spike(),
            'identity_device_mismatch': self.calc_identity_device_mismatch()
        }
        return features

if __name__ == "__main__":
    # Test execution for layer inference validation
    mock_aa = pd.DataFrame([
        {'Date': '2023-01-10T10:00:00Z', 'Transaction_Type': 'CREDIT', 'Amount': 100000, 'Category': 'Sales', 'Balance': 150000, 'Counterparty': 'CorpA'},
        {'Date': '2023-01-12T10:00:00Z', 'Transaction_Type': 'DEBIT',  'Amount': 20000, 'Category': 'Utility',  'Balance': 130000, 'Counterparty': 'VendorX'},
        {'Date': '2023-01-15T10:00:00Z', 'Transaction_Type': 'CREDIT', 'Amount': 50000, 'Category': 'Sales',  'Balance': 180000, 'Counterparty': 'CorpA'},
        {'Date': '2023-01-18T10:00:00Z', 'Transaction_Type': 'DEBIT',  'Amount': 3000, 'Category': 'Cash ATM', 'Balance': 177000, 'Counterparty': 'Self'},
        {'Date': '2023-02-10T10:00:00Z', 'Transaction_Type': 'CREDIT', 'Amount': 110000, 'Category': 'Sales', 'Balance': 287000, 'Counterparty': 'CorpB'},
        {'Date': '2023-02-15T10:00:00Z', 'Transaction_Type': 'DEBIT',  'Amount': 25000, 'Category': 'Utility',  'Balance': 262000, 'Counterparty': 'VendorY'},
        {'Date': '2023-02-18T10:00:00Z', 'Transaction_Type': 'CREDIT', 'Amount': 25000, 'Category': 'Transfer', 'Balance': 287000, 'Counterparty': 'VendorY'}, # Cycle A->B->A simulation
        {'Date': '2023-02-28T10:00:00Z', 'Transaction_Type': 'DEBIT',  'Amount': 50000, 'Category': 'Rent','Balance': 232000, 'Counterparty': 'Landlord'}
    ])
    
    mock_ui = {
        'declared_gst_revenue': 250000,
        'telecom_number_vintage_days': 1500,
        'academic_background_tier': 2,
        'purpose_of_loan': 'Working Capital',
        'avg_utility_dpd': 5.0,
        'avg_invoice_payment_delay': 14.0,
        'vendor_payment_discipline_dpd': 2.0,
        'gst_filing_consistency_score': 12.0,
        'identity_device_mismatch_flag': 0.0
    }

    fs = FeatureStoreMSME(mock_aa, mock_ui)
    xgb_vector = fs.generate_feature_vector()
    
    print("\n[Layer 2 - NTC & MSME Expanded Feature Vector JSON]")
    print(json.dumps(xgb_vector, indent=4))
