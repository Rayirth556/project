"""
investigate_feature_leakage.py
=============================
Investigates why academic_background_tier has 69.6% feature importance.
This is suspicious and likely indicates data leakage or target encoding.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency

BASE_DIR = Path(__file__).parent
FEAT_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
LABEL_PATH = BASE_DIR / "data" / "processed" / "labels.parquet"

def load_data():
    """Load features and labels"""
    X = pd.read_parquet(FEAT_PATH)
    y = pd.read_parquet(LABEL_PATH).squeeze()
    return X, y

def analyze_academic_background():
    """Deep dive into academic_background_tier feature"""
    X, y = load_data()
    
    print("=" * 60)
    print("ACADEMIC BACKGROUND TIER ANALYSIS")
    print("=" * 60)
    
    # Basic stats
    academic = X['academic_background_tier']
    print(f"\nBasic Statistics:")
    print(f"  Unique values: {academic.nunique()}")
    print(f"  Value counts:")
    for val, count in academic.value_counts().sort_index().items():
        print(f"    Tier {int(val)}: {count:,} ({count/len(academic):.1%})")
    
    # Correlation with target
    correlation = academic.corr(y)
    print(f"\nCorrelation with default: {correlation:.4f}")
    
    # Default rate by tier
    print(f"\nDefault rate by academic tier:")
    for tier in sorted(academic.unique()):
        mask = academic == tier
        default_rate = y[mask].mean()
        count = mask.sum()
        print(f"  Tier {int(tier)}: {default_rate:.1%} ({count:,} samples)")
    
    # Chi-square test for independence
    contingency_table = pd.crosstab(academic, y)
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-square test:")
    print(f"  Chi2 statistic: {chi2:.2f}")
    print(f"  P-value: {p_value:.2e}")
    print(f"  Degrees of freedom: {dof}")
    
    # Effect size (Cramér's V)
    n = contingency_table.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * min(contingency_table.shape) - 1))
    print(f"  Cramér's V (effect size): {cramers_v:.3f}")
    
    # Compare with other top features
    print(f"\nComparison with other top features:")
    feature_importance = {
        'academic_background_tier': 0.6956,
        'essential_vs_lifestyle_ratio': 0.0832,
        'vendor_payment_discipline': 0.0306,
        'utility_payment_consistency': 0.0183,
    }
    
    for feature, importance in feature_importance.items():
        if feature in X.columns:
            corr = abs(X[feature].corr(y))
            print(f"  {feature:<30}: corr={corr:.4f}, importance={importance:.4f}")
    
    return academic, y

def check_data_leage_patterns():
    """Check for obvious data leakage patterns"""
    X, y = load_data()
    
    print("\n" + "=" * 60)
    print("DATA LEAKAGE PATTERN CHECK")
    print("=" * 60)
    
    # Check if academic_background_tier is derived from target
    academic = X['academic_background_tier']
    
    # Perfect separation check
    print(f"\nPerfect separation check:")
    for tier in sorted(academic.unique()):
        mask = academic == tier
        defaults = y[mask].sum()
        non_defaults = (~y[mask]).sum()
        
        if defaults == 0:
            print(f"  Tier {int(tier)}: NEVER defaults ({non_defaults:,} non-defaults)")
        elif non_defaults == 0:
            print(f"  Tier {int(tier)}: ALWAYS defaults ({defaults:,} defaults)")
        else:
            print(f"  Tier {int(tier)}: Mixed ({defaults:,} defaults, {non_defaults:,} non-defaults)")
    
    # Check if feature is essentially an encoding of target
    print(f"\nTarget encoding suspicion:")
    unique_combinations = len(set(zip(academic, y)))
    total_samples = len(academic)
    print(f"  Unique (tier, target) combinations: {unique_combinations}")
    print(f"  Total samples: {total_samples}")
    print(f"  Ratio: {unique_combinations/total_samples:.3f}")
    
    if unique_combinations <= 8:  # 4 tiers * 2 targets
        print("  ⚠️  WARNING: Very few unique combinations suggests target encoding!")

def synthetic_feature_test():
    """Create synthetic academic tier to test importance"""
    X, y = load_data()
    
    print("\n" + "=" * 60)
    print("SYNTHETIC FEATURE TEST")
    print("=" * 60)
    
    # Create random academic tier with same distribution
    np.random.seed(42)
    original_academic = X['academic_background_tier']
    synthetic_academic = np.random.choice(
        original_academic.unique(),
        size=len(original_academic),
        p=original_academic.value_counts(normalize=True).values
    )
    
    # Compare correlations
    orig_corr = abs(original_academic.corr(y))
    synth_corr = abs(pd.Series(synthetic_academic).corr(y))
    
    print(f"\nCorrelation with target:")
    print(f"  Original academic_background_tier: {orig_corr:.4f}")
    print(f"  Synthetic (random) tier:          {synth_corr:.4f}")
    print(f"  Ratio: {orig_corr/synth_corr:.1f}x")
    
    if orig_corr > 0.5:
        print("  ⚠️  WARNING: Original correlation is suspiciously high!")

def main():
    """Run complete investigation"""
    try:
        academic, y = analyze_academic_background()
        check_data_leage_patterns()
        synthetic_feature_test()
        
        print("\n" + "=" * 60)
        print("INVESTIGATION SUMMARY")
        print("=" * 60)
        print("1. academic_background_tier shows extremely high feature importance")
        print("2. This suggests either:")
        print("   - Data leakage (feature contains target information)")
        print("   - Target encoding during preprocessing")
        print("   - Feature is essentially a proxy for the target")
        print("3. RECOMMENDATION: Remove or re-engineer this feature")
        
    except Exception as e:
        print(f"Error during investigation: {e}")

if __name__ == "__main__":
    main()
