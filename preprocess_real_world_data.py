"""
preprocess_real_world_data.py
==============================
Converts raw Kaggle datasets into two artefacts:

  1. data/processed/features.parquet  — per-borrower feature vectors (Layer 2 output)
  2. data/processed/labels.parquet    — binary TARGET column (1 = default)

Pipeline:
  Raw Kaggle CSVs  →  synthetic AA transaction DataFrame  →  FeatureStoreMSME  →  feature dict

The trick: since Kaggle datasets don't contain raw bank transactions, we RECONSTRUCT
plausible transaction streams for each borrower from the summary statistics available
(income, instalment history, payment delays, etc.).  This is the bridge between
the aggregate Kaggle features and our FeatureStoreMSME which expects a transaction feed.

This is the correct way to do it — we're not just passing Kaggle features directly
into the model.  We translate them into the *same feature space* our inference pipeline
uses in production, so the model generalises to real AA data.

Run:
    python preprocess_real_world_data.py [--source home_credit|lending_club|indian_loan|all]
                                         [--max-rows 50000]
                                         [--output data/processed]
"""

import argparse
import os
import sys
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from pdr_pipeline.layer_2_feature_engine import FeatureStoreMSME  # noqa: E402

HOME_CREDIT_DIR  = BASE_DIR / "data" / "raw" / "home_credit"
LENDING_CLUB_DIR = BASE_DIR / "data" / "raw" / "lending_club"
INDIAN_LOAN_DIR  = BASE_DIR / "data" / "raw" / "indian_loan"
OUTPUT_DIR       = BASE_DIR / "data" / "processed"


# ══════════════════════════════════════════════════════════════════════════════
#  Synthetic Transaction Reconstructor
# ══════════════════════════════════════════════════════════════════════════════

def _make_date_range(n_months: int = 12, end: str = "2024-01-01") -> pd.DatetimeIndex:
    """Generate n_months of monthly dates ending at 'end'."""
    end_dt = pd.Timestamp(end, tz="UTC")
    return pd.date_range(end=end_dt, periods=n_months * 4, freq="W", tz="UTC")


def build_synthetic_transactions(
    amt_income: float,
    n_months: int = 12,
    n_late_payments: int = 0,
    n_bounced: int = 0,
    cash_dep_ratio: float = 0.05,
    loan_amt: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Reconstruct a plausible weekly transaction stream for one borrower.
    
    This is the BRIDGE: we turn aggregate Kaggle stats into the same
    pd.DataFrame schema that FeatureStoreMSME.generate_feature_vector() consumes.

    Schema:
        Date, Transaction_Type, Amount, Category, Balance
    """
    if rng is None:
        rng = np.random.default_rng()

    dates = _make_date_range(n_months)
    rows = []
    balance = amt_income * 1.5  # starting balance heuristic

    monthly_income = max(amt_income, 1000.0)
    monthly_expenses = monthly_income * rng.uniform(0.55, 0.85)

    late_payment_indices = set(
        rng.choice(len(dates), size=min(n_late_payments, len(dates)), replace=False)
    ) if n_late_payments > 0 else set()

    bounced_indices = set(
        rng.choice(len(dates), size=min(n_bounced, len(dates)), replace=False)
    ) if n_bounced > 0 else set()

    for i, dt in enumerate(dates):
        # ── Income credit (salary / revenue) every 4 weeks ──────────────────
        if i % 4 == 0:
            salary = monthly_income * rng.uniform(0.9, 1.1)
            balance += salary
            rows.append({
                "Date": dt,
                "Transaction_Type": "CREDIT",
                "Amount": round(salary, 2),
                "Category": "Salary",
                "Balance": round(balance, 2),
            })

        # ── Utility payment ──────────────────────────────────────────────────
        if i % 4 == 1:
            utility = monthly_expenses * 0.15 * rng.uniform(0.8, 1.2)
            category = "Utility"
            if i in late_payment_indices:
                # Late payment → penalty row
                penalty = utility * 0.05
                balance -= penalty
                rows.append({
                    "Date": dt,
                    "Transaction_Type": "DEBIT",
                    "Amount": round(penalty, 2),
                    "Category": "Penalty",
                    "Balance": round(balance, 2),
                })
            balance -= utility
            rows.append({
                "Date": dt,
                "Transaction_Type": "DEBIT",
                "Amount": round(utility, 2),
                "Category": category,
                "Balance": round(balance, 2),
            })

        # ── Rent / Lease ─────────────────────────────────────────────────────
        if i % 4 == 2:
            rent = monthly_expenses * 0.30 * rng.uniform(0.95, 1.05)
            balance -= rent
            rows.append({
                "Date": dt,
                "Transaction_Type": "DEBIT",
                "Amount": round(rent, 2),
                "Category": "Rent",
                "Balance": round(balance, 2),
            })

        # ── Loan EMI ─────────────────────────────────────────────────────────
        if loan_amt > 0 and i % 4 == 3:
            emi = (loan_amt / 60) * rng.uniform(0.98, 1.02)  # 5-yr amortisation approx
            if i in bounced_indices:
                # Bounced EMI
                rows.append({
                    "Date": dt,
                    "Transaction_Type": "DEBIT",
                    "Amount": round(emi * 0.02, 2),
                    "Category": "Bounce Charge",
                    "Balance": round(balance, 2),
                })
            else:
                balance -= emi
                rows.append({
                    "Date": dt,
                    "Transaction_Type": "DEBIT",
                    "Amount": round(emi, 2),
                    "Category": "Loan EMI",
                    "Balance": round(balance, 2),
                })

        # ── Discretionary (groceries + dining) ───────────────────────────────
        grocery = monthly_expenses * 0.20 / 4 * rng.uniform(0.7, 1.3)
        balance -= grocery
        rows.append({
            "Date": dt,
            "Transaction_Type": "DEBIT",
            "Amount": round(grocery, 2),
            "Category": "Groceries",
            "Balance": round(balance, 2),
        })

        # ── Cash withdrawal ───────────────────────────────────────────────────
        if rng.random() < cash_dep_ratio:
            cash = monthly_income * 0.05 * rng.uniform(0.5, 2.0)
            balance -= cash
            rows.append({
                "Date": dt,
                "Transaction_Type": "DEBIT",
                "Amount": round(cash, 2),
                "Category": "Cash ATM Withdrawal",
                "Balance": round(balance, 2),
            })

    df = pd.DataFrame(rows)
    # Clamp balance to a small positive so we don't get negative balance artefacts
    df["Balance"] = df["Balance"].clip(lower=100.0)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset-Specific Loaders
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Home Credit ────────────────────────────────────────────────────────────

def load_home_credit(max_rows: int = 50_000) -> pd.DataFrame:
    """
    Load application_train.csv + installments_payments.csv.

    Key columns used:
      AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, TARGET
      → DAYS_INSTALMENT, DAYS_ENTRY_PAYMENT (for delay calc)
    """
    app_path = HOME_CREDIT_DIR / "application_train.csv"
    inst_path = HOME_CREDIT_DIR / "installments_payments.csv"

    if not app_path.exists():
        print(f"  ⚠  Home Credit not found at {app_path} — skipping.")
        return pd.DataFrame()

    print("  Loading application_train.csv …")
    app = pd.read_csv(app_path, nrows=max_rows, low_memory=False)

    delay_by_id = pd.Series(dtype=float, name="avg_payment_delay")
    if inst_path.exists():
        print("  Loading installments_payments.csv …")
        inst = pd.read_csv(inst_path, low_memory=False)
        inst = inst[inst["SK_ID_CURR"].isin(app["SK_ID_CURR"])]
        inst["delay_days"] = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
        delay_by_id = inst.groupby("SK_ID_CURR")["delay_days"].mean()

    app = app.join(delay_by_id, on="SK_ID_CURR")

    # Normalise key columns
    app["amt_income"] = app["AMT_INCOME_TOTAL"].fillna(25_000).clip(5000, 2_000_000)
    app["loan_amt"]   = app["AMT_CREDIT"].fillna(0).clip(0, 10_000_000)
    # If installments_payments.csv was missing or failed to join, the column may not exist
    app["avg_payment_delay"] = app.get("avg_payment_delay", pd.Series(0, index=app.index)).fillna(0).clip(0, 365)
    app["TARGET"]     = app["TARGET"].fillna(0).astype(int)

    # Proxies for late / bounced
    app["n_late"]    = (app["avg_payment_delay"] / 30).round().clip(0, 12).astype(int)
    app["n_bounced"] = app.get("DEF_30_CNT_SOCIAL_CIRCLE", pd.Series(0, index=app.index)).fillna(0).clip(0, 5).astype(int)
    app["cash_dep"]  = (app.get("AMT_REQ_CREDIT_BUREAU_MON", pd.Series(0, index=app.index)).fillna(0) / 100).clip(0, 0.3)

    print(f"  ✓  Home Credit: {len(app):,} rows  (default rate: {app['TARGET'].mean():.1%})")
    app["_source"] = "home_credit"
    return app[["amt_income", "loan_amt", "avg_payment_delay", "n_late", "n_bounced", "cash_dep", "TARGET", "_source"]]


# ── 2. Lending Club ───────────────────────────────────────────────────────────

def load_lending_club(max_rows: int = 50_000) -> pd.DataFrame:
    """
    Lending Club: 2.2M loans with loan_status (Fully Paid / Charged Off / Default).
    Key columns: annual_inc, loan_amnt, delinq_2yrs, out_prncp, tot_coll_amt
    """
    # The file can be called accepted_*.csv, loan.csv, or shipped as a .csv.gz
    # Only keep actual files (ignore directories named like CSVs)
    candidates = [
        p for p in (
            list(LENDING_CLUB_DIR.glob("accepted_*.csv")) +
            list(LENDING_CLUB_DIR.glob("loan.csv")) +
            list(LENDING_CLUB_DIR.glob("*.csv"))
        )
        if p.is_file()
    ]

    compression = None
    if not candidates:
        # Fallback: look for gzipped CSVs from Kaggle (accepted_2007_to_2018Q4.csv.gz, etc.)
        gz_candidates = [
            p for p in LENDING_CLUB_DIR.glob("*.csv.gz")
            if p.is_file()
        ]
        if not gz_candidates:
            print(f"  ⚠  Lending Club CSV not found in {LENDING_CLUB_DIR} — skipping.")
            return pd.DataFrame()
        candidates = gz_candidates
        compression = "gzip"

    csv_path = candidates[0]
    print(f"  Loading {csv_path.name} …")
    df = pd.read_csv(csv_path, nrows=max_rows, low_memory=False, on_bad_lines="skip", compression=compression)

    # Binary target: Charged Off / Default / Late = 1
    if "loan_status" not in df.columns:
        print("  ⚠  loan_status column missing — trying alternative column …")
        return pd.DataFrame()

    df["TARGET"] = df["loan_status"].isin(["Charged Off", "Default", "Late (31-120 days)", "Late (16-30 days)"]).astype(int)

    df["amt_income"]  = pd.to_numeric(df.get("annual_inc", 0), errors="coerce").fillna(30_000).clip(5000, 2_000_000) / 12
    df["loan_amt"]    = pd.to_numeric(df.get("loan_amnt", 0), errors="coerce").fillna(0)
    df["n_late"]      = pd.to_numeric(df.get("delinq_2yrs", 0), errors="coerce").fillna(0).clip(0, 12).astype(int)
    df["n_bounced"]   = (df["n_late"] // 3).astype(int)
    df["cash_dep"]    = pd.to_numeric(df.get("revol_util", 0), errors="coerce").fillna(0).clip(0, 100) / 1000  # very small ratio
    df["avg_payment_delay"] = df["n_late"] * 15  # approx 15 days per delinquency
    df["_source"] = "lending_club"

    print(f"  ✓  Lending Club: {len(df):,} rows  (default rate: {df['TARGET'].mean():.1%})")
    return df[["amt_income", "loan_amt", "avg_payment_delay", "n_late", "n_bounced", "cash_dep", "TARGET", "_source"]]


# ── 3. Indian Loan Default ────────────────────────────────────────────────────

def load_indian_loan(max_rows: int = 50_000) -> pd.DataFrame:
    """
    Indian loan default synthetic dataset.  Column names vary by specific file.
    We try multiple common schemas seen on Kaggle.

    NOTE: Prefer train.csv (labelled) over test.csv if both exist.
    """
    train_path = INDIAN_LOAN_DIR / "train.csv"
    if train_path.exists():
        csv_path = train_path
    else:
        candidates = list(INDIAN_LOAN_DIR.glob("*.csv"))
        if not candidates:
            print(f"  ⚠  Indian Loan dataset not found in {INDIAN_LOAN_DIR} — skipping.")
            return pd.DataFrame()
        csv_path = candidates[0]

    print(f"  Loading {csv_path.name} …")
    df = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)

    # Detect target column
    target_col = None
    for col in ["Risk_Flag", "loan_status", "Loan Status", "default", "Default", "TARGET", "target", "label"]:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        # Last resort: use last column if it looks binary
        last_col = df.columns[-1]
        if df[last_col].nunique() == 2:
            target_col = last_col
    if target_col is None:
        print("  ⚠  Cannot identify target column — skipping Indian Loan dataset.")
        return pd.DataFrame()

    df["TARGET"] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).clip(0, 1).astype(int)

    # Income
    income_col = next((c for c in ["Income", "annual_inc", "INCOME", "ApplicantIncome", "income", "AMT_INCOME"] if c in df.columns), None)
    df["amt_income"] = pd.to_numeric(df[income_col], errors="coerce").fillna(20_000).clip(1000, 2_000_000) if income_col else 20_000.0

    # Loan amount
    loan_col = next((c for c in ["LOAN", "LoanAmount", "loan_amnt", "LoanAmnt", "Loan_amount"] if c in df.columns), None)
    df["loan_amt"] = pd.to_numeric(df[loan_col], errors="coerce").fillna(0) if loan_col else 0.0

    df["n_late"]    = 0
    df["n_bounced"] = 0
    df["cash_dep"]  = 0.05
    df["avg_payment_delay"] = 0.0
    df["_source"]   = "indian_loan"

    print(f"  ✓  Indian Loan: {len(df):,} rows  (default rate: {df['TARGET'].mean():.1%})")
    return df[["amt_income", "loan_amt", "avg_payment_delay", "n_late", "n_bounced", "cash_dep", "TARGET", "_source"]]


# ══════════════════════════════════════════════════════════════════════════════
#  Feature Engineering via Layer 2 FeatureStoreMSME
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_UI = {
    "avg_utility_dpd":             0.0,
    "telecom_number_vintage_days": 730.0,
    "academic_background_tier":    3.0,
    "purpose_of_loan":             "Working Capital",
    "avg_invoice_payment_delay":   0.0,
    "vendor_payment_discipline_dpd": 0.0,
    "gst_filing_consistency_score": 6.0,
    "identity_device_mismatch_flag": 0.0,
    "declared_gst_revenue":        0.0,
}


def engineer_features_for_row(row: pd.Series, rng: np.random.Generator) -> Optional[dict]:
    """
    For one borrower row from any Kaggle dataset:
      1. Reconstruct synthetic AA transaction DataFrame
      2. Build ui_data from available signals
      3. Run FeatureStoreMSME.generate_feature_vector()
      4. Return the feature dict
    """
    try:
        aa_df = build_synthetic_transactions(
            amt_income   = float(row["amt_income"]),
            n_months     = 12,
            n_late_payments = int(row["n_late"]),
            n_bounced    = int(row["n_bounced"]),
            cash_dep_ratio = float(row["cash_dep"]),
            loan_amt     = float(row["loan_amt"]),
            rng          = rng,
        )

        ui_data = dict(_DEFAULT_UI)
        ui_data["avg_invoice_payment_delay"] = float(row["avg_payment_delay"])
        ui_data["avg_utility_dpd"] = float(min(row["avg_payment_delay"] / 2, 30))
        # Declared GST revenue ≈ 10x monthly income (rough SME proxy)
        ui_data["declared_gst_revenue"] = float(row["amt_income"]) * 10

        fs = FeatureStoreMSME(aa_df, ui_data)
        return fs.generate_feature_vector()

    except Exception as exc:
        # Silently drop bad rows — they'll be NaN filled later
        return None


def process_dataset(df: pd.DataFrame, max_rows: int, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Engineer features for every row in df using Layer 2."""
    rng = np.random.default_rng(seed)
    df = df.sample(min(max_rows, len(df)), random_state=seed).reset_index(drop=True)

    feature_rows = []
    labels = []
    n_failed = 0

    total = len(df)
    print(f"  Engineering features for {total:,} borrowers via Layer 2 …")

    for i, (_, row) in enumerate(df.iterrows()):
        if (i + 1) % 1000 == 0 or i == total - 1:
            pct = (i + 1) / total * 100
            print(f"    {i+1:,}/{total:,}  ({pct:.0f}%)", end="\r")

        feat = engineer_features_for_row(row, rng)
        if feat is None:
            n_failed += 1
            continue

        feature_rows.append(feat)
        labels.append(int(row["TARGET"]))

    print()  # newline after \r
    print(f"  ✓  Features engineered: {len(feature_rows):,}  (failed: {n_failed})")

    features_df = pd.DataFrame(feature_rows)
    labels_s    = pd.Series(labels, name="TARGET")
    return features_df, labels_s


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Kaggle datasets for PDR Layer 3 training.")
    parser.add_argument("--source", choices=["home_credit", "lending_club", "indian_loan", "all"],
                        default="all", help="Which dataset(s) to process")
    parser.add_argument("--max-rows", type=int, default=30_000,
                        help="Max rows per dataset (default 30,000 for speed)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for features.parquet and labels.parquet")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 62)
    print("  PDR Pipeline — Real-World Preprocessing (Layer 1→2 Bridge)")
    print("=" * 62)

    # Load raw data
    dfs = []
    if args.source in ("home_credit", "all"):
        print("\n[1/3] Home Credit Default Risk")
        hc = load_home_credit(args.max_rows)
        if not hc.empty:
            dfs.append(hc)

    if args.source in ("lending_club", "all"):
        print("\n[2/3] Lending Club")
        lc = load_lending_club(args.max_rows)
        if not lc.empty:
            dfs.append(lc)

    if args.source in ("indian_loan", "all"):
        print("\n[3/3] Indian Loan Default")
        il = load_indian_loan(args.max_rows)
        if not il.empty:
            dfs.append(il)

    if not dfs:
        print("\n❌  No data loaded. Run download_datasets.py first.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n[Combined] {len(combined):,} borrowers  (default rate: {combined['TARGET'].mean():.1%})")
    print(f"  Sources: {combined['_source'].value_counts().to_dict()}")

    print("\n[Layer 2] Feature Engineering ──────────────────────────────")
    features_df, labels_s = process_dataset(combined, args.max_rows * len(dfs))

    # Save
    feat_path   = output_dir / "features.parquet"
    label_path  = output_dir / "labels.parquet"
    meta_path   = output_dir / "metadata.json"

    features_df.to_parquet(feat_path, index=False)
    labels_s.to_frame().to_parquet(label_path, index=False)

    metadata = {
        "n_samples":      len(features_df),
        "n_features":     len(features_df.columns),
        "feature_names":  list(features_df.columns),
        "default_rate":   float(labels_s.mean()),
        "sources":        combined["_source"].value_counts().to_dict(),
        "features_path":  str(feat_path),
        "labels_path":    str(label_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"\n✅  Output saved:")
    print(f"   Features : {feat_path}  ({len(features_df):,} rows × {len(features_df.columns)} cols)")
    print(f"   Labels   : {label_path}")
    print(f"   Metadata : {meta_path}")
    print(f"\n   Default rate: {labels_s.mean():.1%}")
    print(f"\n   Next step: python train_real_world_model.py")


if __name__ == "__main__":
    main()
