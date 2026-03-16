"""
download_datasets.py
====================
One-shot downloader for the three real-world credit scoring datasets needed
to train the PDR pipeline's Layer 3 XGBoost model.

Datasets:
  1. Home Credit Default Risk  (kaggle.com/c/home-credit-default-risk)
  2. Lending Club Loan Data     (kaggle.com/datasets/wordsforthewise/lending-club)
  3. Indian Loan Default        (kaggle.com/datasets/hemanthsai7/loandefault)

SETUP (one-time):
  1. Go to https://www.kaggle.com/settings → Account → API → Create New Token
  2. This downloads a kaggle.json file.
  3. Copy it to:  C:\\Users\\<you>\\.kaggle\\kaggle.json
  4. Then run:    python download_datasets.py
"""

import os
import sys
import json
import zipfile
import shutil
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
RAW_DIR       = BASE_DIR / "data" / "raw"
HOME_CREDIT   = RAW_DIR / "home_credit"
LENDING_CLUB  = RAW_DIR / "lending_club"
INDIAN_LOAN   = RAW_DIR / "indian_loan"
KAGGLE_JSON   = Path.home() / ".kaggle" / "kaggle.json"

DATASETS = [
    {
        "name": "Home Credit Default Risk",
        "type": "competition",
        "slug": "home-credit-default-risk",
        "dest": HOME_CREDIT,
        "key_files": ["application_train.csv", "installments_payments.csv"],
    },
    {
        "name": "Lending Club Loan Data",
        "type": "dataset",
        "slug": "wordsforthewise/lending-club",
        "dest": LENDING_CLUB,
        "key_files": ["accepted_2007_to_2018q4.csv"],
    },
    {
        "name": "Indian Loan Default",
        "type": "dataset",
        "slug": "hemanthsai7/loandefault",
        "dest": INDIAN_LOAN,
        "key_files": ["Training Data.csv"],
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def _check_kaggle_token() -> bool:
    if not KAGGLE_JSON.exists():
        print("\n❌  kaggle.json NOT found.")
        print("   Steps to fix:")
        print("   1. Log in at  https://www.kaggle.com")
        print("   2. Go to  Settings → Account → API → 'Create New Token'")
        print("   3. Move the downloaded file to:")
        print(f"      {KAGGLE_JSON}")
        print("   4. Re-run this script.\n")
        return False
    # Validate JSON structure
    try:
        data = json.loads(KAGGLE_JSON.read_text())
        assert "username" in data and "key" in data
    except Exception:
        print(f"❌  {KAGGLE_JSON} is malformed. Re-download from Kaggle settings.\n")
        return False
    print(f"✓  Kaggle token found  ({data['username']})\n")
    return True


def _already_downloaded(dest: Path, key_files: list[str]) -> bool:
    """Check if at least one key file already exists in dest."""
    for f in key_files:
        if (dest / f).exists() or list(dest.glob("**/" + f)):
            return True
    return False


def _extract_zip(zip_path: Path, dest: Path) -> None:
    print(f"   Extracting  {zip_path.name}  →  {dest} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    zip_path.unlink()  # remove zip after extraction


def download_dataset(ds: dict) -> bool:
    import subprocess   # noqa: PLC0415

    dest: Path = ds["dest"]
    dest.mkdir(parents=True, exist_ok=True)

    if _already_downloaded(dest, ds["key_files"]):
        print(f"   ⏭  Already downloaded — skipping {ds['name']}")
        return True

    kaggle_exe = Path(sys.executable).parent / "kaggle"
    if not kaggle_exe.exists():
        kaggle_exe = Path(sys.executable).parent / "kaggle.exe"

    if ds["type"] == "competition":
        cmd = [str(kaggle_exe), "competitions", "download", "-c", ds["slug"],
               "-p", str(dest)]
    else:
        cmd = [str(kaggle_exe), "datasets", "download", "-d", ds["slug"],
               "-p", str(dest), "--unzip"]

    print(f"   ⬇  Downloading {ds['name']} …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"   ❌  Error:\n{result.stderr.strip()}")
        # Surface the most common issue
        if "403" in result.stderr or "401" in result.stderr:
            print("   →  Competition requires manual acceptance of rules.")
            print(f"      Visit: https://www.kaggle.com/c/{ds['slug']}")
            print("      Click 'Join Competition' / accept the rules, then retry.\n")
        return False

    # For competition downloads we get a zip — extract it
    if ds["type"] == "competition":
        for zf in dest.glob("*.zip"):
            _extract_zip(zf, dest)

    print(f"   ✓  {ds['name']} → {dest}")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  PDR Pipeline — Real-World Dataset Downloader")
    print("=" * 60)

    if not _check_kaggle_token():
        sys.exit(1)

    results = {}
    for ds in DATASETS:
        print(f"\n[{ds['name']}]")
        results[ds["name"]] = download_dataset(ds)

    print("\n" + "=" * 60)
    print("  Download Summary")
    print("=" * 60)
    for name, ok in results.items():
        status = "✓ OK" if ok else "✗ FAILED / SKIPPED"
        print(f"  {status}  {name}")

    if all(results.values()):
        print("\n✅  All datasets ready.")
        print("   Next step: python preprocess_real_world_data.py")
    else:
        print("\n⚠  Some datasets failed. Fix errors above and re-run.")
        print("   You can still run preprocessing on whichever succeeded.")


if __name__ == "__main__":
    main()
