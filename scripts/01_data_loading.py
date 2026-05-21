"""
STEP 4: DATA LOADING & INSPECTION
===================================
Business Goal: Load and understand the e-commerce dataset structure.
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Ecommerce_price_Analysis.csv")

def load_data(path: str) -> pd.DataFrame:
    """Load CSV and return DataFrame."""
    print(f"📂 Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"✅ Loaded successfully! Shape: {df.shape}")
    return df

def inspect_data(df: pd.DataFrame) -> None:
    """Print a full inspection report of the dataset."""
    print("\n" + "="*60)
    print("📊 DATASET OVERVIEW")
    print("="*60)

    print(f"\n📐 Shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"💾 Memory Usage : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\n📋 COLUMN NAMES & DATA TYPES:")
    print("-"*40)
    for col, dtype in df.dtypes.items():
        print(f"  {col:<40} {str(dtype):<12}")

    print("\n🔍 FIRST 5 ROWS:")
    print(df.head().to_string())

    print("\n📈 NUMERIC SUMMARY STATISTICS:")
    print(df.describe().round(2).to_string())

    print("\n❓ MISSING VALUES:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct
    }).query("`Missing Count` > 0").sort_values("Missing %", ascending=False)

    if missing_df.empty:
        print("  ✅ No missing values found!")
    else:
        print(missing_df.to_string())

    print("\n🔁 DUPLICATE ROWS:")
    dupes = df.duplicated().sum()
    print(f"  {'⚠️  ' + str(dupes) + ' duplicates found!' if dupes > 0 else '✅ No duplicates found.'}")

    print("\n📌 CATEGORICAL COLUMN VALUE COUNTS:")
    cat_cols = ["order_status", "payment_type", "customer_state", "seller_state"]
    for col in cat_cols:
        if col in df.columns:
            print(f"\n  [{col}]")
            print(df[col].value_counts().head(8).to_string())

    print("\n" + "="*60)
    print("✅ DATA INSPECTION COMPLETE")
    print("="*60)

def main():
    df = load_data(DATA_PATH)
    inspect_data(df)
    return df

if __name__ == "__main__":
    main()