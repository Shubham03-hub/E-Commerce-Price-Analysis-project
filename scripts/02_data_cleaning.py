"""
STEP 5: DATA CLEANING & PREPROCESSING
=======================================
- Handle missing values
- Fix data types (datetime parsing)
- Remove duplicates
- Feature engineering (delivery days, delay, price bands, etc.)
- Save cleaned dataset
"""

import pandas as pd
import numpy as np
import os

DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "Ecommerce_price_Analysis.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_data.csv")

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
def load_raw(path: str) -> pd.DataFrame:
    print("📂 Loading raw data …")
    df = pd.read_csv(path)
    print(f"   Raw shape: {df.shape}")
    return df

# ─────────────────────────────────────────────
# 2. FIX DATA TYPES
# ─────────────────────────────────────────────
DATETIME_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "shipping_limit_date",
    "review_creation_date",
    "review_answer_timestamp",
]

def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🔧 Fixing data types …")
    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            print(f"   ✅ Converted '{col}' → datetime")

    # Ensure numeric cols are correct
    for col in ["price", "freight_value", "payment_value", "review_score", "payment_installments"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ─────────────────────────────────────────────
# 3. HANDLE MISSING VALUES
# ─────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🧹 Handling missing values …")
    before = df.shape[0]

    # Drop rows where key business fields are missing
    df = df.dropna(subset=["price", "freight_value", "payment_value", "payment_type"])
    print(f"   Dropped {before - df.shape[0]} rows with missing key fields")

    # Fill review_score median
    median_score = df["review_score"].median()
    df["review_score"] = df["review_score"].fillna(median_score)
    print(f"   Filled review_score NaN → median ({median_score})")

    # Fill comment text with empty string
    df["review_comment_title"]   = df["review_comment_title"].fillna("")
    df["review_comment_message"] = df["review_comment_message"].fillna("")

    # Fill delivery dates — rows without delivery date = not yet delivered
    # Keep as NaT; feature engineering will handle them

    print(f"   Remaining NaNs:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")
    return df

# ─────────────────────────────────────────────
# 4. REMOVE DUPLICATES
# ─────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🔁 Removing duplicates …")
    before = df.shape[0]
    df = df.drop_duplicates()
    print(f"   Removed {before - df.shape[0]} duplicate rows")
    return df

# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("\n⚙️  Feature Engineering …")

    # --- Time-based features ---
    df["purchase_hour"]   = df["order_purchase_timestamp"].dt.hour
    df["purchase_dow"]    = df["order_purchase_timestamp"].dt.day_name()
    df["purchase_month"]  = df["order_purchase_timestamp"].dt.month
    df["purchase_year"]   = df["order_purchase_timestamp"].dt.year

    # --- Delivery metrics ---
    df["actual_delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days

    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days

    df["delivery_delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days

    df["is_delayed"] = (df["delivery_delay_days"] > 0).astype(int)

    print("   ✅ purchase_hour, purchase_dow, purchase_month, purchase_year")
    print("   ✅ actual_delivery_days, estimated_delivery_days, delivery_delay_days, is_delayed")

    # --- Price features ---
    df["total_order_value"] = df["price"] + df["freight_value"]
    df["freight_ratio"]     = (df["freight_value"] / df["total_order_value"]).round(4)

    # Price band (quantile-based)
    df["price_band"] = pd.qcut(
        df["price"],
        q=4,
        labels=["Budget", "Economy", "Mid-range", "Premium"],
        duplicates="drop"
    )
    print("   ✅ total_order_value, freight_ratio, price_band")

    # --- Review label ---
    df["review_label"] = df["review_score"].apply(
        lambda x: "Positive" if x >= 4 else ("Neutral" if x == 3 else "Negative")
    )
    print("   ✅ review_label")

    # --- Same-state delivery ---
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
    print("   ✅ same_state")

    return df

# ─────────────────────────────────────────────
# 6. SAVE CLEANED DATA
# ─────────────────────────────────────────────
def save_clean(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"\n💾 Cleaned data saved → {path}")
    print(f"   Final shape: {df.shape}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    df = load_raw(DATA_PATH)
    df = fix_types(df)
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = feature_engineering(df)
    save_clean(df, OUTPUT_PATH)

    print("\n" + "="*50)
    print("✅ DATA CLEANING COMPLETE")
    print("="*50)
    print(f"\nNew engineered columns: {[c for c in df.columns if c not in pd.read_csv(DATA_PATH, nrows=1).columns]}")
    return df

if __name__ == "__main__":
    main()