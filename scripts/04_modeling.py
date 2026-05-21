"""
STEP 7: DATA MODELING
======================
Three ML tasks on this e-commerce dataset:

  Task A – REGRESSION   : Predict order price
  Task B – CLASSIFICATION: Predict if review is Positive (score ≥ 4)
  Task C – CLUSTERING   : Customer segmentation by RFM-like features

Models used:
  - Linear/Logistic Regression (baseline)
  - Random Forest
  - XGBoost
"""

import pandas as pd
import numpy as np
import os, joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (mean_absolute_error, r2_score,
                             classification_report, confusion_matrix,
                             silhouette_score)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost not installed — skipping XGB models (pip install xgboost)")

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_data.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
def load_data():
    print("📂 Loading cleaned data …")
    df = pd.read_csv(DATA_PATH)
    print(f"   Shape: {df.shape}")
    return df

# ─────────────────────────────────────────────
# FEATURE PREPARATION HELPER
# ─────────────────────────────────────────────
NUMERIC_FEATURES = [
    "freight_value", "payment_installments", "payment_value",
    "review_score", "actual_delivery_days", "delivery_delay_days",
    "freight_ratio", "same_state", "purchase_hour", "purchase_month",
    "is_delayed"
]

CATEGORICAL_FEATURES = ["payment_type", "customer_state", "seller_state", "order_status"]

def prepare_features(df, target_col, extra_drop=None):
    """Return X (numpy array), y, feature_names."""
    cols_to_drop = [target_col] + (extra_drop or [])

    # Encode categoricals
    df_enc = df.copy()
    le = LabelEncoder()
    for col in CATEGORICAL_FEATURES:
        if col in df_enc.columns:
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))

    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES
                    if c in df_enc.columns and c not in cols_to_drop]

    X = df_enc[feature_cols].copy()
    y = df_enc[target_col]

    # Impute remaining NaN
    imp = SimpleImputer(strategy="median")
    X = pd.DataFrame(imp.fit_transform(X), columns=feature_cols)

    return X, y, feature_cols

# ─────────────────────────────────────────────
# TASK A – REGRESSION: Predict Price
# ─────────────────────────────────────────────
def run_regression(df):
    print("\n" + "="*55)
    print("📈 TASK A: REGRESSION — Predict Order Price")
    print("="*55)

    feat_cols = [
        "freight_value", "payment_installments", "payment_value",
        "same_state", "purchase_month", "delivery_delay_days",
        "freight_ratio"
    ]
    target = "price"
    df_r = df.dropna(subset=[target] + feat_cols)

    X = df_r[feat_cols]
    y = df_r[target]

    # Log-transform target (skewed price distribution)
    y_log = np.log1p(y)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_log, test_size=0.2, random_state=42)

    results = {}

    # --- Baseline: Linear Regression ---
    lr = Pipeline([("imp", SimpleImputer()), ("scaler", StandardScaler()), ("reg", LinearRegression())])
    lr.fit(X_tr, y_tr)
    y_pred = lr.predict(X_te)
    mae  = mean_absolute_error(np.expm1(y_te), np.expm1(y_pred))
    r2   = r2_score(y_te, y_pred)
    results["LinearRegression"] = {"MAE": round(mae, 2), "R2": round(r2, 4)}
    print(f"\n  Linear Regression   → MAE: R${mae:.2f}  |  R²: {r2:.4f}")

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred_rf = rf.predict(X_te)
    mae_rf = mean_absolute_error(np.expm1(y_te), np.expm1(y_pred_rf))
    r2_rf  = r2_score(y_te, y_pred_rf)
    results["RandomForest"] = {"MAE": round(mae_rf, 2), "R2": round(r2_rf, 4)}
    print(f"  Random Forest       → MAE: R${mae_rf:.2f}  |  R²: {r2_rf:.4f}")

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print(f"\n  Top Feature Importances (RF):\n{importances.to_string()}")

    # --- XGBoost ---
    if HAS_XGB:
        xgb = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                            random_state=42, n_jobs=-1, verbosity=0)
        xgb.fit(X_tr, y_tr)
        y_pred_xgb = xgb.predict(X_te)
        mae_xgb = mean_absolute_error(np.expm1(y_te), np.expm1(y_pred_xgb))
        r2_xgb  = r2_score(y_te, y_pred_xgb)
        results["XGBoost"] = {"MAE": round(mae_xgb, 2), "R2": round(r2_xgb, 4)}
        print(f"  XGBoost             → MAE: R${mae_xgb:.2f}  |  R²: {r2_xgb:.4f}")
        joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_price_regressor.pkl"))
        print("  💾 Saved: xgb_price_regressor.pkl")

    # Save best model (RF)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_price_regressor.pkl"))
    print("  💾 Saved: rf_price_regressor.pkl")

    return results

# ─────────────────────────────────────────────
# TASK B – CLASSIFICATION: Positive Review?
# ─────────────────────────────────────────────
def run_classification(df):
    print("\n" + "="*55)
    print("🏷️  TASK B: CLASSIFICATION — Predict Positive Review")
    print("="*55)

    feat_cols = [
        "price", "freight_value", "payment_installments",
        "actual_delivery_days", "delivery_delay_days", "is_delayed",
        "same_state", "payment_value", "freight_ratio"
    ]
    target = "positive_review"

    df_c = df.dropna(subset=["review_score"] + feat_cols).copy()
    df_c[target] = (df_c["review_score"] >= 4).astype(int)

    X = df_c[feat_cols]
    y = df_c[target]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42,
                                               stratify=y)

    results = {}

    # --- Baseline: Logistic Regression ---
    lr = Pipeline([("imp", SimpleImputer()), ("scaler", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=500, random_state=42))])
    lr.fit(X_tr, y_tr)
    y_pred_lr = lr.predict(X_te)
    print(f"\n  Logistic Regression Classification Report:")
    report = classification_report(y_te, y_pred_lr, target_names=["Non-Positive", "Positive"])
    print(report)
    results["LogisticRegression"] = classification_report(y_te, y_pred_lr, output_dict=True)

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred_rf = rf.predict(X_te)
    print(f"  Random Forest Classification Report:")
    report_rf = classification_report(y_te, y_pred_rf, target_names=["Non-Positive", "Positive"])
    print(report_rf)
    results["RandomForest"] = classification_report(y_te, y_pred_rf, output_dict=True)

    fi = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print(f"  Top Features (RF):\n{fi.to_string()}")

    # --- XGBoost ---
    if HAS_XGB:
        xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                             use_label_encoder=False, eval_metric="logloss",
                             random_state=42, n_jobs=-1, verbosity=0)
        xgb.fit(X_tr, y_tr)
        y_pred_xgb = xgb.predict(X_te)
        print(f"  XGBoost Classification Report:")
        print(classification_report(y_te, y_pred_xgb, target_names=["Non-Positive", "Positive"]))
        results["XGBoost"] = classification_report(y_te, y_pred_xgb, output_dict=True)
        joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_review_classifier.pkl"))
        print("  💾 Saved: xgb_review_classifier.pkl")

    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_review_classifier.pkl"))
    print("  💾 Saved: rf_review_classifier.pkl")

    return results

# ─────────────────────────────────────────────
# TASK C – CLUSTERING: Customer Segmentation
# ─────────────────────────────────────────────
def run_clustering(df):
    print("\n" + "="*55)
    print("🔵 TASK C: CLUSTERING — Customer Segmentation")
    print("="*55)

    # Aggregate per customer
    customer_df = df.groupby("customer_unique_id").agg(
        total_spend    = ("payment_value", "sum"),
        num_orders     = ("order_id",      "count"),
        avg_price      = ("price",         "mean"),
        avg_review     = ("review_score",  "mean"),
        avg_freight    = ("freight_value", "mean"),
        pct_delayed    = ("is_delayed",    "mean"),
    ).dropna()

    print(f"   Customer records: {len(customer_df):,}")

    X = customer_df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method
    inertias = []
    sil_scores = []
    K_range = range(2, 8)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))

    best_k = K_range[np.argmax(sil_scores)]
    print(f"\n   Best K by Silhouette = {best_k}  (score={max(sil_scores):.3f})")

    # Final clustering
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    customer_df["cluster"] = km_final.fit_predict(X_scaled)

    cluster_profile = customer_df.groupby("cluster").mean().round(2)
    cluster_profile["count"] = customer_df["cluster"].value_counts()
    print(f"\n   Cluster Profiles:\n{cluster_profile.to_string()}")

    # Save
    customer_df.to_csv(os.path.join(MODEL_DIR, "customer_segments.csv"), index=True)
    joblib.dump(km_final, os.path.join(MODEL_DIR, "kmeans_customer_segmentation.pkl"))
    print("  💾 Saved: kmeans_customer_segmentation.pkl  &  customer_segments.csv")

    return customer_df, cluster_profile

# ─────────────────────────────────────────────
def main():
    df = load_data()
    reg_results  = run_regression(df)
    clf_results  = run_classification(df)
    seg_df, seg_profile = run_clustering(df)

    print("\n" + "="*55)
    print("✅ ALL MODELS TRAINED & SAVED")
    print("="*55)
    print("\nRegression Summary:")
    for name, m in reg_results.items():
        print(f"  {name:<22} MAE=R${m['MAE']:.2f}  R²={m['R2']:.4f}")

    print("\nClassification Summary (F1 macro):")
    for name, m in clf_results.items():
        print(f"  {name:<22} F1={m['macro avg']['f1-score']:.3f}")

if __name__ == "__main__":
    main()