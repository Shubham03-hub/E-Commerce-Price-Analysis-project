"""
STEP 8: EVALUATION & INSIGHTS
================================
- Load saved models
- Evaluate on test data
- Plot confusion matrix, ROC curve, feature importances
- Print actionable business recommendations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os, joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve,
                              confusion_matrix, ConfusionMatrixDisplay,
                              mean_absolute_error, r2_score)
from sklearn.impute import SimpleImputer

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_data.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"   💾 {name}")

# ─────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["positive_review"] = (df["review_score"] >= 4).astype(int)
    return df

# ─────────────────────────────────────────────
# CONFUSION MATRIX + ROC
# ─────────────────────────────────────────────
def evaluate_classifier(df):
    print("\n" + "="*55)
    print("🏷️  CLASSIFIER EVALUATION")
    print("="*55)

    feat_cols = [
        "price", "freight_value", "payment_installments",
        "actual_delivery_days", "delivery_delay_days", "is_delayed",
        "same_state", "payment_value", "freight_ratio"
    ]
    target = "positive_review"

    df_c = df.dropna(subset=[target] + feat_cols)
    X = df_c[feat_cols]
    y = df_c[target]
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)

    model_path = os.path.join(MODEL_DIR, "rf_review_classifier.pkl")
    if not os.path.exists(model_path):
        print("  ⚠️  rf_review_classifier.pkl not found. Run 04_modeling.py first.")
        return

    clf = joblib.load(model_path)
    clf.fit(X_tr, y_tr)          # re-fit on same split
    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1]

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_te, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Classifier Evaluation", fontsize=14, fontweight="bold")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Non-Positive", "Positive"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix")

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_te, y_proba)
    auc = roc_auc_score(y_te, y_proba)
    axes[1].plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}", color="#2196F3", lw=2)
    axes[1].plot([0,1],[0,1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()

    save(fig, "11_classifier_roc_cm.png")
    print(f"   AUC-ROC: {auc:.4f}")

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────
def plot_feature_importance():
    print("\n📊 Feature Importance (Classifier)")
    model_path = os.path.join(MODEL_DIR, "rf_review_classifier.pkl")
    if not os.path.exists(model_path):
        print("  ⚠️  Model not found.")
        return

    feat_cols = [
        "price", "freight_value", "payment_installments",
        "actual_delivery_days", "delivery_delay_days", "is_delayed",
        "same_state", "payment_value", "freight_ratio"
    ]

    clf = joblib.load(model_path)
    fi = pd.Series(clf.feature_importances_, index=feat_cols).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    fi.plot.barh(ax=ax, color=sns.color_palette("muted", len(fi)))
    ax.set_title("Feature Importances — Review Score Classifier", fontweight="bold")
    ax.set_xlabel("Importance")
    save(fig, "12_feature_importances.png")

# ─────────────────────────────────────────────
# CLUSTER PROFILES VISUALIZATION
# ─────────────────────────────────────────────
def plot_clusters():
    print("\n📊 Customer Segment Profiles")
    seg_path = os.path.join(MODEL_DIR, "customer_segments.csv")
    if not os.path.exists(seg_path):
        print("  ⚠️  customer_segments.csv not found. Run 04_modeling.py first.")
        return

    seg_df = pd.read_csv(seg_path)
    profile = seg_df.groupby("cluster")[
        ["total_spend","num_orders","avg_price","avg_review","pct_delayed"]
    ].mean().round(2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Customer Segment Profiles", fontsize=14, fontweight="bold")

    profile[["total_spend","avg_price"]].plot.bar(ax=axes[0], colormap="Blues")
    axes[0].set_title("Spend & Price by Segment")
    axes[0].set_xlabel("Cluster")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

    profile[["avg_review","pct_delayed"]].plot.bar(ax=axes[1], colormap="Greens")
    axes[1].set_title("Review Score & Delay Rate by Segment")
    axes[1].set_xlabel("Cluster")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

    save(fig, "13_cluster_profiles.png")

# ─────────────────────────────────────────────
# BUSINESS RECOMMENDATIONS
# ─────────────────────────────────────────────
def print_recommendations(df):
    print("\n" + "="*60)
    print("🚀 ACTIONABLE BUSINESS RECOMMENDATIONS")
    print("="*60)

    # Delivery impact
    delayed_pct = df["is_delayed"].mean() * 100
    score_drop   = (df.loc[df["is_delayed"]==0,"review_score"].mean() -
                    df.loc[df["is_delayed"]==1,"review_score"].mean())

    print(f"""
1. ⏰ DELIVERY IMPROVEMENT (Priority: HIGH)
   • {delayed_pct:.1f}% of orders are delayed → avg score drops {score_drop:.2f} pts
   • ACTION: Negotiate faster logistics contracts in high-delay states
   • ACTION: Set realistic delivery estimates (over-promise = low scores)

2. 💰 PRICING STRATEGY (Priority: MEDIUM)
   • Avg freight is {df['freight_ratio'].mean()*100:.1f}% of total order value
   • Budget-tier orders have lower review scores
   • ACTION: Offer free shipping threshold (e.g., orders > R$150)
   • ACTION: Bundle low-value items to hit free-shipping threshold

3. 💳 PAYMENT OPTIMISATION (Priority: MEDIUM)
   • Credit card users avg {df.loc[df['payment_type']=='credit_card','payment_value'].mean():.0f} R$ per order
   • Avg installments = {df['payment_installments'].mean():.1f} → buyers prefer spreading cost
   • ACTION: Promote Buy-Now-Pay-Later for premium items (>R$200)
   • ACTION: Offer discounts for upfront (boleto) payments

4. 🌍 GEOGRAPHIC FOCUS (Priority: HIGH)
   • Top state (SP) dominates — diversification opportunity
   • Cross-state deliveries take longer and have lower scores
   • ACTION: Expand seller base in NE/N regions to cut delivery time

5. ⭐ REVIEW MANAGEMENT (Priority: HIGH)
   • {(df['review_label']=='Negative').mean()*100:.1f}% of reviews are Negative (1-2 stars)
   • Delivery delay is the #1 predictor of negative reviews
   • ACTION: Auto-trigger apology discount voucher for delayed orders
   • ACTION: Follow-up post-delivery survey for 3-star reviews
""")
    print("="*60)

# ─────────────────────────────────────────────
def main():
    df = load_data()
    evaluate_classifier(df)
    plot_feature_importance()
    plot_clusters()
    print_recommendations(df)
    print(f"\n✅ Evaluation complete. Charts in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()