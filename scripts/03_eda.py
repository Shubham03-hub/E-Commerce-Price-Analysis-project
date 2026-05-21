"""
STEP 6: EXPLORATORY DATA ANALYSIS (EDA)
=========================================
- Univariate analysis (distributions)
- Bivariate analysis (relationships)
- Correlation heatmap
- Time-series trends
- Business insights printed to console
All charts saved to outputs/ folder.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os, warnings
warnings.filterwarnings("ignore")

DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_data.csv")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = sns.color_palette("muted", 10)

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"   💾 Saved: {name}")

# ─────────────────────────────────────────────
def load_data():
    print("📂 Loading cleaned data …")
    df = pd.read_csv(DATA_PATH, parse_dates=[
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ])
    print(f"   Shape: {df.shape}")
    return df

# ─────────────────────────────────────────────
# PLOT 1 – Price Distribution
# ─────────────────────────────────────────────
def plot_price_distribution(df):
    print("\n📊 Plot 1: Price Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Price Distribution", fontsize=15, fontweight="bold")

    subset = df["price"].dropna()
    axes[0].hist(subset, bins=60, color=COLORS[0], edgecolor="white")
    axes[0].set_title("Raw Price Distribution")
    axes[0].set_xlabel("Price (R$)")
    axes[0].set_ylabel("Count")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))

    axes[1].hist(np.log1p(subset), bins=60, color=COLORS[1], edgecolor="white")
    axes[1].set_title("Log-transformed Price Distribution")
    axes[1].set_xlabel("log(Price + 1)")
    axes[1].set_ylabel("Count")

    save(fig, "01_price_distribution.png")

# ─────────────────────────────────────────────
# PLOT 2 – Payment Type Breakdown
# ─────────────────────────────────────────────
def plot_payment_types(df):
    print("📊 Plot 2: Payment Type Breakdown")
    counts = df["payment_type"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Payment Type Analysis", fontsize=15, fontweight="bold")

    axes[0].bar(counts.index, counts.values, color=COLORS[:len(counts)])
    axes[0].set_title("Orders by Payment Type")
    axes[0].set_xlabel("Payment Type")
    axes[0].set_ylabel("Order Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 200, f"{v:,}", ha="center", fontsize=9)

    avg_price = df.groupby("payment_type")["payment_value"].mean().sort_values(ascending=False)
    axes[1].barh(avg_price.index, avg_price.values, color=COLORS[:len(avg_price)])
    axes[1].set_title("Avg Payment Value by Type")
    axes[1].set_xlabel("Avg Payment Value (R$)")
    for i, v in enumerate(avg_price.values):
        axes[1].text(v + 2, i, f"R${v:.0f}", va="center", fontsize=9)

    save(fig, "02_payment_types.png")

# ─────────────────────────────────────────────
# PLOT 3 – Review Score Distribution
# ─────────────────────────────────────────────
def plot_review_scores(df):
    print("📊 Plot 3: Review Score Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Customer Review Scores", fontsize=15, fontweight="bold")

    score_counts = df["review_score"].value_counts().sort_index()
    axes[0].bar(score_counts.index.astype(str), score_counts.values,
                color=["#e74c3c","#e67e22","#f1c40f","#2ecc71","#27ae60"])
    axes[0].set_title("Review Score Distribution")
    axes[0].set_xlabel("Review Score (1–5)")
    axes[0].set_ylabel("Count")
    for i, (score, cnt) in enumerate(score_counts.items()):
        axes[0].text(i, cnt + 200, f"{cnt:,}", ha="center", fontsize=9)

    label_counts = df["review_label"].value_counts()
    axes[1].pie(label_counts.values, labels=label_counts.index,
                autopct="%1.1f%%", colors=["#2ecc71","#f1c40f","#e74c3c"],
                startangle=140)
    axes[1].set_title("Sentiment Breakdown")

    save(fig, "03_review_scores.png")

# ─────────────────────────────────────────────
# PLOT 4 – Order Volume Over Time
# ─────────────────────────────────────────────
def plot_orders_over_time(df):
    print("📊 Plot 4: Orders Over Time")
    monthly = df.groupby(df["order_purchase_timestamp"].dt.to_period("M")).size()
    monthly.index = monthly.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly.index, monthly.values, marker="o", color=COLORS[0], linewidth=2, markersize=5)
    ax.fill_between(monthly.index, monthly.values, alpha=0.15, color=COLORS[0])
    ax.set_title("Monthly Order Volume Trend", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Orders")
    ax.xaxis.set_tick_params(rotation=45)
    save(fig, "04_orders_over_time.png")

# ─────────────────────────────────────────────
# PLOT 5 – Orders by Day of Week & Hour
# ─────────────────────────────────────────────
def plot_temporal_patterns(df):
    print("📊 Plot 5: Temporal Patterns")
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_counts = df["purchase_dow"].value_counts().reindex(dow_order)

    hour_counts = df["purchase_hour"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Purchase Temporal Patterns", fontsize=15, fontweight="bold")

    axes[0].bar(dow_counts.index, dow_counts.values, color=COLORS[2])
    axes[0].set_title("Orders by Day of Week")
    axes[0].set_xticklabels(dow_counts.index, rotation=30, ha="right")
    axes[0].set_ylabel("Order Count")

    axes[1].bar(hour_counts.index, hour_counts.values, color=COLORS[3])
    axes[1].set_title("Orders by Hour of Day")
    axes[1].set_xlabel("Hour (0–23)")
    axes[1].set_ylabel("Order Count")

    save(fig, "05_temporal_patterns.png")

# ─────────────────────────────────────────────
# PLOT 6 – Price vs Review Score (Box)
# ─────────────────────────────────────────────
def plot_price_vs_review(df):
    print("📊 Plot 6: Price vs Review Score")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Price & Freight vs Review Score", fontsize=15, fontweight="bold")

    data = df[df["price"] < df["price"].quantile(0.99)]   # remove top 1% outliers
    sns.boxplot(x="review_score", y="price", data=data,
                palette="muted", ax=axes[0])
    axes[0].set_title("Price by Review Score")
    axes[0].set_xlabel("Review Score")
    axes[0].set_ylabel("Price (R$)")

    sns.boxplot(x="review_score", y="freight_value", data=df,
                palette="muted", ax=axes[1])
    axes[1].set_title("Freight Value by Review Score")
    axes[1].set_xlabel("Review Score")
    axes[1].set_ylabel("Freight Value (R$)")

    save(fig, "06_price_vs_review.png")

# ─────────────────────────────────────────────
# PLOT 7 – Delivery Delay Analysis
# ─────────────────────────────────────────────
def plot_delivery_delay(df):
    print("📊 Plot 7: Delivery Delay")
    delay_df = df.dropna(subset=["delivery_delay_days"])
    delay_clip = delay_df["delivery_delay_days"].clip(-30, 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Delivery Delay Analysis", fontsize=15, fontweight="bold")

    axes[0].hist(delay_clip, bins=50, color=COLORS[4], edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", label="On-time threshold")
    axes[0].set_title("Delivery Delay Distribution")
    axes[0].set_xlabel("Delay (days) — negative = early")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    avg_score_delay = delay_df.groupby("is_delayed")["review_score"].mean()
    labels = ["On-Time / Early", "Delayed"]
    axes[1].bar(labels, avg_score_delay.values, color=[COLORS[5], COLORS[6]])
    axes[1].set_title("Avg Review Score: On-time vs Delayed")
    axes[1].set_ylabel("Avg Review Score")
    axes[1].set_ylim(0, 5)
    for i, v in enumerate(avg_score_delay.values):
        axes[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")

    save(fig, "07_delivery_delay.png")

# ─────────────────────────────────────────────
# PLOT 8 – Correlation Heatmap
# ─────────────────────────────────────────────
def plot_correlation(df):
    print("📊 Plot 8: Correlation Heatmap")
    num_cols = ["price", "freight_value", "payment_value", "payment_installments",
                "review_score", "actual_delivery_days", "delivery_delay_days",
                "freight_ratio", "same_state", "is_delayed"]
    corr_df = df[num_cols].dropna().corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm",
                mask=mask, ax=ax, linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    save(fig, "08_correlation_heatmap.png")

# ─────────────────────────────────────────────
# PLOT 9 – Top Customer States
# ─────────────────────────────────────────────
def plot_geo_analysis(df):
    print("📊 Plot 9: Geographic Analysis")
    top_states = df["customer_state"].value_counts().head(10)
    avg_price_state = df.groupby("customer_state")["price"].mean().sort_values(ascending=False).head(10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Geographic Analysis", fontsize=15, fontweight="bold")

    axes[0].barh(top_states.index[::-1], top_states.values[::-1], color=COLORS[0])
    axes[0].set_title("Top 10 States by Order Count")
    axes[0].set_xlabel("Orders")

    axes[1].barh(avg_price_state.index[::-1], avg_price_state.values[::-1], color=COLORS[2])
    axes[1].set_title("Top 10 States by Avg Price")
    axes[1].set_xlabel("Avg Price (R$)")

    save(fig, "09_geo_analysis.png")

# ─────────────────────────────────────────────
# PLOT 10 – Price Band Analysis
# ─────────────────────────────────────────────
def plot_price_bands(df):
    print("📊 Plot 10: Price Band Analysis")
    band_df = df.dropna(subset=["price_band"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Price Band Analysis", fontsize=15, fontweight="bold")

    band_counts = band_df["price_band"].value_counts()
    axes[0].bar(band_counts.index, band_counts.values, color=COLORS[:4])
    axes[0].set_title("Orders by Price Band")
    axes[0].set_ylabel("Count")

    avg_review = band_df.groupby("price_band", observed=True)["review_score"].mean()
    axes[1].bar(avg_review.index, avg_review.values, color=COLORS[4:8])
    axes[1].set_title("Avg Review Score by Price Band")
    axes[1].set_ylabel("Avg Review Score")
    axes[1].set_ylim(0, 5)

    save(fig, "10_price_bands.png")

# ─────────────────────────────────────────────
# PRINT BUSINESS INSIGHTS
# ─────────────────────────────────────────────
def print_insights(df):
    print("\n" + "="*60)
    print("💡 KEY BUSINESS INSIGHTS FROM EDA")
    print("="*60)

    print(f"\n1. 💰 PRICING")
    print(f"   • Avg Price        : R$ {df['price'].mean():.2f}")
    print(f"   • Median Price     : R$ {df['price'].median():.2f}")
    print(f"   • Avg Freight      : R$ {df['freight_value'].mean():.2f}")
    print(f"   • Freight % of val : {df['freight_ratio'].mean()*100:.1f}%")

    print(f"\n2. ⭐ REVIEWS")
    pos = (df["review_label"] == "Positive").sum() / len(df) * 100
    neg = (df["review_label"] == "Negative").sum() / len(df) * 100
    print(f"   • Avg Review Score : {df['review_score'].mean():.2f} / 5")
    print(f"   • Positive (4–5)   : {pos:.1f}%")
    print(f"   • Negative (1–2)   : {neg:.1f}%")

    print(f"\n3. 🚚 DELIVERY")
    delayed = df["is_delayed"].mean() * 100
    avg_delay = df.loc[df["is_delayed"] == 1, "delivery_delay_days"].mean()
    score_ontime = df.loc[df["is_delayed"] == 0, "review_score"].mean()
    score_delayed = df.loc[df["is_delayed"] == 1, "review_score"].mean()
    print(f"   • Delayed Orders   : {delayed:.1f}%")
    print(f"   • Avg Delay Days   : {avg_delay:.1f}")
    print(f"   • Score (on-time)  : {score_ontime:.2f}")
    print(f"   • Score (delayed)  : {score_delayed:.2f}")

    print(f"\n4. 💳 PAYMENTS")
    top_pay = df["payment_type"].value_counts().index[0]
    avg_inst = df["payment_installments"].mean()
    print(f"   • Top Payment Type : {top_pay}")
    print(f"   • Avg Installments : {avg_inst:.1f}")

    print(f"\n5. 📍 GEOGRAPHY")
    top_state = df["customer_state"].value_counts().index[0]
    print(f"   • Top Customer State: {top_state}")
    same = df["same_state"].mean() * 100
    print(f"   • Same-state orders : {same:.1f}%")
    print("="*60)

# ─────────────────────────────────────────────
def main():
    df = load_data()
    plot_price_distribution(df)
    plot_payment_types(df)
    plot_review_scores(df)
    plot_orders_over_time(df)
    plot_temporal_patterns(df)
    plot_price_vs_review(df)
    plot_delivery_delay(df)
    plot_correlation(df)
    plot_geo_analysis(df)
    plot_price_bands(df)
    print_insights(df)
    print(f"\n✅ All 10 EDA charts saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()