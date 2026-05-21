"""
STEP 9: STREAMLIT DASHBOARD
=============================
Run with:  streamlit run scripts/06_dashboard.py

Interactive E-Commerce Price Analysis Dashboard with:
  • KPI cards
  • Sales trend
  • Payment analysis
  • Review score explorer
  • Delivery performance
  • Geo heatmap
  • Price predictor (ML model)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, joblib
from sklearn.impute import SimpleImputer

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Price Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH  = os.path.join(BASE_DIR, "data", "cleaned_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_review_classifier.pkl")

# ─────────────────────────────────────────────
# LOAD DATA (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["order_purchase_timestamp"])
    df["positive_review"] = (df["review_score"] >= 4).astype(int)
    return df

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
def sidebar_filters(df):
    st.sidebar.header("🔧 Filters")

    # Date range
    min_date = df["order_purchase_timestamp"].min().date()
    max_date = df["order_purchase_timestamp"].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Order status
    statuses = ["All"] + sorted(df["order_status"].dropna().unique().tolist())
    selected_status = st.sidebar.selectbox("Order Status", statuses)

    # Payment type
    payments = ["All"] + sorted(df["payment_type"].dropna().unique().tolist())
    selected_payment = st.sidebar.selectbox("Payment Type", payments)

    # Price range
    price_min, price_max = float(df["price"].min()), float(df["price"].quantile(0.99))
    price_range = st.sidebar.slider(
        "Price Range (R$)", price_min, price_max,
        (price_min, price_max), step=10.0
    )

    return date_range, selected_status, selected_payment, price_range

def apply_filters(df, date_range, status, payment, price_range):
    mask = pd.Series([True] * len(df))
    if len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        mask &= (df["order_purchase_timestamp"] >= start) & (df["order_purchase_timestamp"] <= end)
    if status != "All":
        mask &= df["order_status"] == status
    if payment != "All":
        mask &= df["payment_type"] == payment
    mask &= (df["price"] >= price_range[0]) & (df["price"] <= price_range[1])
    return df[mask]

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
def kpi_cards(df):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📦 Total Orders",       f"{len(df):,}")
    c2.metric("💰 Avg Price",          f"R$ {df['price'].mean():.2f}")
    c3.metric("⭐ Avg Review Score",   f"{df['review_score'].mean():.2f}")
    c4.metric("🚚 Delay Rate",         f"{df['is_delayed'].mean()*100:.1f}%")
    c5.metric("💳 Avg Payment Value",  f"R$ {df['payment_value'].mean():.2f}")

# ─────────────────────────────────────────────
# TAB 1 — SALES OVERVIEW
# ─────────────────────────────────────────────
def tab_sales(df):
    st.subheader("📈 Monthly Revenue & Order Volume")

    monthly = df.groupby(df["order_purchase_timestamp"].dt.to_period("M")).agg(
        orders=("order_id","count"),
        revenue=("payment_value","sum")
    ).reset_index()
    monthly["order_purchase_timestamp"] = monthly["order_purchase_timestamp"].dt.to_timestamp()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=monthly["order_purchase_timestamp"], y=monthly["orders"],
                         name="Orders", marker_color="#90CAF9"), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly["order_purchase_timestamp"], y=monthly["revenue"],
                              name="Revenue (R$)", line=dict(color="#1565C0", width=2.5)),
                  secondary_y=True)
    fig.update_layout(title="Monthly Orders & Revenue", hovermode="x unified",
                      legend=dict(orientation="h"), height=400)
    fig.update_yaxes(title_text="Order Count", secondary_y=False)
    fig.update_yaxes(title_text="Revenue (R$)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # Orders by day of week
    col1, col2 = st.columns(2)
    with col1:
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = df["purchase_dow"].value_counts().reindex(dow_order).reset_index()
        dow.columns = ["Day", "Orders"]
        fig2 = px.bar(dow, x="Day", y="Orders", title="Orders by Day of Week",
                      color="Orders", color_continuous_scale="Blues")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        hour = df["purchase_hour"].value_counts().sort_index().reset_index()
        hour.columns = ["Hour", "Orders"]
        fig3 = px.area(hour, x="Hour", y="Orders", title="Orders by Hour of Day",
                       color_discrete_sequence=["#1976D2"])
        st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────
# TAB 2 — PRICE & PAYMENT
# ─────────────────────────────────────────────
def tab_pricing(df):
    st.subheader("💰 Price & Payment Analysis")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df[df["price"] < df["price"].quantile(0.99)],
                           x="price", nbins=60, title="Price Distribution",
                           color_discrete_sequence=["#42A5F5"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pay_counts = df["payment_type"].value_counts().reset_index()
        pay_counts.columns = ["Payment Type", "Count"]
        fig2 = px.pie(pay_counts, names="Payment Type", values="Count",
                      title="Payment Type Share", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        avg_pay = df.groupby("payment_type")["payment_value"].mean().reset_index()
        avg_pay.columns = ["Payment Type", "Avg Value"]
        fig3 = px.bar(avg_pay.sort_values("Avg Value", ascending=True),
                      x="Avg Value", y="Payment Type", orientation="h",
                      title="Avg Payment Value by Type", color="Avg Value",
                      color_continuous_scale="Blues")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        inst = df[df["payment_installments"] > 0]["payment_installments"].value_counts().head(10)
        inst = inst.reset_index()
        inst.columns = ["Installments", "Count"]
        fig4 = px.bar(inst.sort_values("Installments"), x="Installments", y="Count",
                      title="Installment Distribution", color_discrete_sequence=["#66BB6A"])
        st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────
# TAB 3 — REVIEWS & DELIVERY
# ─────────────────────────────────────────────
def tab_reviews(df):
    st.subheader("⭐ Reviews & Delivery Performance")

    col1, col2 = st.columns(2)
    with col1:
        score_counts = df["review_score"].value_counts().sort_index().reset_index()
        score_counts.columns = ["Score", "Count"]
        colors = ["#E53935","#FB8C00","#FDD835","#43A047","#1E88E5"]
        fig = px.bar(score_counts, x="Score", y="Count", title="Review Score Distribution",
                     color="Score", color_discrete_sequence=colors)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        delay_df = df.dropna(subset=["delivery_delay_days"])
        delay_clip = delay_df["delivery_delay_days"].clip(-30, 60)
        fig2 = px.histogram(delay_clip, nbins=50, title="Delivery Delay Distribution",
                             color_discrete_sequence=["#EF5350"],
                             labels={"value": "Days (negative=early)"})
        fig2.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="On-time")
        st.plotly_chart(fig2, use_container_width=True)

    # Score vs delay
    fig3 = px.box(df.dropna(subset=["review_score"]),
                  x=df["is_delayed"].map({0:"On-Time", 1:"Delayed"}),
                  y="review_score",
                  title="Review Score: On-Time vs Delayed Orders",
                  color=df["is_delayed"].map({0:"On-Time", 1:"Delayed"}),
                  color_discrete_map={"On-Time":"#43A047","Delayed":"#E53935"})
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────
# TAB 4 — GEOGRAPHY
# ─────────────────────────────────────────────
def tab_geo(df):
    st.subheader("🗺️ Geographic Analysis")

    col1, col2 = st.columns(2)
    with col1:
        state_orders = df["customer_state"].value_counts().reset_index()
        state_orders.columns = ["State", "Orders"]
        fig = px.bar(state_orders.head(15).sort_values("Orders"),
                     x="Orders", y="State", orientation="h",
                     title="Top 15 States by Order Count",
                     color="Orders", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        state_avg = df.groupby("customer_state")["price"].mean().reset_index()
        state_avg.columns = ["State", "Avg Price"]
        fig2 = px.bar(state_avg.sort_values("Avg Price", ascending=False).head(15),
                      x="State", y="Avg Price", title="Avg Price by State",
                      color="Avg Price", color_continuous_scale="Oranges")
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter: delivery days vs review
    sample = df.dropna(subset=["actual_delivery_days","review_score"]).sample(
        min(3000, len(df)), random_state=42)
    fig3 = px.scatter(sample, x="actual_delivery_days", y="review_score",
                      color="is_delayed", opacity=0.4,
                      title="Delivery Days vs Review Score",
                      color_continuous_scale="RdYlGn_r",
                      labels={"actual_delivery_days":"Delivery Days",
                               "review_score":"Review Score",
                               "is_delayed":"Delayed"})
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────
# TAB 5 — ML PREDICTOR
# ─────────────────────────────────────────────
def tab_predictor(df):
    st.subheader("🤖 Review Score Predictor")
    st.write("Fill in order details below to predict if the customer will leave a **Positive review (≥4 stars)**.")

    model = load_model()
    if model is None:
        st.warning("⚠️ Model not found. Run `python scripts/04_modeling.py` first.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        price        = st.number_input("Price (R$)", 5.0, 5000.0, 100.0, step=5.0)
        freight      = st.number_input("Freight Value (R$)", 0.0, 200.0, 15.0, step=1.0)
        payment_val  = st.number_input("Payment Value (R$)", 5.0, 5000.0, 115.0, step=5.0)
    with col2:
        installments = st.number_input("Installments", 1, 24, 1)
        delivery_days= st.number_input("Actual Delivery Days", 1, 60, 10)
        delay_days   = st.number_input("Delay Days (0=on-time)", -30, 60, 0)
    with col3:
        is_delayed   = st.selectbox("Is Delayed?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        same_state   = st.selectbox("Same State?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        freight_ratio= round(freight / (price + freight + 0.001), 4)
        st.metric("Freight Ratio", f"{freight_ratio:.2%}")

    if st.button("🔮 Predict Review", use_container_width=True):
        X_input = np.array([[price, freight, installments, delivery_days,
                              delay_days, is_delayed, same_state, payment_val, freight_ratio]])
        imp = SimpleImputer(strategy="median")
        # Use training data to fit imputer
        feat_cols = ["price","freight_value","payment_installments",
                     "actual_delivery_days","delivery_delay_days","is_delayed",
                     "same_state","payment_value","freight_ratio"]
        df_ref = df.dropna(subset=feat_cols)[feat_cols]
        imp.fit(df_ref.values)
        X_clean = imp.transform(X_input)

        prob = model.predict_proba(X_clean)[0][1]
        pred = model.predict(X_clean)[0]

        st.markdown("---")
        if pred == 1:
            st.success(f"✅ **Positive Review Likely** — Confidence: {prob*100:.1f}%")
        else:
            st.error(f"❌ **Negative/Neutral Review Likely** — Confidence: {(1-prob)*100:.1f}%")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob*100, 1),
            title={"text": "Positive Review Probability (%)"},
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color": "#4CAF50" if prob >= 0.5 else "#F44336"},
                   "steps":[{"range":[0,40],"color":"#FFEBEE"},
                             {"range":[40,70],"color":"#FFF9C4"},
                             {"range":[70,100],"color":"#E8F5E9"}]}
        ))
        gauge.update_layout(height=300)
        st.plotly_chart(gauge, use_container_width=True)

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    st.title("🛒 E-Commerce Price Analysis Dashboard")
    st.caption("Brazilian E-Commerce Dataset | 119,000+ Orders")

    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Data file not found at: {DATA_PATH}\nRun `python scripts/02_data_cleaning.py` first.")
        st.stop()

    df = load_data()

    # Sidebar filters
    date_range, status, payment, price_range = sidebar_filters(df)
    df_f = apply_filters(df, date_range, status, payment, price_range)

    st.info(f"Showing **{len(df_f):,}** orders after filters (original: {len(df):,})")

    # KPI row
    kpi_cards(df_f)

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Sales Overview",
        "💰 Price & Payment",
        "⭐ Reviews & Delivery",
        "🗺️ Geography",
        "🤖 ML Predictor"
    ])

    with tab1: tab_sales(df_f)
    with tab2: tab_pricing(df_f)
    with tab3: tab_reviews(df_f)
    with tab4: tab_geo(df_f)
    with tab5: tab_predictor(df)        # use full df for model

if __name__ == "__main__":
    main()