"""
Toronto Island Ferry Ticket Demand Forecasting & Predictive Decision Support System
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import os
warnings.filterwarnings("ignore")

# Always resolve CSV paths relative to the script's own folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🛥️ Ferry Ticket Demand Forecasting",
    page_icon="🛥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card h3 { color: #e94560; font-size: 0.85rem; margin: 0 0 0.3rem 0; }
    .metric-card h1 { color: #ffffff; font-size: 1.8rem; margin: 0; }
    .metric-card p  { color: #a0aec0; font-size: 0.75rem; margin: 0.3rem 0 0 0; }
    .section-header {
        background: linear-gradient(90deg, #e94560, #0f3460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load & cache data
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data(path="Toronto_Island_Ferry_Tickets.csv"):
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Feature engineering
    df["Hour"]        = df["Timestamp"].dt.hour
    df["DayOfWeek"]   = df["Timestamp"].dt.dayofweek
    df["Month"]       = df["Timestamp"].dt.month
    df["Year"]        = df["Timestamp"].dt.year
    df["IsWeekend"]   = df["DayOfWeek"].isin([5, 6]).astype(int)
    df["DayOfYear"]   = df["Timestamp"].dt.dayofyear

    for col in ["Sales Count", "Redemption Count"]:
        for lag in [1, 2, 4, 8]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        for w in [4, 8]:
            df[f"{col}_roll_mean{w}"] = df[col].shift(1).rolling(w).mean()
            df[f"{col}_roll_std{w}"]  = df[col].shift(1).rolling(w).std()
        df[f"{col}_roll_max4"]  = df[col].shift(1).rolling(4).max()

    return df.dropna().reset_index(drop=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Toronto_Island_Ferry_Ongiara.jpg/320px-Toronto_Island_Ferry_Ongiara.jpg",
                 width=260)

st.sidebar.title("🛥️ Ferry Forecasting")
st.sidebar.markdown("**Predictive Decision Support System**")
st.sidebar.divider()

CSV_PATH = r"C:\Users\ACER\Downloads\Ferry Ticket Demand Forecasting\Toronto_Island_Ferry_Tickets.csv"

# Try loading
if not os.path.exists(CSV_PATH):
    st.sidebar.error("❌ CSV not found!")
    st.error(f"""
### ❌ File Not Found

The app looked for the CSV at:
```
{CSV_PATH}
```
**Make sure `Toronto_Island_Ferry_Tickets.csv` is in the same folder as `ferry_forecast_app.py`.**

Your folder should look like:
```
Ferry Ticket Demand Forecasting\\
├── ferry_forecast_app.py
└── Toronto_Island_Ferry_Tickets.csv   ← must be here
```
If it's somewhere else, paste the full path into the **CSV file path** box in the sidebar.
""")
    st.stop()

try:
    df = load_data(CSV_PATH)
except Exception as e:
    st.sidebar.error("❌ Load failed!")
    st.error(f"❌ Could not load CSV: {e}")
    st.stop()

st.sidebar.success(f"✅ {len(df):,} rows loaded")
st.sidebar.divider()

TARGET = st.sidebar.selectbox("🎯 Target Variable", ["Sales Count", "Redemption Count"])
HORIZON_LABEL = st.sidebar.selectbox("⏱️ Forecast Horizon", ["15 min", "30 min", "1 hour", "2 hours"])
HORIZON_MAP   = {"15 min": 1, "30 min": 2, "1 hour": 4, "2 hours": 8}
HORIZON_STEPS = HORIZON_MAP[HORIZON_LABEL]

MODEL_NAME = st.sidebar.selectbox(
    "🤖 Forecasting Model",
    ["Moving Average (Baseline)", "Linear Regression", "Random Forest", "Gradient Boosting", "ARIMA"]
)

st.sidebar.divider()
st.sidebar.markdown("**Filter Date Range**")
min_date = df["Timestamp"].min().date()
max_date = df["Timestamp"].max().date()
date_range = st.sidebar.date_input(
    "Select range",
    value=(max_date - pd.Timedelta(days=90), max_date),
    min_value=min_date,
    max_value=max_date,
)
if len(date_range) == 2:
    start_dt, end_dt = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_dt = end_dt = pd.Timestamp(date_range[0])

df_view = df[(df["Timestamp"] >= start_dt) & (df["Timestamp"] <= end_dt)].copy()

st.sidebar.divider()
st.sidebar.caption("📊 Toronto Parks, Forestry & Recreation\nUnified Mentor Project")


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("🛥️ Toronto Island Ferry — Demand Forecasting")
st.markdown("**Short-Term Predictive Decision Support System** | 15-min interval data from 2015–2025")
st.divider()

# ─────────────────────────────────────────────
# KPI Row
# ─────────────────────────────────────────────
avg_sales   = df_view["Sales Count"].mean()
avg_redeem  = df_view["Redemption Count"].mean()
peak_sales  = df_view["Sales Count"].max()
peak_redeem = df_view["Redemption Count"].max()
total_sales = df_view["Sales Count"].sum()

c1, c2, c3, c4, c5 = st.columns(5)
for col, label, val, sub in zip(
    [c1, c2, c3, c4, c5],
    ["Avg Sales / 15 min", "Avg Redemptions / 15 min", "Peak Sales", "Peak Redemptions", "Total Sales (Range)"],
    [f"{avg_sales:.1f}", f"{avg_redeem:.1f}", f"{peak_sales:,}", f"{peak_redeem:,}", f"{total_sales:,}"],
    ["tickets", "tickets", "single interval", "single interval", "selected period"],
):
    col.markdown(f"""
    <div class="metric-card">
        <h3>{label}</h3>
        <h1>{val}</h1>
        <p>{sub}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 EDA & Trends",
    "🔮 Forecast",
    "📊 Model Comparison",
    "📉 Feature Importance",
    "📋 Data Table"
])

# ─── TAB 1: EDA ───────────────────────────────
with tab1:
    st.subheader("📈 Exploratory Data Analysis")

    # Time series plot
    fig, ax = plt.subplots(figsize=(14, 4))
    ts_daily = df_view.set_index("Timestamp")[["Sales Count", "Redemption Count"]].resample("D").sum()
    ax.plot(ts_daily.index, ts_daily["Sales Count"],      color="#e94560", linewidth=1.2, label="Sales")
    ax.plot(ts_daily.index, ts_daily["Redemption Count"], color="#0f3460", linewidth=1.2, label="Redemptions", alpha=0.85)
    ax.set_title(f"Daily Total — {start_dt.date()} to {end_dt.date()}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Tickets")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    ax.grid(alpha=0.2)
    fig.patch.set_alpha(0)
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)

    with col1:
        # Hourly pattern
        fig, ax = plt.subplots(figsize=(7, 3.5))
        hourly = df_view.groupby("Hour")[TARGET].mean()
        bars = ax.bar(hourly.index, hourly.values, color="#e94560", alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_title(f"Avg {TARGET} by Hour of Day", fontweight="bold")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Avg Tickets")
        ax.grid(axis="y", alpha=0.3)
        fig.patch.set_alpha(0)
        st.pyplot(fig)
        plt.close()

    with col2:
        # Day of week pattern
        fig, ax = plt.subplots(figsize=(7, 3.5))
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow = df_view.groupby("DayOfWeek")[TARGET].mean()
        colors = ["#e94560" if i in [5, 6] else "#0f3460" for i in range(7)]
        ax.bar(days, dow.values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(f"Avg {TARGET} by Day of Week", fontweight="bold")
        ax.set_ylabel("Avg Tickets")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(handles=[
            plt.Rectangle((0,0),1,1, color="#e94560"), plt.Rectangle((0,0),1,1, color="#0f3460")
        ], labels=["Weekend", "Weekday"], fontsize=8)
        fig.patch.set_alpha(0)
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        # Monthly pattern
        fig, ax = plt.subplots(figsize=(7, 3.5))
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly = df_view.groupby("Month")[TARGET].mean()
        ax.plot(monthly.index, monthly.values, marker="o", color="#e94560", linewidth=2)
        ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="#e94560")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(months, rotation=45)
        ax.set_title(f"Monthly Seasonality — {TARGET}", fontweight="bold")
        ax.set_ylabel("Avg Tickets")
        ax.grid(alpha=0.3)
        fig.patch.set_alpha(0)
        st.pyplot(fig)
        plt.close()

    with col4:
        # Distribution
        fig, ax = plt.subplots(figsize=(7, 3.5))
        data_clip = df_view[TARGET].clip(upper=df_view[TARGET].quantile(0.99))
        ax.hist(data_clip, bins=60, color="#0f3460", edgecolor="white", linewidth=0.4, alpha=0.9)
        ax.axvline(data_clip.mean(),   color="#e94560", linestyle="--", linewidth=1.5, label=f"Mean: {data_clip.mean():.1f}")
        ax.axvline(data_clip.median(), color="#f7b731", linestyle="--", linewidth=1.5, label=f"Median: {data_clip.median():.1f}")
        ax.set_title(f"Distribution of {TARGET}", fontweight="bold")
        ax.set_xlabel("Tickets")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.patch.set_alpha(0)
        st.pyplot(fig)
        plt.close()

    # Weekend vs weekday box
    fig, ax = plt.subplots(figsize=(14, 3))
    wday  = df_view[df_view["IsWeekend"] == 0][TARGET].clip(upper=df_view[TARGET].quantile(0.99))
    wend  = df_view[df_view["IsWeekend"] == 1][TARGET].clip(upper=df_view[TARGET].quantile(0.99))
    ax.boxplot([wday, wend], labels=["Weekday", "Weekend"], patch_artist=True,
               boxprops=dict(facecolor="#0f3460", color="white"),
               medianprops=dict(color="#e94560", linewidth=2),
               whiskerprops=dict(color="white"), capprops=dict(color="white"),
               flierprops=dict(marker=".", color="gray", alpha=0.3))
    ax.set_title(f"Weekday vs Weekend — {TARGET}", fontweight="bold")
    ax.set_ylabel("Tickets")
    ax.grid(axis="y", alpha=0.3)
    fig.patch.set_alpha(0)
    st.pyplot(fig)
    plt.close()


# ─── TAB 2: FORECAST ──────────────────────────
with tab2:
    st.subheader(f"🔮 {MODEL_NAME} — {HORIZON_LABEL} Forecast on {TARGET}")

    # Build features
    FEAT_COLS = [
        "Hour", "DayOfWeek", "Month", "IsWeekend",
        f"{TARGET}_lag1", f"{TARGET}_lag2", f"{TARGET}_lag4", f"{TARGET}_lag8",
        f"{TARGET}_roll_mean4", f"{TARGET}_roll_mean8",
        f"{TARGET}_roll_std4",  f"{TARGET}_roll_max4",
    ]

    df_model = df[["Timestamp", TARGET] + FEAT_COLS].dropna().copy()

    # Create horizon target
    df_model["target"] = df_model[TARGET].shift(-HORIZON_STEPS)
    df_model = df_model.dropna()

    split_idx = int(len(df_model) * 0.85)
    train     = df_model.iloc[:split_idx]
    test      = df_model.iloc[split_idx:]

    X_train, y_train = train[FEAT_COLS], train["target"]
    X_test,  y_test  = test[FEAT_COLS],  test["target"]

    # ── Train selected model ──
    with st.spinner(f"Training {MODEL_NAME}…"):
        if MODEL_NAME == "Moving Average (Baseline)":
            window = max(HORIZON_STEPS, 4)
            preds  = df_model[TARGET].shift(1).rolling(window).mean().iloc[split_idx:]
            preds  = preds.fillna(method="bfill").values

        elif MODEL_NAME == "Linear Regression":
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_te_s = scaler.transform(X_test)
            model  = LinearRegression()
            model.fit(X_tr_s, y_train)
            preds  = model.predict(X_te_s).clip(0)

        elif MODEL_NAME == "Random Forest":
            from sklearn.ensemble import RandomForestRegressor
            model  = RandomForestRegressor(n_estimators=150, max_depth=12, n_jobs=-1, random_state=42)
            model.fit(X_train, y_train)
            preds  = model.predict(X_test).clip(0)

        elif MODEL_NAME == "Gradient Boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            model  = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
            model.fit(X_train, y_train)
            preds  = model.predict(X_test).clip(0)

        elif MODEL_NAME == "ARIMA":
            try:
                from statsmodels.tsa.arima.model import ARIMA
                # Use last 2000 rows of training for speed
                train_series = train[TARGET].values[-2000:]
                arima_model  = ARIMA(train_series, order=(2, 1, 2)).fit()
                # Forecast on test (rolling approach approximated by walk-forward last 500)
                n_test = min(len(test), 500)
                preds  = arima_model.forecast(steps=n_test)
                preds  = np.clip(preds, 0, None)
                y_test = y_test.values[:n_test]
                test   = test.iloc[:n_test]
            except Exception as e:
                st.warning(f"ARIMA failed ({e}). Falling back to Moving Average.")
                window = max(HORIZON_STEPS, 4)
                preds  = df_model[TARGET].shift(1).rolling(window).mean().iloc[split_idx:]
                preds  = preds.fillna(method="bfill").values

    y_actual = np.array(y_test) if hasattr(y_test, "values") else y_test
    preds    = np.array(preds)

    # Align lengths
    min_len  = min(len(y_actual), len(preds))
    y_actual = y_actual[:min_len]
    preds    = preds[:min_len]
    ts_plot  = test["Timestamp"].values[:min_len]

    # Metrics
    mae  = np.mean(np.abs(y_actual - preds))
    rmse = np.sqrt(np.mean((y_actual - preds) ** 2))
    mask = y_actual != 0
    mape = np.mean(np.abs((y_actual[mask] - preds[mask]) / y_actual[mask])) * 100 if mask.sum() > 0 else np.nan

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"{mae:.2f}",  "Mean Absolute Error")
    m2.metric("RMSE", f"{rmse:.2f}", "Root Mean Squared Error")
    m3.metric("MAPE", f"{mape:.1f}%", "Mean Abs Percentage Error")

    # Confidence bands (residual std)
    res_std = np.std(y_actual - preds)

    # Plot last N points
    N_plot = st.slider("Points to display", 100, min(2000, min_len), 500, step=50)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ts_plot[-N_plot:], y_actual[-N_plot:], color="#0f3460", linewidth=1.2, label="Actual", alpha=0.9)
    ax.plot(ts_plot[-N_plot:], preds[-N_plot:],    color="#e94560", linewidth=1.2, label="Forecast", linestyle="--")
    ax.fill_between(ts_plot[-N_plot:],
                    (preds[-N_plot:] - 1.96 * res_std).clip(0),
                     preds[-N_plot:] + 1.96 * res_std,
                    alpha=0.15, color="#e94560", label="95% CI")
    ax.set_title(f"{MODEL_NAME} — {HORIZON_LABEL} ahead forecast | {TARGET}", fontweight="bold", fontsize=13)
    ax.set_ylabel("Tickets")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    fig.autofmt_xdate()
    fig.patch.set_alpha(0)
    st.pyplot(fig)
    plt.close()

    # Error distribution
    errors = y_actual - preds
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(errors, bins=60, color="#0f3460", edgecolor="white", linewidth=0.4, alpha=0.9)
    ax.axvline(0,              color="#e94560", linestyle="--", linewidth=1.5, label="Zero Error")
    ax.axvline(errors.mean(),  color="#f7b731", linestyle="--", linewidth=1.5, label=f"Mean Error: {errors.mean():.2f}")
    ax.set_title("Forecast Error Distribution", fontweight="bold")
    ax.set_xlabel("Error (Actual − Predicted)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.patch.set_alpha(0)
    st.pyplot(fig)
    plt.close()


# ─── TAB 3: MODEL COMPARISON ──────────────────
with tab3:
    st.subheader("📊 Model Comparison Across Horizons")
    st.info("Trains all models on the same train-test split and compares MAE & RMSE.", icon="ℹ️")

    @st.cache_data(show_spinner="Running model comparison…")
    def run_comparison(csv_path, target):
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler

        df_c = load_data(csv_path)
        feat_cols = [
            "Hour", "DayOfWeek", "Month", "IsWeekend",
            f"{target}_lag1", f"{target}_lag2", f"{target}_lag4", f"{target}_lag8",
            f"{target}_roll_mean4", f"{target}_roll_mean8",
            f"{target}_roll_std4",  f"{target}_roll_max4",
        ]
        results = []
        horizons = {"15 min": 1, "30 min": 2, "1 hour": 4, "2 hours": 8}

        for hlabel, hsteps in horizons.items():
            dm = df_c[["Timestamp", target] + feat_cols].dropna().copy()
            dm["target"] = dm[target].shift(-hsteps)
            dm = dm.dropna()
            split = int(len(dm) * 0.85)
            X_tr, y_tr = dm.iloc[:split][feat_cols], dm.iloc[:split]["target"]
            X_te, y_te = dm.iloc[split:][feat_cols], dm.iloc[split:]["target"]

            models = {
                "Moving Avg":        None,
                "Linear Regression": LinearRegression(),
                "Random Forest":     RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
            }

            for mname, model in models.items():
                if model is None:
                    w = max(hsteps, 4)
                    p = dm[target].shift(1).rolling(w).mean().iloc[split:].fillna(method="bfill").values
                else:
                    if mname == "Linear Regression":
                        sc = StandardScaler()
                        model.fit(sc.fit_transform(X_tr), y_tr)
                        p = model.predict(sc.transform(X_te)).clip(0)
                    else:
                        model.fit(X_tr, y_tr)
                        p = model.predict(X_te).clip(0)

                y = y_te.values
                ml = min(len(y), len(p))
                y, p = y[:ml], p[:ml]
                mae_v  = np.mean(np.abs(y - p))
                rmse_v = np.sqrt(np.mean((y - p) ** 2))
                results.append({"Model": mname, "Horizon": hlabel, "MAE": round(mae_v,2), "RMSE": round(rmse_v,2)})

        return pd.DataFrame(results)

    cmp_df = run_comparison(CSV_PATH, TARGET)

    # Pivot for display
    pivot_mae  = cmp_df.pivot(index="Model", columns="Horizon", values="MAE")[["15 min","30 min","1 hour","2 hours"]]
    pivot_rmse = cmp_df.pivot(index="Model", columns="Horizon", values="RMSE")[["15 min","30 min","1 hour","2 hours"]]

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**MAE by Model & Horizon**")
        st.dataframe(pivot_mae.style.background_gradient(cmap="RdYlGn_r", axis=None).format("{:.2f}"),
                     use_container_width=True)

    with col_b:
        st.markdown("**RMSE by Model & Horizon**")
        st.dataframe(pivot_rmse.style.background_gradient(cmap="RdYlGn_r", axis=None).format("{:.2f}"),
                     use_container_width=True)

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    horizon_order = ["15 min", "30 min", "1 hour", "2 hours"]
    model_list    = cmp_df["Model"].unique()
    x = np.arange(len(horizon_order))
    width = 0.2
    colors = ["#e94560", "#0f3460", "#f7b731", "#2ecc71"]

    for i, (mname, color) in enumerate(zip(model_list, colors)):
        subset = cmp_df[cmp_df["Model"] == mname].set_index("Horizon").reindex(horizon_order)
        axes[0].bar(x + i * width, subset["MAE"].values,  width, label=mname, color=color, alpha=0.85)
        axes[1].bar(x + i * width, subset["RMSE"].values, width, label=mname, color=color, alpha=0.85)

    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        ax.set_title(f"{metric} Comparison", fontweight="bold")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(horizon_order)
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Model Performance — {TARGET}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    st.pyplot(fig)
    plt.close()


# ─── TAB 4: FEATURE IMPORTANCE ────────────────
with tab4:
    st.subheader("📉 Feature Importance & Lag Analysis")

    @st.cache_data(show_spinner="Computing feature importance…")
    def get_feature_importance(csv_path, target, h_steps):
        from sklearn.ensemble import RandomForestRegressor
        df_fi = load_data(csv_path)
        feat_cols = [
            "Hour", "DayOfWeek", "Month", "IsWeekend",
            f"{target}_lag1", f"{target}_lag2", f"{target}_lag4", f"{target}_lag8",
            f"{target}_roll_mean4", f"{target}_roll_mean8",
            f"{target}_roll_std4",  f"{target}_roll_max4",
        ]
        dm = df_fi[["Timestamp", target] + feat_cols].dropna().copy()
        dm["target"] = dm[target].shift(-h_steps)
        dm = dm.dropna()
        split = int(len(dm) * 0.85)
        X_tr, y_tr = dm.iloc[:split][feat_cols], dm.iloc[:split]["target"]
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
        rf.fit(X_tr, y_tr)
        return pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=True)

    fi = get_feature_importance(CSV_PATH, TARGET, HORIZON_STEPS)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_fi = ["#e94560" if v == fi.max() else "#0f3460" for v in fi.values]
    ax.barh(fi.index, fi.values, color=colors_fi, edgecolor="white", linewidth=0.4)
    ax.set_title(f"Random Forest Feature Importance\n{TARGET} | {HORIZON_LABEL} horizon", fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(fi.values):
        ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.patch.set_alpha(0)
    st.pyplot(fig)
    plt.close()

    # Correlation heatmap
    st.markdown("**Lag Correlation with Target**")
    lag_cols = [c for c in df.columns if "lag" in c and TARGET in c]
    if lag_cols:
        corrs = df[lag_cols + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
        fig, ax = plt.subplots(figsize=(8, 3))
        bar_c = ["#e94560" if v > 0 else "#0f3460" for v in corrs.values]
        ax.bar(corrs.index, corrs.values, color=bar_c, edgecolor="white", linewidth=0.4)
        ax.axhline(0, color="white", linewidth=0.8)
        ax.set_title(f"Lag Correlations with {TARGET}", fontweight="bold")
        ax.set_ylabel("Pearson Correlation")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.patch.set_alpha(0)
        st.pyplot(fig)
        plt.close()


# ─── TAB 5: DATA TABLE ────────────────────────
with tab5:
    st.subheader("📋 Raw Data Preview")

    st.markdown(f"Showing data from **{start_dt.date()}** to **{end_dt.date()}** — {len(df_view):,} rows")

    cols_show = st.multiselect(
        "Select columns",
        df_view.columns.tolist(),
        default=["Timestamp", "Sales Count", "Redemption Count", "Hour", "DayOfWeek", "IsWeekend"]
    )

    n_rows = st.slider("Rows to display", 50, 1000, 200, step=50)
    st.dataframe(df_view[cols_show].tail(n_rows), use_container_width=True)

    # Download
    csv_download = df_view[cols_show].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered data as CSV",
        data=csv_download,
        file_name="ferry_filtered.csv",
        mime="text/csv"
    )

    # Summary stats
    st.markdown("**Summary Statistics**")
    st.dataframe(df_view[["Sales Count", "Redemption Count"]].describe().round(2), use_container_width=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption("🛥️ Toronto Island Ferry Forecasting System | Unified Mentor Project | Data: Toronto Parks, Forestry & Recreation (2015–2025)")