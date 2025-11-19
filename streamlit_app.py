import os
import io
import csv
import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =====================================================================
# ADMIN & VERSION
# =====================================================================


def get_admin_key() -> str:
    """Reads ADMIN_KEY from Streamlit secrets if available."""
    try:
        return st.secrets["ADMIN_KEY"]
    except Exception:
        return ""


ADMIN_KEY = get_admin_key()
MODEL_VERSION = "v2.5-DRG-Timeline"

st.set_page_config(
    page_title="Clinical–Actuarial UPI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Clinical–Actuarial Scoring & Risk Dashboard (UPI + DRG variance)")

admin_input = st.sidebar.text_input("Admin key (optional)", type="password")
IS_ADMIN = ADMIN_KEY != "" and admin_input == ADMIN_KEY

if IS_ADMIN:
    st.sidebar.success("Admin mode active")
else:
    st.sidebar.caption("Public mode – admin features disabled")

# =====================================================================
# FILE UPLOAD
# =====================================================================

st.markdown("### 1) Upload Patient-Level Dataset")

uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded is None:
    st.info("Please upload a dataset file to continue.")
    st.stop()

dataset_name = uploaded.name
ext = dataset_name.split(".")[-1].lower()

if ext == "csv":
    data = pd.read_csv(uploaded)
else:
    data = pd.read_excel(uploaded)

# ---------------------------------------------------------------------
# REQUIRED / OPTIONAL COLUMNS
# ---------------------------------------------------------------------

required_base = [
    "patient_id",
    "BAS",
    "CRS",
    "PCS",
    "PPS",
    "FEI",
    "claims_amount",
    "provider_name",
    "drg_group",
    "period",  # شهر/ربع/سنة
]

missing_base = [c for c in required_base if c not in data.columns]
if missing_base:
    st.error(f"Missing required columns: {missing_base}")
    st.stop()

# expected_cost يمكن أن يكون بأحد اسمين
if "expected_cost" in data.columns:
    expected_col = "expected_cost"
elif "drg_expected_cost" in data.columns:
    expected_col = "drg_expected_cost"
else:
    st.error("Missing expected cost column: please add 'expected_cost' or 'drg_expected_cost'.")
    st.stop()

has_region = "region" in data.columns
has_premium = "premium_amount" in data.columns

# =====================================================================
# NORMALIZATION & WEIGHTS
# =====================================================================

st.sidebar.markdown("### Normalization method")

norm_method = st.sidebar.selectbox(
    "Choose normalization",
    ["MinMax (0–100)", "Z-score (50±10)", "None"],
    index=0,
)


def normalize(series: pd.Series) -> pd.Series:
    """Normalize numeric series according to selected method."""
    s = pd.to_numeric(series, errors="coerce")
    if norm_method == "None":
        return s
    if norm_method == "MinMax (0–100)":
        return 100.0 * (s - s.min()) / (s.max() - s.min() + 1e-9)
    if norm_method == "Z-score (50±10)":
        z = (s - s.mean()) / (s.std() + 1e-9)
        return 50.0 + 10.0 * z
    return s


st.sidebar.markdown("### UPI Weights")

wBAS = st.sidebar.slider("Weight BAS", 0.0, 1.0, 0.25)
wCRS = st.sidebar.slider("Weight CRS", 0.0, 1.0, 0.25)
wCARS = st.sidebar.slider("Weight CARS", 0.0, 1.0, 0.25)
wPEN = st.sidebar.slider("Weight Provider Penalty", 0.0, 1.0, 0.15)
wFEI = st.sidebar.slider("Weight FEI", 0.0, 1.0, 0.10)

total_w = wBAS + wCRS + wCARS + wPEN + wFEI
if abs(total_w - 1.0) > 0.02:
    st.sidebar.warning(f"Total weights = {total_w:.2f} (ideal = 1.00).")

st.sidebar.markdown("### Risk thresholds")

high_thr = st.sidebar.slider("High risk ≥", 50, 95, 80)
med_thr = st.sidebar.slider("Medium risk ≥", 40, 89, 60)

# =====================================================================
# DRG VARIANCE + CARS
# =====================================================================


def compute_drg_variance(df: pd.DataFrame, claims_col: str, exp_col: str) -> pd.DataFrame:
    """Compute DRG variance = (claims - expected) / expected * 100."""
    out = df.copy()
    claims = pd.to_numeric(out[claims_col], errors="coerce")
    expected = pd.to_numeric(out[exp_col], errors="coerce")
    out["drg_variance"] = (claims - expected) / (expected + 1e-9) * 100.0
    return out


def compute_cars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build CARS from:
    - CRS (clinical risk)
    - PCS (provider compliance)
    - drg_variance (financial deviation)
    """
    out = df.copy()
    crs = pd.to_numeric(out["CRS"], errors="coerce")
    pcs = pd.to_numeric(out["PCS"], errors="coerce")

    if "drg_variance" in out.columns and out["drg_variance"].notna().any():
        dv = pd.to_numeric(out["drg_variance"], errors="coerce").clip(-200, 200)
        dv_norm = 100.0 * (dv - dv.min()) / (dv.max() - dv.min() + 1e-9)
        out["CARS"] = 0.5 * crs + 0.3 * pcs + 0.2 * dv_norm
    else:
        out["CARS"] = crs

    return out


# ---------------------------------------------------------------------
# APPLY PIPELINE
# ---------------------------------------------------------------------

# 1) Normalize base scores
for col in ["BAS", "CRS", "PCS", "PPS", "FEI"]:
    data[col] = normalize(data[col])

# 2) Compute DRG variance
data = compute_drg_variance(data, "claims_amount", expected_col)

# 3) Compute CARS using CRS + PCS + DRG variance
data = compute_cars(data)
data["CARS"] = normalize(data["CARS"])

# =====================================================================
# PROVIDER PENALTY + UPI + RISK CLASS
# =====================================================================


def calc_provider_penalty(row: pd.Series) -> float:
    return 0.6 * (100.0 - row["PCS"]) + 0.4 * (100.0 - row["PPS"])


def calc_upi(row: pd.Series) -> float:
    return (
        wBAS * row["BAS"]
        + wCRS * row["CRS"]
        + wCARS * row["CARS"]
        + wPEN * row["provider_penalty"]
        + wFEI * row["FEI"]
    )


def classify_risk(upi: float) -> str:
    if upi >= float(high_thr):
        return "High Risk"
    if upi >= float(med_thr):
        return "Medium Risk"
    return "Low Risk"


data["provider_penalty"] = data.apply(calc_provider_penalty, axis=1)
data["UPI"] = data.apply(calc_upi, axis=1)
data["risk_level"] = data["UPI"].apply(classify_risk)

# =====================================================================
# LOGGING
# =====================================================================

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "upi_runs_log.csv")


def log_run(df: pd.DataFrame, ds_name: str) -> None:
    """Append summary of a run to CSV log."""
    os.makedirs(LOG_DIR, exist_ok=True)

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    total = len(df)
    high = int((df["risk_level"] == "High Risk").sum())
    pct = (100.0 * high / total) if total > 0 else 0.0
    avg_upi = float(df["UPI"].mean()) if total > 0 else 0.0

    weights = f"BAS={wBAS:.2f};CRS={wCRS:.2f};CARS={wCARS:.2f};PEN={wPEN:.2f};FEI={wFEI:.2f}"
    thresholds = f"high={high_thr};medium={med_thr}"

    header = [
        "timestamp",
        "model_version",
        "dataset_name",
        "total_patients",
        "high_risk_patients",
        "high_risk_pct",
        "avg_upi",
        "norm_method",
        "weights",
        "thresholds",
    ]

    row = [
        now,
        MODEL_VERSION,
        ds_name,
        total,
        high,
        f"{pct:.2f}",
        f"{avg_upi:.2f}",
        norm_method,
        weights,
        thresholds,
    ]

    exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


# =====================================================================
# TABS
# =====================================================================

tab1, tab2, tab3, tab_drg, tab_val, tab_logs = st.tabs(
    [
        "Level 1 – Strategic Executive Dashboard",
        "Level 2 – Facility Performance",
        "Level 3 – Patient Risk Panel",
        "DRG-Level Risk Comparison",
        "Model Validation Lab",
        "Model Governance / Logs",
    ]
)

# ---------------------------------------------------------------------
# LEVEL 1 – STRATEGIC
# ---------------------------------------------------------------------
with tab1:
    st.subheader("Level 1 – Strategic Executive Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total patients", len(data))
    c2.metric("High-risk patients", int((data["risk_level"] == "High Risk").sum()))
    c3.metric("Average UPI", f"{float(data['UPI'].mean()):.1f}")

    if st.button("Log this run (append to logs/upi_runs_log.csv)"):
        log_run(data, dataset_name)
        st.success("Run logged successfully.")

    st.markdown("---")

    st.markdown("### UPI Trend over time")
    trend = (
        data.groupby("period")["UPI"]
        .mean()
        .reset_index()
        .sort_values("period")
    )
    fig_trend = px.line(
        trend,
        x="period",
        y="UPI",
        markers=True,
        title="Average UPI by period",
        labels={"period": "Period", "UPI": "Avg UPI"},
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("### Loss ratio vs UPI (if premium available)")
    if has_premium:
        tmp = (
            data.groupby("period")
            .agg(
                avg_upi=("UPI", "mean"),
                claims=("claims_amount", "sum"),
                premium=("premium_amount", "sum"),
            )
            .reset_index()
        )
        tmp["loss_ratio"] = 100.0 * tmp["claims"] / (tmp["premium"] + 1e-9)
        fig_lr = px.line(
            tmp,
            x="period",
            y=["avg_upi", "loss_ratio"],
            markers=True,
            title="Average UPI vs Loss Ratio",
            labels={"value": "Value", "period": "Period", "variable": "Metric"},
        )
        st.plotly_chart(fig_lr, use_container_width=True)
    else:
        st.info("Column 'premium_amount' not found – loss ratio cannot be computed.")

    st.markdown("### Average UPI by region")
    if has_region:
        reg = data.groupby("region")["UPI"].mean().reset_index()
        fig_reg = px.bar(
            reg,
            x="region",
            y="UPI",
            title="Average UPI by region",
            labels={"region": "Region", "UPI": "Avg UPI"},
        )
        st.plotly_chart(fig_reg, use_container_width=True)
    else:
        st.info("Add 'region' column to see regional risk.")

    st.markdown("### Top 10 providers by average UPI")
    prov = (
        data.groupby("provider_name")["UPI"]
        .mean()
        .reset_index()
        .sort_values("UPI", ascending=False)
        .head(10)
    )
    fig_prov = px.bar(
        prov,
        x="provider_name",
        y="UPI",
        title="Top 10 providers by Avg UPI",
        labels={"provider_name": "Provider", "UPI": "Avg UPI"},
    )
    st.plotly_chart(fig_prov, use_container_width=True)
    st.dataframe(prov, use_container_width=True)

# ---------------------------------------------------------------------
# LEVEL 2 – FACILITY PERFORMANCE
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Level 2 – Facility Performance")

    providers = sorted(data["provider_name"].unique())
    selected_provider = st.selectbox("Select provider", providers)

    d = data[data["provider_name"] == selected_provider]

    c1, c2, c3 = st.columns(3)
    c1.metric("Patients", len(d))
    c2.metric("High-risk", int((d["risk_level"] == "High Risk").sum()))
    c3.metric("Avg UPI", f"{float(d['UPI'].mean()):.1f}")

    st.markdown("---")

    st.markdown("### Risk composition (mean of scores)")
    comp = d[["BAS", "CRS", "CARS", "PCS", "PPS", "FEI"]].mean().reset_index()
    comp.columns = ["Score", "Value"]
    fig_comp = px.bar(
        comp,
        x="Score",
        y="Value",
        title=f"Risk composition – {selected_provider}",
        labels={"Score": "Component", "Value": "Mean value"},
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("### Provider vs Network average UPI")
    col_a, col_b = st.columns(2)
    col_a.metric("Provider Avg UPI", f"{float(d['UPI'].mean()):.1f}")
    col_b.metric("Network Avg UPI", f"{float(data['UPI'].mean()):.1f}")

    st.markdown("### DRG risk trend for this provider")
    drg_trend = (
        d.groupby(["period", "drg_group"])["UPI"]
        .mean()
        .reset_index()
        .sort_values(["period", "drg_group"])
    )
    fig_drg_trend = px.line(
        drg_trend,
        x="period",
        y="UPI",
        color="drg_group",
        markers=True,
        title="Average UPI by DRG over time (selected provider)",
        labels={"period": "Period", "UPI": "Avg UPI", "drg_group": "DRG"},
    )
    st.plotly_chart(fig_drg_trend, use_container_width=True)

    st.markdown("### PCS (documentation/compliance) trend")
    pcs_trend = (
        d.groupby("period")["PCS"]
        .mean()
        .reset_index()
        .sort_values("period")
    )
    fig_pcs = px.line(
        pcs_trend,
        x="period",
        y="PCS",
        markers=True,
        title="PCS trend (selected provider)",
        labels={"period": "Period", "PCS": "PCS"},
    )
    st.plotly_chart(fig_pcs, use_container_width=True)

# ---------------------------------------------------------------------
# LEVEL 3 – PATIENT PANEL (with Timeline)
# ---------------------------------------------------------------------
with tab3:
    st.subheader("Level 3 – Patient Risk Panel")

    # اختيار المريض
    patients = sorted(data["patient_id"].unique())
    selected_patient = st.selectbox("Select patient", patients)

    # كل بيانات هذا المريض عبر الفترات
    pt = data[data["patient_id"] == selected_patient].copy()
    pt = pt.sort_values("period")
    latest = pt.iloc[-1]

    # كروت ملخصة
    c1, c2, c3 = st.columns(3)
    c1.metric("UPI (latest)", f"{float(latest['UPI']):.1f}")
    c2.metric("Risk level", latest["risk_level"])
    c3.metric("Provider (latest)", str(latest["provider_name"]))

    st.markdown("---")

    # ---------------- Risk drivers (latest snapshot) ----------------
    st.markdown("### A) Risk drivers – latest record")

    drivers = pd.DataFrame(
        {
            "Score": ["BAS", "CRS", "CARS", "PCS", "PPS", "FEI"],
            "Value": [
                latest["BAS"],
                latest["CRS"],
                latest["CARS"],
                latest["PCS"],
                latest["PPS"],
                latest["FEI"],
            ],
        }
    )

    fig_drv = px.bar(
        drivers,
        x="Score",
        y="Value",
        title="Risk drivers (latest)",
        labels={"Score": "Component", "Value": "Score value"},
    )
    st.plotly_chart(fig_drv, use_container_width=True)

    # Provider penalty snapshot
    c4, c5 = st.columns(2)
    c4.metric("PCS (latest)", f"{float(latest['PCS']):.1f}")
    c5.metric("Provider penalty (latest)", f"{float(latest['provider_penalty']):.1f}")

    st.markdown("---")

    # ---------------- Patient Risk Timeline ----------------
    st.markdown("### B) Patient Risk Timeline (multi-metric)")

    if len(pt) >= 2:
        # المقاييس التي نتابعها عبر الزمن
        timeline_metrics = ["UPI", "BAS", "CRS", "CARS", "PCS", "PPS", "FEI"]
        timeline_df = pt[["period"] + timeline_metrics].copy()

        tidy = timeline_df.melt(
            id_vars="period",
            value_vars=timeline_metrics,
            var_name="Metric",
            value_name="Value",
        )

        fig_timeline = px.line(
            tidy,
            x="period",
            y="Value",
            color="Metric",
            markers=True,
            title="Risk metrics over time for selected patient",
            labels={"period": "Period", "Value": "Score value", "Metric": "Metric"},
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        st.caption(
            "هذه الرسوم توضح تطوّر UPI وباقي المؤشرات (BAS, CRS, CARS, PCS, PPS, FEI) "
            "عبر الفترات الزمنية لهذا المريض."
        )
    else:
        st.info("Only one record available for this patient – timeline requires at least 2 periods.")

    st.markdown("---")

    # ---------------- Financial Timeline ----------------
    st.markdown("### C) Financial & DRG variance Timeline")

    if len(pt) >= 2:
        fin_df = pt[["period", "claims_amount", "drg_variance"]].copy()

        # claims over time
        fig_claims = px.bar(
            fin_df,
            x="period",
            y="claims_amount",
            title="Claims amount over time",
            labels={"period": "Period", "claims_amount": "Claims amount"},
        )
        st.plotly_chart(fig_claims, use_container_width=True)

        # DRG variance over time
        fig_var = px.line(
            fin_df,
            x="period",
            y="drg_variance",
            markers=True,
            title="DRG variance (%) over time",
            labels={"period": "Period", "drg_variance": "DRG variance (%)"},
        )
        st.plotly_chart(fig_var, use_container_width=True)

        st.caption(
            "الرسوم أعلاه تربط بين تطوّر التكلفة الفعلية (claims) والانحراف عن تكلفة DRG المتوقعة (DRG variance) "
            "لنفس المريض عبر الفترات."
        )
    else:
        st.info("Financial timeline requires at least 2 records for this patient.")

# ---------------------------------------------------------------------
# DRG-LEVEL RISK COMPARISON
# ---------------------------------------------------------------------
with tab_drg:
    st.subheader("DRG-Level Risk Comparison")

    st.markdown("### 1) Average UPI by DRG (network)")
    drg_rank = (
        data.groupby("drg_group")["UPI"]
        .mean()
        .reset_index()
        .sort_values("UPI", ascending=False)
    )
    fig_drg_rank = px.bar(
        drg_rank,
        x="drg_group",
        y="UPI",
        title="Average UPI by DRG (network)",
        labels={"drg_group": "DRG", "UPI": "Avg UPI"},
    )
    st.plotly_chart(fig_drg_rank, use_container_width=True)
    st.dataframe(drg_rank, use_container_width=True)

    st.markdown("### 2) Average DRG variance (%) by DRG (network)")
    drg_var_rank = (
        data.groupby("drg_group")["drg_variance"]
        .mean()
        .reset_index()
        .sort_values("drg_variance", ascending=False)
    )
    fig_drg_var = px.bar(
        drg_var_rank,
        x="drg_group",
        y="drg_variance",
        title="Average DRG variance by DRG (network)",
        labels={"drg_group": "DRG", "drg_variance": "Avg DRG variance (%)"},
    )
    st.plotly_chart(fig_drg_var, use_container_width=True)
    st.dataframe(drg_var_rank, use_container_width=True)

    st.markdown("### 3) Provider × DRG – UPI heatmap")
    matrix = (
        data.groupby(["provider_name", "drg_group"])["UPI"]
        .mean()
        .reset_index()
    )
    pivot = matrix.pivot(index="provider_name", columns="drg_group", values="UPI")
    st.dataframe(pivot, use_container_width=True)

    fig_hm = px.imshow(
        pivot,
        aspect="auto",
        labels={"x": "DRG", "y": "Provider", "color": "Avg UPI"},
        title="Provider × DRG – Average UPI heatmap",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("### 4) DRG profile for a selected provider")
    providers2 = sorted(data["provider_name"].unique())
    selected_provider2 = st.selectbox(
        "Select provider for DRG profile", providers2, key="prov_drg"
    )

    df_prov = data[data["provider_name"] == selected_provider2]
    drg_prov = (
        df_prov.groupby("drg_group")["UPI"]
        .mean()
        .reset_index()
        .sort_values("UPI", ascending=False)
    )
    fig_drg_prov = px.bar(
        drg_prov,
        x="drg_group",
        y="UPI",
        title=f"Average UPI by DRG – {selected_provider2}",
        labels={"drg_group": "DRG", "UPI": "Avg UPI"},
    )
    st.plotly_chart(fig_drg_prov, use_container_width=True)
    st.dataframe(drg_prov, use_container_width=True)

# ---------------------------------------------------------------------
# MODEL VALIDATION LAB
# ---------------------------------------------------------------------
with tab_val:
    st.subheader("Model Validation Lab – Correlation & Calibration")

    # 1) Correlation matrix
    st.markdown("### 1) Correlation matrix")

    candidate_cols = [
        "BAS",
        "CRS",
        "CARS",
        "PCS",
        "PPS",
        "FEI",
        "UPI",
        "drg_variance",
        "claims_amount",
    ]
    corr_cols = [c for c in candidate_cols if c in data.columns]

    if len(corr_cols) >= 2:
        corr_df = data[corr_cols].apply(pd.to_numeric, errors="coerce").corr()
        st.dataframe(corr_df, use_container_width=True)
        fig_corr = px.imshow(
            corr_df,
            aspect="auto",
            text_auto=True,
            labels={"color": "Correlation"},
            title="Correlation heatmap",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric columns to compute correlation matrix.")

    st.markdown("---")

    # 2) UPI vs claims
    st.markdown("### 2) UPI vs claims amount")

    df_sc = data.copy()
    df_sc["claims_amount"] = pd.to_numeric(df_sc["claims_amount"], errors="coerce")
    df_sc = df_sc.dropna(subset=["UPI", "claims_amount"])

    if len(df_sc) > 0:
        fig_sc = px.scatter(
            df_sc,
            x="UPI",
            y="claims_amount",
            color="provider_name",
            trendline="ols",  # requires statsmodels
            title="UPI vs claims amount (with OLS trendline)",
            labels={"UPI": "UPI", "claims_amount": "Claims amount"},
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        corr_upi_claims = df_sc["UPI"].corr(df_sc["claims_amount"])
        st.caption(f"Correlation(UPI, claims) = {corr_upi_claims:.3f}")
    else:
        st.info("Not enough valid data to plot UPI vs claims.")

    st.markdown("---")

    # 3) Provider avg UPI vs avg DRG variance
    st.markdown("### 3) Provider avg UPI vs avg DRG variance")

    prov_val = (
        data.groupby("provider_name")
        .agg(
            avg_upi=("UPI", "mean"),
            avg_drg_var=("drg_variance", "mean"),
            n=("patient_id", "count"),
        )
        .reset_index()
    )

    if len(prov_val) > 0:
        fig_pv = px.scatter(
            prov_val,
            x="avg_upi",
            y="avg_drg_var",
            size="n",
            hover_name="provider_name",
            title="Provider avg UPI vs avg DRG variance",
            labels={
                "avg_upi": "Avg UPI",
                "avg_drg_var": "Avg DRG variance (%)",
                "n": "Patients",
            },
        )
        st.plotly_chart(fig_pv, use_container_width=True)
    else:
        st.info("No provider-level aggregates available.")

    st.markdown("---")

    # 4) DRG variance distribution
    st.markdown("### 4) DRG variance distribution")
    dv = pd.to_numeric(data["drg_variance"], errors="coerce").dropna()
    if len(dv) > 0:
        fig_hist = px.histogram(
            dv,
            nbins=40,
            title="DRG variance (%) distribution",
            labels={"value": "DRG variance (%)", "count": "Count"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No DRG variance data to display.")

# ---------------------------------------------------------------------
# LOGS
# ---------------------------------------------------------------------
with tab_logs:
    st.subheader("Model Governance / Logs")

    st.markdown("### Run history (from logs/upi_runs_log.csv)")

    if os.path.isfile(LOG_FILE):
        log_df = pd.read_csv(LOG_FILE)
        log_df = log_df.sort_values("timestamp", ascending=False)
        st.dataframe(log_df, use_container_width=True)

        if len(log_df) >= 2:
            trend_log = log_df.copy()
            fig_log_trend = px.line(
                trend_log,
                x="timestamp",
                y="avg_upi",
                markers=True,
                title="Logged runs – Average UPI over time",
                labels={"timestamp": "Timestamp", "avg_upi": "Avg UPI"},
            )
            st.plotly_chart(fig_log_trend, use_container_width=True)
    else:
        st.info("No log file found yet. Use 'Log this run' in Level 1 to create it.")

# =====================================================================
# FOOTER SIGNATURE
# =====================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#1f77b4; font-size:16px;'>Developed by "
    "<b>Mudather</b></p>",
    unsafe_allow_html=True,
)
st.caption("Model " + MODEL_VERSION + " • " + pd.Timestamp.today().strftime("%Y-%m-%d"))
