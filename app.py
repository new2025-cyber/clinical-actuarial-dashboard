import io, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Optional validation (installed via requirements.txt)
import pandera as pa
from pandera import Column, DataFrameSchema, Check

st.set_page_config(page_title="Clinical–Actuarial Profiling Dashboard", layout="wide")
st.title("Clinical–Actuarial Scoring & Risk Dashboard")

# ====== REQUIRED & OPTIONAL COLUMNS ======
REQUIRED_COLUMNS = [
    "patient_id", "BAS", "CRS", "CARS", "PCS", "PPS", "FEI",
]
OPTIONAL_READMIT_COLS = ["readmitted_30d", "expected_readmit_rate"]
OPTIONAL_PROVIDER_COLS = ["provider_id"]     # for O/E by provider (optional)
OPTIONAL_TIME_COLS = ["period"]              # e.g., "2025-10" (optional)

# ====== HELPERS ======
def check_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing

def classify_patient(upi: float, hi=80, med=60) -> str:
    if upi >= hi:
        return "High Risk"
    elif upi >= med:
        return "Medium Risk"
    else:
        return "Low Risk"

def normalize_series(s: pd.Series, method: str) -> pd.Series:
    """MinMax 0–100 (if needed) or robust Quantile rank scaled to 0–100."""
    s = pd.to_numeric(s, errors="coerce")
    if method == "MinMax (0–100)":
        if s.min() >= 0 and s.max() <= 100:
            return s
        if s.max() == s.min():
            return pd.Series([100]*len(s), index=s.index)
        return (s - s.min()) / (s.max() - s.min()) * 100
    else:
        # Quantile rank (robust to outliers)
        return s.rank(pct=True) * 100

def calc_provider_penalty(row, readmit_weight=0.0):
    """
    Base penalty uses PCS and PPS_eff (risk-adjusted PPS if available).
    If readmission score exists, blend a readmission penalty with given weight.
    """
    base_pen = 0.6 * (100 - row["PCS"]) + 0.4 * (100 - row["PPS_eff"])
    if "provider_readmit_score" in row and pd.notnull(row["provider_readmit_score"]):
        readmit_pen = 100 - row["provider_readmit_score"]
        return (1 - readmit_weight) * base_pen + readmit_weight * readmit_pen
    else:
        return base_pen

def calc_upi(row, w_bas, w_crs, w_cars, w_penalty, w_fei):
    return (
        w_bas * row["BAS"]
        + w_crs * row["CRS"]
        + w_cars * row["CARS"]
        + w_penalty * row["provider_penalty"]
        + w_fei * row["FEI"]
    )

# ====== SIDEBAR: Weights, Readmission, Normalization, Thresholds, Settings I/O ======
st.sidebar.header("Weights Configuration")
w_bas = st.sidebar.slider("Weight: BAS", 0.0, 1.0, 0.25, 0.01)
w_crs = st.sidebar.slider("Weight: CRS", 0.0, 1.0, 0.25, 0.01)
w_cars = st.sidebar.slider("Weight: CARS", 0.0, 1.0, 0.25, 0.01)
w_penalty = st.sidebar.slider("Weight: Provider Penalty", 0.0, 1.0, 0.15, 0.01)
w_fei = st.sidebar.slider("Weight: FEI", 0.0, 1.0, 0.10, 0.01)

total_w = w_bas + w_crs + w_cars + w_penalty + w_fei
if abs(total_w - 1.0) > 0.001:
    st.sidebar.warning(f"Current total weight = {total_w:.2f}. Consider adjusting near 1.00.")

st.sidebar.header("Readmission Settings")
readmit_weight = st.sidebar.slider(
    "Weight of readmission in provider penalty", 0.0, 1.0, 0.20, 0.05
)
st.sidebar.caption("Applies only if 'readmitted_30d' and 'expected_readmit_rate' exist in the file.")

norm_method = st.sidebar.selectbox(
    "Normalization method",
    ["MinMax (0–100)", "Quantile rank (robust)"],
    index=0
)

st.sidebar.header("Classification thresholds")
thr_high = st.sidebar.slider("High-risk threshold", 70, 95, 80)
thr_med  = st.sidebar.slider("Medium threshold", 50, 79, 60)

st.sidebar.header("Settings I/O")
if st.sidebar.button("Save current settings to JSON"):
    settings = {
        "weights": {"BAS": w_bas, "CRS": w_crs, "CARS": w_cars, "Penalty": w_penalty, "FEI": w_fei},
        "thresholds": {"high": thr_high, "medium": thr_med},
        "norm_method": norm_method,
        "readmit_weight": readmit_weight,
    }
    st.session_state["_settings_json"] = json.dumps(settings, indent=2)
    st.sidebar.success("Settings captured. Download from the main panel.")

uploaded_settings = st.sidebar.file_uploader("Load settings JSON", type=["json"], key="settings_json")
if uploaded_settings is not None:
    try:
        s = json.load(uploaded_settings)
        w_bas = s["weights"]["BAS"]; w_crs = s["weights"]["CRS"]; w_cars = s["weights"]["CARS"]
        w_penalty = s["weights"]["Penalty"]; w_fei = s["weights"]["FEI"]
        thr_high = s["thresholds"]["high"]; thr_med = s["thresholds"]["medium"]
        norm_method = s["norm_method"]; readmit_weight = s["readmit_weight"]
        st.sidebar.success("Settings applied. Re-run calculations below.")
    except Exception as e:
        st.sidebar.error(f"Invalid settings file: {e}")

# ====== UPLOAD ======
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
if uploaded_file is None:
    st.info("Upload a file to start.")
    st.stop()

try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

missing = check_columns(df)
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ====== TABS: Data Quality / Dashboard ======
tab_quality, tab_dash = st.tabs(["🧪 Data Quality", "📊 Dashboard"])

with tab_quality:
    st.markdown(f"**Rows:** {df.shape[0]} &nbsp;&nbsp; **Columns:** {df.shape[1]}")
    st.markdown("**Missing values per required column:**")
    st.write(df[REQUIRED_COLUMNS].isna().sum())

    # Unique patient_id check
    if "patient_id" in df.columns:
        dup = df["patient_id"].duplicated().sum()
        if dup > 0:
            st.warning(f"Found {dup} duplicated patient_id(s).")

    # Strict schema (0–100) for required numeric columns
    SCHEMA = DataFrameSchema({
        "patient_id": Column(object),
        "BAS": Column(float, checks=Check.in_range(0, 100)),
        "CRS": Column(float, checks=Check.in_range(0, 100)),
        "CARS": Column(float, checks=Check.in_range(0, 100)),
        "PCS": Column(float, checks=Check.in_range(0, 100)),
        "PPS": Column(float, checks=Check.in_range(0, 100)),
        "FEI": Column(float, checks=Check.in_range(0, 100)),
    }, coerce=True)

    try:
        _ = SCHEMA.validate(df[REQUIRED_COLUMNS], lazy=True)
        st.success("Schema validation passed (types + 0–100 bounds).")
    except pa.errors.SchemaErrors as err:
        st.error("Schema validation failed. First issues shown below.")
        st.dataframe(err.failure_cases.head(25), use_container_width=False)

with tab_dash:
    # ====== NORMALIZATION ======
    data = df.copy()
    for col in ["BAS", "CRS", "CARS", "PCS", "PPS", "FEI"]:
        data[col] = normalize_series(df[col], norm_method)

    data = data.dropna(subset=["patient_id"])

    # ====== READMISSION (per-patient) ======
    if all(c in data.columns for c in OPTIONAL_READMIT_COLS):
        data["readmitted_30d"] = pd.to_numeric(data["readmitted_30d"], errors="coerce")
        data["expected_readmit_rate"] = pd.to_numeric(data["expected_readmit_rate"], errors="coerce")

        def calc_readmit_score(row):
            if pd.isna(row["expected_readmit_rate"]) or row["expected_readmit_rate"] == 0:
                return None
            observed = row["readmitted_30d"]  # 0/1
            expected = row["expected_readmit_rate"]  # e.g., 0.15
            score = 100 * (1 - (observed / expected))
            return max(0, min(100, score))

        data["provider_readmit_score"] = data.apply(calc_readmit_score, axis=1)
    else:
        data["provider_readmit_score"] = None

    # ====== PROVIDER O/E (risk-adjusted PPS with shrinkage), optional ======
    has_provider = all(c in data.columns for c in OPTIONAL_PROVIDER_COLS)
    has_readmit = all(c in data.columns for c in OPTIONAL_READMIT_COLS)
    if has_provider and has_readmit:
        # Clean numeric
        data["readmitted_30d"] = pd.to_numeric(data["readmitted_30d"], errors="coerce").fillna(0)
        data["expected_readmit_rate"] = pd.to_numeric(data["expected_readmit_rate"], errors="coerce")

        grp = data.groupby("provider_id").agg(
            obs=("readmitted_30d", "sum"),
            n=("readmitted_30d", "count"),
            exp=("expected_readmit_rate", "sum"),
        ).reset_index()

        grp["oe"] = grp.apply(lambda r: (r["obs"] / r["exp"]) if (pd.notna(r["exp"]) and r["exp"] > 0) else np.nan, axis=1)
        k = st.sidebar.slider("Shrinkage k (larger → more shrink)", 0, 200, 50, 5)

        total_exp = grp["exp"].sum()
        overall_oe = (grp["obs"].sum() / total_exp) if (pd.notna(total_exp) and total_exp > 0) else 1.0

        grp["weight"] = grp["n"] / (grp["n"] + k)
        grp["shrunk_oe"] = grp["weight"] * grp["oe"].fillna(overall_oe) + (1 - grp["weight"]) * overall_oe
        grp["pps_oe_score"] = (100 * (2 - grp["shrunk_oe"])).clip(lower=0, upper=100)

        data = data.merge(grp[["provider_id", "pps_oe_score"]], on="provider_id", how="left")
    else:
        data["pps_oe_score"] = np.nan

    # Choose effective PPS (risk-adjusted if available)
    data["PPS_eff"] = data["pps_oe_score"].where(data["pps_oe_score"].notna(), data["PPS"])

    # ====== PENALTY + UPI ======
    data["provider_penalty"] = data.apply(
        lambda r: calc_provider_penalty(r, readmit_weight=readmit_weight), axis=1
    )
    data["UPI"] = data.apply(
        lambda r: calc_upi(r, w_bas, w_crs, w_cars, w_penalty, w_fei), axis=1
    )
    data["risk_level"] = data["UPI"].apply(lambda x: classify_patient(x, thr_high, thr_med))

    # ====== KPIs ======
    total_patients = len(data)
    high_risk = (data["risk_level"] == "High Risk").sum()
    avg_upi = data["UPI"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Patients", total_patients)
    c2.metric("High-Risk Patients", high_risk)
    c3.metric("Average UPI", f"{avg_upi:.2f}")

    # ====== CHART ======
    st.subheader("UPI Distribution")
    fig = px.histogram(data, x="UPI", nbins=20, title="UPI Histogram")
    st.plotly_chart(fig, width="stretch")  # use_container_width deprecated

    with st.expander("ℹ️ Method Summary"):
        st.markdown(
            """
            **Normalization:** MinMax 0–100 or robust Quantile rank → 0–100  
            **Provider penalty:** 0.6×(100−PCS) + 0.4×(100−PPS_eff)  
            If readmission present: Final penalty = (1−r)×base + r×(100−ReadmissionScore)  
            **UPI:** BAS, CRS, CARS, provider_penalty, FEI (weights adjustable)  
            **Risk levels:** thresholds are configurable in the sidebar.
            """
        )

    # ====== TRENDS (optional) ======
    if all(c in data.columns for c in OPTIONAL_TIME_COLS):
        st.subheader("UPI Trends over time")
        try:
            agg = (
                data.groupby("period")["UPI"]
                .agg(["count", "mean"])
                .reset_index()
                .sort_values("period")
            )
            fig_tr = px.line(agg, x="period", y="mean", markers=True, title="Average UPI by period")
            st.plotly_chart(fig_tr, width="stretch")
            st.dataframe(agg, width="stretch")
        except Exception as e:
            st.warning(f"Cannot plot trends: {e}")

    # ====== TABLE ======
    st.subheader("Patient-Level Results")
    display_cols = [
        "patient_id", "BAS", "CRS", "CARS", "PCS", "PPS", "PPS_eff", "FEI",
        "provider_readmit_score", "provider_penalty", "UPI", "risk_level",
    ]
    display_cols = [c for c in display_cols if c in data.columns]
    st.dataframe(data[display_cols], width="stretch")

    # ====== DOWNLOAD ======
    st.subheader("Download Results")
    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download as CSV",
        data=csv_buffer.getvalue(),
        file_name="upi_results.csv",
        mime="text/csv",
    )

    if "_settings_json" in st.session_state:
        st.download_button(
            "Download Settings (JSON)",
            st.session_state["_settings_json"],
            file_name="dashboard_settings.json",
            mime="application/json",
        )

# ====== FOOTER SIGNATURE ======
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#1f77b4; font-size:16px;'>Developed by <b>Mudather</b></p>",
    unsafe_allow_html=True,
)
st.caption("Model v1.2 • " + pd.Timestamp.today().strftime("%Y-%m-%d"))


