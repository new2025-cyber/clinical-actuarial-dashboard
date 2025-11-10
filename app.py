import io
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Clinical–Actuarial Profiling Dashboard", layout="wide")
st.title("Clinical–Actuarial Scoring & Risk Dashboard")

REQUIRED_COLUMNS = [
    "patient_id",
    "BAS",
    "CRS",
    "CARS",
    "PCS",
    "PPS",
    "FEI",
]

# Optional columns for readmission analysis
OPTIONAL_READMIT_COLS = ["readmitted_30d", "expected_readmit_rate"]


def check_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing


def normalize_to_0_100(series: pd.Series) -> pd.Series:
    if series.min() >= 0 and series.max() <= 100:
        return series
    smin = series.min()
    smax = series.max()
    if smax == smin:
        return pd.Series([100] * len(series), index=series.index)
    return (series - smin) / (smax - smin) * 100


def calc_provider_penalty(row, readmit_weight=0.0):
    """
    Base: 0.6*(100-PCS) + 0.4*(100-PPS)
    If readmission columns exist, blend readmission penalty using the given weight.
    """
    base_pen = 0.6 * (100 - row["PCS"]) + 0.4 * (100 - row["PPS"])
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


def classify_patient(upi: float) -> str:
    if upi >= 80:
        return "High Risk"
    elif upi >= 60:
        return "Medium Risk"
    else:
        return "Low Risk"


# ---------------- SIDEBAR ----------------
st.sidebar.header("Weights Configuration")
w_bas = st.sidebar.slider("Weight: BAS", 0.0, 1.0, 0.25, 0.01)
w_crs = st.sidebar.slider("Weight: CRS", 0.0, 1.0, 0.25, 0.01)
w_cars = st.sidebar.slider("Weight: CARS", 0.0, 1.0, 0.25, 0.01)
w_penalty = st.sidebar.slider("Weight: Provider Penalty", 0.0, 1.0, 0.15, 0.01)
w_fei = st.sidebar.slider("Weight: FEI", 0.0, 1.0, 0.10, 0.01)

total_w = w_bas + w_crs + w_cars + w_penalty + w_fei
if abs(total_w - 1.0) > 0.001:
    st.sidebar.warning(f"Current total weight = {total_w:.2f}. Adjust to ≈ 1.00 for accuracy.")

# --- Readmission Settings ---
st.sidebar.header("Readmission Settings")
readmit_weight = st.sidebar.slider(
    "Weight of readmission in provider penalty", 0.0, 1.0, 0.20, 0.05
)
st.sidebar.caption(
    "Applies only if 'readmitted_30d' and 'expected_readmit_rate' columns exist in the uploaded file."
)

# ---------------- UPLOAD ----------------
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

data = df.copy()

# Normalize all main columns
for col in ["BAS", "CRS", "CARS", "PCS", "PPS", "FEI"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data[col] = normalize_to_0_100(data[col])

data = data.dropna(subset=["patient_id"])

# ---------------- READMISSION LOGIC ----------------
if all(col in data.columns for col in OPTIONAL_READMIT_COLS):
    data["readmitted_30d"] = pd.to_numeric(data["readmitted_30d"], errors="coerce")
    data["expected_readmit_rate"] = pd.to_numeric(
        data["expected_readmit_rate"], errors="coerce"
    )

    def calc_readmit_score(row):
        if pd.isna(row["expected_readmit_rate"]) or row["expected_readmit_rate"] == 0:
            return None
        observed = row["readmitted_30d"]
        expected = row["expected_readmit_rate"]
        score = 100 * (1 - (observed / expected))
        return max(0, min(100, score))

    data["provider_readmit_score"] = data.apply(calc_readmit_score, axis=1)
else:
    data["provider_readmit_score"] = None

# ---------------- PENALTY + UPI ----------------
data["provider_penalty"] = data.apply(
    lambda r: calc_provider_penalty(r, readmit_weight=readmit_weight), axis=1
)
data["UPI"] = data.apply(
    lambda r: calc_upi(r, w_bas, w_crs, w_cars, w_penalty, w_fei), axis=1
)
data["risk_level"] = data["UPI"].apply(classify_patient)

# ---------------- KPIs ----------------
total_patients = len(data)
high_risk = (data["risk_level"] == "High Risk").sum()
avg_upi = data["UPI"].mean()

c1, c2, c3 = st.columns(3)
c1.metric("Total Patients", total_patients)
c2.metric("High-Risk Patients", high_risk)
c3.metric("Average UPI", f"{avg_upi:.2f}")

# ---------------- CHART ----------------
st.subheader("UPI Distribution")
fig = px.histogram(data, x="UPI", nbins=20, title="UPI Histogram")
st.plotly_chart(fig, width="stretch")

with st.expander("ℹ️ Explanation"):
    st.markdown("""
    **Provider Penalty** = 0.6×(100−PCS) + 0.4×(100−PPS)  
    If readmission data are provided:  
    Final penalty = (1−r)×base + r×(100−ReadmissionScore)  
    where *r* = readmission weight slider.
    """)

# ---------------- TABLE ----------------
st.subheader("Patient-Level Results")
display_cols = [
    "patient_id",
    "BAS",
    "CRS",
    "CARS",
    "PCS",
    "PPS",
    "FEI",
    "provider_readmit_score",
    "provider_penalty",
    "UPI",
    "risk_level",
]
display_cols = [c for c in display_cols if c in data.columns]
st.dataframe(data[display_cols], width="stretch")

# ---------------- DOWNLOAD ----------------
st.subheader("Download Results")
csv_buffer = io.StringIO()
data.to_csv(csv_buffer, index=False)
st.download_button(
    label="Download as CSV",
    data=csv_buffer.getvalue(),
    file_name="upi_results.csv",
    mime="text/csv",
)

# ---------------- SIGNATURE ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#1f77b4; font-size:16px;'>Developed by <b>Mudather</b></p>",
    unsafe_allow_html=True,
)
