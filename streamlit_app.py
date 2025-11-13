import io
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import pandera as pa
from pandera import Column, DataFrameSchema, Check

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

from streamlit_autorefresh import st_autorefresh

# ----------------- Page config -----------------
st.set_page_config(page_title="Clinical–Actuarial Profiling Dashboard", layout="wide")
st.title("Clinical–Actuarial Scoring & Risk Dashboard")


# ----------------- Admin Mode -----------------
def resolve_secret(key: str, default: str = ""):
    try:
        return st.secrets[key]
    except Exception:
        return default


ADMIN_KEY = resolve_secret("ADMIN_KEY", "")  # set in Streamlit Cloud → Settings → Secrets

admin_input = st.sidebar.text_input(
    "Admin key", type="password", help="Leave blank for read-only view"
)

if ADMIN_KEY:
    # Cloud: only exact match is admin
    is_admin = admin_input == ADMIN_KEY
else:
    # Local: any non-empty value is admin
    is_admin = admin_input.strip() != ""

DISABLE_KW = {"disabled": (not is_admin)}

# ----------------- Auto-refresh -----------------
if not is_admin:
    # Public viewers: auto-refresh every 15 minutes
    st_autorefresh(interval=900_000, key="public_autorefresh")
    st.sidebar.caption("Auto-refresh: every 15 minutes (read-only viewers).")
else:
    refresh_mins = st.sidebar.selectbox(
        "Auto-refresh (admin)", [0, 10, 15, 30], index=0
    )
    if refresh_mins > 0:
        st_autorefresh(interval=refresh_mins * 60 * 1000, key="admin_autorefresh")


# ----------------- Column definitions -----------------
REQUIRED_COLUMNS = ["patient_id", "BAS", "CRS", "CARS", "PCS", "PPS", "FEI"]
OPTIONAL_READMIT_COLS = ["readmitted_30d", "expected_readmit_rate"]
OPTIONAL_PROVIDER_COLS = ["provider_id"]
OPTIONAL_TIME_COLS = ["period"]


def check_columns(df: pd.DataFrame):
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def classify_patient(upi: float, hi=80, med=60) -> str:
    if upi >= hi:
        return "High Risk"
    if upi >= med:
        return "Medium Risk"
    return "Low Risk"


def normalize_series(s: pd.Series, method: str) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if method == "MinMax (0–100)":
        if s.min() >= 0 and s.max() <= 100:
            return s
        if s.max() == s.min():
            return pd.Series([100] * len(s), index=s.index)
        return (s - s.min()) / (s.max() - s.min()) * 100
    # Quantile rank (robust to outliers)
    return s.rank(pct=True) * 100


def calc_provider_penalty(row, readmit_weight=0.0):
    base_pen = 0.6 * (100 - row["PCS"]) + 0.4 * (100 - row["PPS_eff"])
    if "provider_readmit_score" in row and pd.notnull(row["provider_readmit_score"]):
        readmit_pen = 100 - row["provider_readmit_score"]
        return (1 - readmit_weight) * base_pen + readmit_weight * readmit_pen
    return base_pen


def calc_upi(row, w_bas, w_crs, w_cars, w_penalty, w_fei):
    return (
        w_bas * row["BAS"]
        + w_crs * row["CRS"]
        + w_cars * row["CARS"]
        + w_penalty * row["provider_penalty"]
        + w_fei * row["FEI"]
    )


def generate_pdf_report(
    data: pd.DataFrame,
    total_patients: int,
    high_risk: int,
    avg_upi: float,
    w_bas: float,
    w_crs: float,
    w_cars: float,
    w_penalty: float,
    w_fei: float,
    thr_high: float,
    thr_med: float,
    norm_method: str,
    readmit_weight: float,
) -> io.BytesIO:
    """
    Build a PDF report (KPIs + config + Top 20 High-Risk).
    Returns an in-memory BytesIO ready for download.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    margin = 2 * cm
    y = height - margin

    def line(text: str, font="Helvetica", size=10, leading=12):
        nonlocal y
        if y < margin:
            c.showPage()
            y = height - margin
        c.setFont(font, size)
        c.drawString(margin, y, text)
        y -= leading

    # Header
    line("Clinical–Actuarial Dashboard Summary", font="Helvetica-Bold", size=14, leading=18)
    line(f"Generated on: {pd.Timestamp.today().strftime('%Y-%m-%d %H:%M')}")
    line("")

    # KPIs
    line("Key Performance Indicators", font="Helvetica-Bold", size=12, leading=16)
    line(f"Total patients: {total_patients}")
    line(f"High-risk patients: {high_risk}")
    line(f"Average UPI: {avg_upi:.2f}")
    line("")

    # Configuration
    line("Configuration", font="Helvetica-Bold", size=12, leading=16)
    line(
        f"Weights: BAS={w_bas:.2f}, CRS={w_crs:.2f}, CARS={w_cars:.2f}, "
        f"Penalty={w_penalty:.2f}, FEI={w_fei:.2f}"
    )
    line(f"Thresholds: High >= {thr_high}, Medium >= {thr_med}")
    line(f"Normalization: {norm_method}")
    line(f"Readmission weight in penalty: {readmit_weight:.2f}")
    line("")

    # Top 20 High-Risk
    line("Top 20 High-Risk Patients", font="Helvetica-Bold", size=12, leading=16)
    if {"UPI", "risk_level", "patient_id"}.issubset(set(data.columns)):
        top = (
            data.sort_values("UPI", ascending=False)
            .head(20)[["patient_id", "UPI", "risk_level"]]
        )
        for _, r in top.iterrows():
            line(f"- {r['patient_id']}: UPI={r['UPI']:.1f}, level={r['risk_level']}")
    else:
        line("UPI / risk_level / patient_id columns not available.", size=9)

    line("")
    line("Notes:", font="Helvetica-Bold", size=11, leading=14)
    line("• UPI combines behavioral, clinical, provider, and financial indices.", size=9)
    line("• Risk levels follow configured thresholds at generation time.", size=9)

    # Signature
    line("")
    line("Developed by Mudather", font="Helvetica-Oblique", size=10, leading=14)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf


# ----------------- Sidebar: configuration -----------------
st.sidebar.header("Weights Configuration")
w_bas = st.sidebar.slider("Weight: BAS", 0.0, 1.0, 0.25, 0.01, **DISABLE_KW)
w_crs = st.sidebar.slider("Weight: CRS", 0.0, 1.0, 0.25, 0.01, **DISABLE_KW)
w_cars = st.sidebar.slider("Weight: CARS", 0.0, 1.0, 0.25, 0.01, **DISABLE_KW)
w_penalty = st.sidebar.slider("Weight: Provider Penalty", 0.0, 1.0, 0.15, 0.01, **DISABLE_KW)
w_fei = st.sidebar.slider("Weight: FEI", 0.0, 1.0, 0.10, 0.01, **DISABLE_KW)

total_w = w_bas + w_crs + w_cars + w_penalty + w_fei
if abs(total_w - 1.0) > 0.001:
    st.sidebar.warning(f"Current total weight = {total_w:.2f}. Consider adjusting near 1.00.")

st.sidebar.header("Readmission Settings")
readmit_weight = st.sidebar.slider(
    "Weight of readmission in provider penalty",
    0.0,
    1.0,
    0.20,
    0.05,
    **DISABLE_KW,
)
st.sidebar.caption("Applies only if 'readmitted_30d' and 'expected_readmit_rate' exist in the file.")

norm_method = st.sidebar.selectbox(
    "Normalization method",
    ["MinMax (0–100)", "Quantile rank (robust)"],
    index=0,
    **DISABLE_KW,
)

st.sidebar.header("Classification thresholds")
thr_high = st.sidebar.slider("High-risk threshold", 70, 95, 80, **DISABLE_KW)
thr_med = st.sidebar.slider("Medium threshold", 50, 79, 60, **DISABLE_KW)

st.sidebar.header("Settings I/O")
if st.sidebar.button("Save current settings to JSON", **DISABLE_KW):
    settings = {
        "weights": {
            "BAS": w_bas,
            "CRS": w_crs,
            "CARS": w_cars,
            "Penalty": w_penalty,
            "FEI": w_fei,
        },
        "thresholds": {"high": thr_high, "medium": thr_med},
        "norm_method": norm_method,
        "readmit_weight": readmit_weight,
    }
    st.session_state["_settings_json"] = json.dumps(settings, indent=2)
    st.sidebar.success("Settings captured. Download from the main panel.")

uploaded_settings = st.sidebar.file_uploader(
    "Load settings JSON", type=["json"], key="settings_json", **DISABLE_KW
)
if uploaded_settings is not None:
    try:
        s = json.load(uploaded_settings)
        w_bas = s["weights"]["BAS"]
        w_crs = s["weights"]["CRS"]
        w_cars = s["weights"]["CARS"]
        w_penalty = s["weights"]["Penalty"]
        w_fei = s["weights"]["FEI"]
        thr_high = s["thresholds"]["high"]
        thr_med = s["thresholds"]["medium"]
        norm_method = s["norm_method"]
        readmit_weight = s["readmit_weight"]
        st.sidebar.success("Settings applied. Re-run calculations below.")
    except Exception as e:
        st.sidebar.error(f"Invalid settings file: {e}")


# ----------------- File upload -----------------
uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    disabled=not is_admin,  # only admin can upload
)
if uploaded_file is None:
    st.info("Upload a file to start (admin only).")
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


# ----------------- Tabs -----------------
tab_quality, tab_dash, tab_fb, tab_about = st.tabs(
    ["🧪 Data Quality", "📊 Dashboard", "💬 Feedback Insights", "ℹ About & Methodology"]
)


# ----------------- Data Quality tab -----------------
with tab_quality:
    st.markdown("### Data Quality Overview")
    st.write(f"**Rows:** {df.shape[0]} &nbsp;&nbsp; **Columns:** {df.shape[1]}")
    st.markdown("**Missing values per required column:**")
    st.write(df[REQUIRED_COLUMNS].isna().sum())

    if "patient_id" in df.columns:
        dup = df["patient_id"].duplicated().sum()
        if dup > 0:
            st.warning(f"Found {dup} duplicated patient_id(s).")

    SCHEMA = DataFrameSchema(
        {
            "patient_id": Column(object),
            "BAS": Column(float, checks=Check.in_range(0, 100)),
            "CRS": Column(float, checks=Check.in_range(0, 100)),
            "CARS": Column(float, checks=Check.in_range(0, 100)),
            "PCS": Column(float, checks=Check.in_range(0, 100)),
            "PPS": Column(float, checks=Check.in_range(0, 100)),
            "FEI": Column(float, checks=Check.in_range(0, 100)),
        },
        coerce=True,
    )

    st.markdown("**Schema validation (types + 0–100 bounds):**")
    try:
        _ = SCHEMA.validate(df[REQUIRED_COLUMNS], lazy=True)
        st.success("Schema validation passed.")
    except pa.errors.SchemaErrors as err:
        st.error("Schema validation failed. First issues shown below.")
        try:
            st.dataframe(err.failure_cases.head(25), width="stretch")
        except Exception:
            st.text(str(err)[:2000])

    st.caption("Fix schema issues in your file then re-upload for clean calculations.")


# ----------------- Dashboard tab -----------------
with tab_dash:
    # Normalization
    data = df.copy()
    for col in ["BAS", "CRS", "CARS", "PCS", "PPS", "FEI"]:
        data[col] = normalize_series(df[col], norm_method)

    data = data.dropna(subset=["patient_id"])

    # Readmission score per patient (optional)
    if all(c in data.columns for c in OPTIONAL_READMIT_COLS):
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

    # Provider O/E with shrinkage (optional)
    has_provider = all(c in data.columns for c in OPTIONAL_PROVIDER_COLS)
    has_readmit = all(c in data.columns for c in OPTIONAL_READMIT_COLS)
    if has_provider and has_readmit:
        data["readmitted_30d"] = pd.to_numeric(
            data["readmitted_30d"], errors="coerce"
        ).fillna(0)
        data["expected_readmit_rate"] = pd.to_numeric(
            data["expected_readmit_rate"], errors="coerce"
        )

        grp = (
            data.groupby("provider_id")
            .agg(
                obs=("readmitted_30d", "sum"),
                n=("readmitted_30d", "count"),
                exp=("expected_readmit_rate", "sum"),
            )
            .reset_index()
        )

        grp["oe"] = grp.apply(
            lambda r: (r["obs"] / r["exp"])
            if (pd.notna(r["exp"]) and r["exp"] > 0)
            else np.nan,
            axis=1,
        )

        k = st.sidebar.slider(
            "Shrinkage k (larger → more shrink)", 0, 200, 50, 5, **DISABLE_KW
        )

        total_exp = grp["exp"].sum()
        overall_oe = (
            grp["obs"].sum() / total_exp if (pd.notna(total_exp) and total_exp > 0) else 1.0
        )

        grp["weight"] = grp["n"] / (grp["n"] + k)
        grp["shrunk_oe"] = grp["weight"] * grp["oe"].fillna(overall_oe) + (
            1 - grp["weight"]
        ) * overall_oe
        grp["pps_oe_score"] = (100 * (2 - grp["shrunk_oe"])).clip(lower=0, upper=100)

        data = data.merge(
            grp[["provider_id", "pps_oe_score"]], on="provider_id", how="left"
        )
    else:
        data["pps_oe_score"] = np.nan

    data["PPS_eff"] = data["pps_oe_score"].where(
        data["pps_oe_score"].notna(), data["PPS"]
    )

    # Penalty + UPI
    data["provider_penalty"] = data.apply(
        lambda r: calc_provider_penalty(r, readmit_weight=readmit_weight), axis=1
    )
    data["UPI"] = data.apply(
        lambda r: calc_upi(r, w_bas, w_crs, w_cars, w_penalty, w_fei), axis=1
    )
    data["risk_level"] = data["UPI"].apply(
        lambda x: classify_patient(x, thr_high, thr_med)
    )

    # KPIs
    total_patients = len(data)
    high_risk = (data["risk_level"] == "High Risk").sum()
    avg_upi = data["UPI"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Patients", total_patients)
    c2.metric("High-Risk Patients", high_risk)
    c3.metric("Average UPI", f"{avg_upi:.2f}")

    # Histogram
    st.subheader("UPI Distribution")
    fig = px.histogram(data, x="UPI", nbins=20, title="UPI Histogram")
    st.plotly_chart(fig, width="stretch")

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

    # Optional trends
    if all(c in data.columns for c in OPTIONAL_TIME_COLS):
        st.subheader("UPI Trends over time")
        try:
            agg = (
                data.groupby("period")["UPI"]
                .agg(["count", "mean"])
                .reset_index()
                .sort_values("period")
            )
            fig_tr = px.line(
                agg, x="period", y="mean", markers=True, title="Average UPI by period"
            )
            st.plotly_chart(fig_tr, width="stretch")
            st.dataframe(agg, width="stretch")
        except Exception as e:
            st.warning(f"Cannot plot trends: {e}")

    # Table
    st.subheader("Patient-Level Results")
    display_cols = [
        "patient_id",
        "BAS",
        "CRS",
        "CARS",
        "PCS",
        "PPS",
        "PPS_eff",
        "FEI",
        "provider_readmit_score",
        "provider_penalty",
        "UPI",
        "risk_level",
    ]
    display_cols = [c for c in display_cols if c in data.columns]
    st.dataframe(data[display_cols], width="stretch")

    # PDF Report (admin only)
    st.subheader("PDF Summary Report")
    st.write(
        "Generate a printable PDF containing KPIs, configuration, "
        "and top 20 high-risk patients."
    )

    if st.button("Generate PDF report", disabled=not is_admin):
        try:
            pdf_buffer = generate_pdf_report(
                data=data,
                total_patients=total_patients,
                high_risk=high_risk,
                avg_upi=avg_upi,
                w_bas=w_bas,
                w_crs=w_crs,
                w_cars=w_cars,
                w_penalty=w_penalty,
                w_fei=w_fei,
                thr_high=thr_high,
                thr_med=thr_med,
                norm_method=norm_method,
                readmit_weight=readmit_weight,
            )
            st.download_button(
                "Download PDF report",
                data=pdf_buffer,
                file_name="upi_summary_report.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Failed to generate PDF: {e}")

    # Downloads
    st.subheader("Download Results (CSV)")
    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download as CSV",
        csv_buffer.getvalue(),
        file_name="upi_results.csv",
        mime="text/csv",
        disabled=not is_admin,
    )

    if "_settings_json" in st.session_state:
        st.download_button(
            "Download Settings (JSON)",
            st.session_state["_settings_json"],
            file_name="dashboard_settings.json",
            mime="application/json",
            disabled=not is_admin,
        )


# ----------------- Feedback Insights (read-only) -----------------
with tab_fb:
    st.subheader("Feedback Insights (read-only)")
    FEEDBACK_CSV = "feedback.csv"
    if not os.path.isfile(FEEDBACK_CSV):
        st.info("No feedback file found.")
    else:
        try:
            fb = pd.read_csv(FEEDBACK_CSV)
        except Exception as e:
            st.error(f"Could not read feedback.csv: {e}")
            fb = None

        if fb is not None:
            fb.columns = [c.lower() for c in fb.columns]
            if "timestamp" in fb.columns:
                with pd.option_context("mode.chained_assignment", None):
                    fb["timestamp"] = pd.to_datetime(
                        fb["timestamp"], errors="coerce"
                    )
                    fb["date"] = fb["timestamp"].dt.date
            else:
                fb["date"] = pd.NA

            total_fb = len(fb)
            if "email" in fb.columns:
                unique_users = fb["email"].nunique(dropna=True)
            elif "name" in fb.columns:
                unique_users = fb["name"].nunique(dropna=True)
            else:
                unique_users = 0

            if "timestamp" in fb.columns and fb["timestamp"].notna().any():
                last_fb_dt = fb["timestamp"].max().strftime("%Y-%m-%d %H:%M")
            else:
                last_fb_dt = "N/A"

            c1, c2, c3 = st.columns(3)
            c1.metric("Total feedback", total_fb)
            c2.metric(
                "Unique users", int(unique_users) if pd.notna(unique_users) else 0
            )
            c3.metric("Last entry", last_fb_dt)

            if "date" in fb.columns and "feedback" in fb.columns and fb["date"].notna().any():
                counts = (
                    fb.groupby("date")["feedback"]
                    .count()
                    .reset_index(name="count")
                    .sort_values("date")
                )
                if len(counts):
                    fig_cnt = px.bar(
                        counts, x="date", y="count", title="Feedback per day"
                    )
                    st.plotly_chart(fig_cnt, width="stretch")

            if "feedback" in fb.columns and fb["feedback"].notna().any():
                txt = " ".join(str(x) for x in fb["feedback"].dropna())
                tokens = re.findall(r"\b[\w\-']{3,}\b", txt.lower())
                stop = set(
                    """
                    the and for with this that are was were from into have has had not you your our out any
                    من على إلى في عن مع هذا هذه ذلك تلك هناك هنا لقد قد كان كانت يكون تكون يكونون التي الذي الذين
                    """.split()
                )
                tokens = [t for t in tokens if t not in stop]
                top_n = Counter(tokens).most_common(15)
                if top_n:
                    top_df = pd.DataFrame(top_n, columns=["term", "count"])
                    fig_top = px.bar(
                        top_df, x="term", y="count", title="Top feedback terms"
                    )
                    st.plotly_chart(fig_top, width="stretch")
                    st.dataframe(top_df, width="stretch")

            st.markdown("### Recent 5 entries")
            cols_to_show = [
                c for c in ["timestamp", "name", "email", "feedback"] if c in fb.columns
            ]
            if cols_to_show:
                st.dataframe(fb.tail(5)[cols_to_show], width="stretch")
            else:
                st.info("No standard feedback columns (timestamp/name/email/feedback) found.")


# ----------------- About & Methodology tab -----------------
with tab_about:
    st.subheader("About this Dashboard")
    st.markdown(
        """
        ### Purpose
        This clinical–actuarial dashboard is designed to:
        - Combine behavioral (BAS), clinical (CRS, CARS), provider (PCS, PPS), and financial (FEI) dimensions  
        - Produce a unified patient indicator (UPI) between 0–100  
        - Classify patients into **High / Medium / Low risk** based on configurable thresholds  
        - Support insurer/provider case-mix analysis and ICU/ward utilization review  

        ### Scoring Logic (high level)
        - **Normalization**  
          - MinMax (0–100), or  
          - Robust quantile rank (percentile-based) mapped to 0–100.
        - **Provider penalty**  
          - Base: `0.6 × (100 − PCS) + 0.4 × (100 − PPS_eff)`  
          - If readmission data is available: blended with a readmission-based penalty using a configurable weight.
        - **UPI**  
          - Linear combination of BAS, CRS, CARS, provider_penalty, and FEI with user-defined weights.
        - **Risk classes**  
          - High risk: `UPI ≥ High threshold`  
          - Medium risk: `Medium threshold ≤ UPI < High threshold`  
          - Low risk: otherwise.

        ### Governance & Usage
        - **Admin mode** (password-protected via `ADMIN_KEY` in secrets) controls:
          - Weights, thresholds, normalization choice  
          - File upload, download, and PDF generation  
        - **Read-only mode** (no or wrong admin key) is intended for:
          - Clinical review boards  
          - Financial / actuarial discussions  
          - Training and case discussion meetings  

        Concept and implementation maintained by **Mudather**.
        """
    )


# ----------------- Footer -----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#1f77b4; font-size:16px;'>Developed by <b>Mudather</b></p>",
    unsafe_allow_html=True,
)
st.caption("Model v1.2 • " + pd.Timestamp.today().strftime("%Y-%m-%d"))

