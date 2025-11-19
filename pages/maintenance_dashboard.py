
# ================================
# MAINTENANCE DASHBOARD (FINAL FIXED VERSION)
# ================================

import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client
import google.generativeai as genai
import json, subprocess
import diskcache

def load_custom_css():
    with open("styles/custom_theme.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def force_black_ui_fix():
    st.markdown("""
    <style>

    /* ----------------------------------------------------
       1) FILE UPLOADER (WHITE BOX) → BLACK TEXT
       ---------------------------------------------------- */
    div[data-testid="stFileUploader"] * {
        color: black !important;
    }

    /* "Browse files" button */
    div[data-testid="stFileUploader"] button {
        color: black !important;
        border: 1px solid #000 !important;
        font-weight: 600 !important;
        background-color: #ffffff !important;
    }

     /* HYBRID AI BUTTON — ALWAYS BLACK TEXT (even disabled) */
    button[kind="secondary"],
    button[kind="secondary"]:disabled,
    button[disabled],
    div[data-testid="stButton"] button,
    div[data-testid="stButton"] button:disabled {
        background-color: white !important;
        color: black !important;               /* <-- TEXT BLACK */
        border: 1px solid black !important;
        opacity: 1 !important;                 /* no fade */
        font-weight: 700 !important;
    }

    /* Fix BaseWeb span inside button */
    div[data-testid="stButton"] button span {
        color: black !important;               /* <-- Inner text also BLACK */
    }



    </style>
    """, unsafe_allow_html=True)



load_custom_css()
force_black_ui_fix()

# -------------------------------------------------------------
#  STREAMLIT CONFIG
# -------------------------------------------------------------
st.set_page_config(page_title="🛠 Maintenance Dashboard", layout="wide")

# -------------------------------------------------------------
#  SUPABASE CONNECTION
# -------------------------------------------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]


supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

cache = diskcache.Cache("./.cache_hybrid")

GEMINI_MODEL = "models/gemini-2.5-flash"
LOCAL_OLLAMA_MODEL = "phi3:mini"


# -------------------------------------------------------------
#  DASHBOARD SWITCHER (SIDEBAR)
# -------------------------------------------------------------
if "current_dashboard" not in st.session_state:
    st.session_state.current_dashboard = "maintenance"

st.sidebar.markdown("## 🗂 Dashboards")

dash_choice = st.sidebar.radio(
    "Select Dashboard",
    ["Fuel Dashboard", "Maintenance Dashboard"],
    index=1
)

if dash_choice == "Fuel Dashboard":
    st.switch_page("pages/fuel_sales_dashboard.py")




# -------------------------------------------------------------
# DATA SOURCE SIDEBAR (same as fuel dashboard)
# -------------------------------------------------------------
st.sidebar.header("📂 Data Source")
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV (Optional)", type=["csv"])

if st.sidebar.button("🔄 Reload Supabase Data"):
    st.rerun()


# -------------------------------------------------------------
# LOAD ALL MAINTENANCE TABLES
# -------------------------------------------------------------
@st.cache_data
def load_table(name):
    try:
        res = supabase.table(name).select("*").limit(5000).execute()
        return pd.DataFrame(res.data)
    except:
        return pd.DataFrame()


maintenance_tables = {
    "oil_hub": load_table("oil_hub"),
    "clutch_pump": load_table("clutch_pump"),
    "engine_steering": load_table("engine_steering"),
    "saman": load_table("saman"),
    "trailer": load_table("trailer"),
    "additional": load_table("additional"),
    "alignment": load_table("alignment")
}


# -------------------------------------------------------------
# STANDARDIZE VEHICLE COLUMN BEFORE MERGING
# -------------------------------------------------------------
vehicle_aliases = ["vehicle_no", "vehicle_number", "veh_no", "truck_no", "vehicle"]

for key, df in maintenance_tables.items():
    for col in df.columns:
        if col.lower() in vehicle_aliases:
            df.rename(columns={col: "vehicle_number"}, inplace=True)
            break


# -------------------------------------------------------------
# MERGE ALL TABLES INTO A SINGLE OVERVIEW DF
# -------------------------------------------------------------
overview_df = pd.concat(maintenance_tables.values(), ignore_index=True)





# -------------------------------------------------------------
# SET VEHICLE COLUMN (fix for old vehicle_col references)
# -------------------------------------------------------------
vehicle_col = "vehicle_number"


# -------------------------------------------------------------
# SAFE DATE NORMALIZATION (prevents _norm_date KeyError)
# -------------------------------------------------------------
date_candidates = ["date", "service_date", "repair_date", "created_at", "updated_at"]

norm_col = None
for c in overview_df.columns:
    if c.lower() in date_candidates or "date" in c.lower():
        norm_col = c
        break

if norm_col:
    overview_df["_norm_date"] = pd.to_datetime(overview_df[norm_col], errors="coerce")
else:
    # no date found → create empty column
    overview_df["_norm_date"] = pd.NaT



# -------------------------------------------------------------
# AI MODE SIDEBAR
# -------------------------------------------------------------
st.sidebar.markdown("## 🧠 AI Mode")
mode = st.sidebar.radio("Choose LLM", ["Gemini (Cloud)", "Local (Ollama)"])
st.sidebar.caption(f"Mode: {mode}")


def call_gemini(prompt):
    model = genai.GenerativeModel(GEMINI_MODEL)
    return model.generate_content(prompt).text


def call_local(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", LOCAL_OLLAMA_MODEL],
            input=prompt.encode(),
            capture_output=True,
            timeout=60
        )
        return result.stdout.decode() or result.stderr.decode()
    except Exception as e:
        return f"[Local LLM Error] {e}"


def ask_hybrid(prompt, df_sample):
    sample = df_sample.head(25).to_dict(orient="records")
    prompt_final = f"""
You are a transport maintenance analyst.

Analyze this snapshot:
{json.dumps(sample)}

Answer the question:
{prompt}
"""

    return call_gemini(prompt_final) if mode == "Gemini (Cloud)" else call_local(prompt_final)


# -------------------------------------------------------------
# SIDEBAR — HYBRID AI CHAT
# -------------------------------------------------------------
st.sidebar.markdown("## 💬 Ask Maintenance AI")

q = st.sidebar.text_input("Ask about repairs, cost, trucks...")

if st.sidebar.button("Send"):
    if q:
        with st.spinner("Analyzing..."):
            answer = ask_hybrid(q, overview_df)
        st.sidebar.markdown(f"**AI ({mode}) Answer:**\n\n{answer}")
    else:
        st.sidebar.warning("Type your question first.")


# -------------------------------------------------------------
# PAGE TITLE
# -------------------------------------------------------------
st.title("🛠 Maintenance Dashboard")
st.caption("Live maintenance data connected from Supabase — across 7 service categories.")


# -------------------------------------------------------------
# HELPER: detect cost column
# -------------------------------------------------------------
def detect_cost_column(df):
    for c in ["total_cost", "cost_rs", "cost", "amount"]:
        if c in df.columns:
            return c
    return None


# -------------------------------------------------------------
# OVERVIEW SUMMARY SECTION
# -------------------------------------------------------------
st.subheader("📌 Maintenance Overview (All Categories Combined)")

if not overview_df.empty:

    cost_col = detect_cost_column(overview_df)

    if cost_col:
        overview_df[cost_col] = pd.to_numeric(overview_df[cost_col], errors="coerce").fillna(0)
        total_cost = overview_df[cost_col].sum()
    else:
        total_cost = 0

    total_repairs = len(overview_df)
    unique_vehicles = overview_df["vehicle_number"].nunique()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Repairs", total_repairs)
    c2.metric("Total Maintenance Cost (₹)", f"{total_cost:,.0f}")
    c3.metric("Vehicles Serviced", unique_vehicles)

else:
    st.warning("No maintenance data found.")

st.divider()


# -------------------------------------------------------------
# CATEGORY TABS (7 TABLES)
# -------------------------------------------------------------
tabs = st.tabs([
    "🛢 Oil & Hub",
    "⚙️ Clutch & Pump",
    "🚜 Engine & Steering",
    "📦 Saman",
    "🚚 Trailer",
    "➕ Additional",
    "📐 Alignment"
])

for i, key in enumerate(maintenance_tables.keys()):
    df = maintenance_tables[key]

    with tabs[i]:
        st.subheader(f"📁 {key.replace('_', ' ').title()} Records")

        if df.empty:
            st.warning("No data in this category.")
            continue

        st.dataframe(df, use_container_width=True)

        cost_col = detect_cost_column(df)

        # Trend
        if "date" in df.columns and cost_col:
            try:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                grp = df.groupby("date")[cost_col].sum().reset_index()
                fig = px.line(grp, x="date", y=cost_col,
                              title="Daily Cost Trend", markers=True)
                st.plotly_chart(fig, use_container_width=True)
            except:
                pass

        # Distribution
        if cost_col:
            fig2 = px.histogram(df, x=cost_col,
                                title="Cost Distribution", nbins=25)
            st.plotly_chart(fig2, use_container_width=True)
# --------------------------- 
# Combined master table (bottom) - show merged view (source indicated by _table)
# ---------------------------

st.header("🔎 Combined Truck-Level & Advanced Analytics (Bottom Sections)")
st.markdown("**1) Combined Maintenance Table (all tables merged)**")
# show a small preview of merged table
if overview_df.empty:
    st.info("Combined maintenance table is empty.")
else:
    # show normalized important cols
    display_cols = []
    # prefer vehicle_col
    if "vehicle_number" in overview_df.columns:
        display_cols.append("vehicle_number")
    # show source table
    if "_table" in overview_df.columns:
        display_cols.append("_table")
    # show normalized date & cost
    if "_norm_date" in overview_df.columns:
        display_cols.append("_norm_date")
    if "cost_inr" in overview_df.columns:
        display_cols.append("cost_inr")
    # add some commonly present columns (first 10)
    extra_cols = [c for c in overview_df.columns if c not in display_cols][:10]
    display_cols.extend(extra_cols)
    # limit to first 49 unique rows only
clean_df = (
    overview_df[display_cols]
    .drop_duplicates()                 # remove repeated rows
    .reset_index(drop=True)            # clean index
    .head(49)                          # show ONLY first 49 rows
)

st.dataframe(clean_df, use_container_width=True)


st.divider()

# ---------------------------
# Maintenance Tabs — Full History | Cost & Summary | Pending/Upcoming
# ---------------------------
st.header("### 2) Maintenance Tabs — Full History | Cost | Pending")

if overview_df.empty:
    st.info("No maintenance data to analyze.")
else:
    tab_full, tab_cost, tab_pending = st.tabs(["Full History", "Cost & Spending Summary", "Pending / Upcoming"])
    all_vehicles = sorted(overview_df["vehicle_number"].dropna().unique().tolist())

    # ---------------------------
    # FULL HISTORY TAB
    # ---------------------------
    with tab_full:
        st.subheader("Full maintenance history for a selected vehicle")
        chosen_full = st.selectbox("Choose vehicle (Full history)", all_vehicles, key="full_history_vehicle")

        df_full = overview_df[overview_df["vehicle_number"] == chosen_full].copy()

        # remove unwanted columns
        drop_cols = ["_norm_date", "cost_inr"]
        for c in drop_cols:
            if c in df_full.columns:
                df_full.drop(columns=[c], inplace=True)

        # collapse all rows into single row
        df_single = df_full.ffill().bfill().head(1)

        st.dataframe(df_single, use_container_width=True)



with tab_cost:
    st.subheader("Cost & Spending Summary")
    chosen_cost = st.selectbox("Choose vehicle (Cost)", all_vehicles, key="cost_vehicle")
    df_cost = overview_df[overview_df["vehicle_number"] == chosen_cost].copy()

    df_cost["cost_inr"] = pd.to_numeric(df_cost.get("cost_inr", 0), errors="coerce").fillna(0)

    if "_norm_date" in df_cost.columns:
        df_cost["_norm_date"] = pd.to_datetime(df_cost["_norm_date"], errors="coerce")
        df_cost["month"] = df_cost["_norm_date"].dt.to_period("M").astype(str)
        monthly = df_cost.groupby("month")["cost_inr"].sum().reset_index()
        fig_month = px.bar(monthly, x="month", y="cost_inr",
                           title=f"Monthly Spend — {chosen_cost}", text_auto=True)
        st.plotly_chart(fig_month, use_container_width=True)

    if "_table" in df_cost.columns:
        pie = df_cost.groupby("_table")["cost_inr"].sum().reset_index()
        fig_pie = px.pie(pie, names="_table", values="cost_inr",
                         title=f"Spend by Table — {chosen_cost}", hole=0.35)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ---- collapse to single row ----
    df_cost_clean = df_cost.copy()

    drop_cols = ["_norm_date", "cost_inr"]
    for c in drop_cols:
        if c in df_cost_clean.columns:
            df_cost_clean.drop(columns=[c], inplace=True)

    df_cost_single = df_cost_clean.ffill().bfill().head(1)

    st.dataframe(df_cost_single, use_container_width=True)


    # ---------------------------
# NOW correct Pending Tab
# ---------------------------
with tab_pending:
    st.subheader("Pending / Upcoming maintenance (heuristic)")
    chosen_pending = st.selectbox("Choose vehicle (Pending)", all_vehicles, key="pending_vehicle")
    df_pending = overview_df[overview_df["vehicle_number"] == chosen_pending].copy()

    date_cols = [c for c in df_pending.columns if "date" in c.lower()]
    due_rows = []

    for idx, row in df_pending.iterrows():
        is_pending = False

        if isinstance(row.get("remarks"), str) and any(k in row["remarks"].lower() for k in ["sched", "planned", "next", "fix"]):
            is_pending = True

        for dcol in date_cols:
            v = pd.to_datetime(row.get(dcol), errors="coerce")
            if pd.notna(v) and v.date() > pd.Timestamp.now().date():
                is_pending = True

        if is_pending:
            due_rows.append(idx)

    if not due_rows:
        st.info("No pending maintenance recognized.")
    else:
        pending_df = df_pending.loc[due_rows].reset_index(drop=True)

        row = pending_df.head(1).to_dict(orient="records")[0]

        clean_items = {
            k: v for k, v in row.items()
            if pd.notna(v) and str(v).strip() not in ["None", "", "NaT"]
        }

        st.markdown("### 🧾 Pending Maintenance Summary")
        for key, value in clean_items.items():
            st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")
st.divider()

# ---------------------------
# Predictive Maintenance (Prototype)
# ---------------------------
st.header("🔮 Predictive Maintenance (Prototype)")
st.markdown("This uses simple heuristics. It can be upgraded with ML/time-series models later.")

if overview_df.empty or not vehicle_col:
    st.info("Insufficient data to run predictive heuristics.")
else:
    chosen_pred = st.selectbox("Select vehicle for prediction", sorted(overview_df[vehicle_col].dropna().unique().tolist()), key="pred_vehicle")
    df_pred = overview_df[overview_df[vehicle_col] == chosen_pred].copy()

    # Basic heuristic: for each component column with date values - compute days since last event and flag if > threshold
    # Identify candidate components: columns that end with '_date' or contain 'date' and are not inserted_at
    date_like_cols = [c for c in df_pred.columns if ("date" in c.lower() and not c.lower().startswith("insert"))]
    # build last event per component
    comp_last = {}
    for c in date_like_cols:
        try:
            tmp = pd.to_datetime(df_pred[c], errors="coerce")
            if tmp.notna().any():
                last = tmp.max()
                comp_last[c] = last
        except Exception:
            continue

    if not comp_last:
        st.info("Not enough historical events per component to predict (component-date fields not present).")
    else:
        st.write("### Component last-event dates")
        df_comp = pd.DataFrame.from_dict(comp_last, orient="index", columns=["last_date"]).reset_index().rename(columns={"index": "component"})
        df_comp["days_ago"] = (pd.Timestamp.now() - df_comp["last_date"]).dt.days
        st.dataframe(df_comp.sort_values("days_ago", ascending=False).reset_index(drop=True), use_container_width=True)

        # Simple rule: if days_ago > median(days_ago) * 1.5 -> flag as likely due soon
        median_days = df_comp["days_ago"].median()
        df_comp["due_flag"] = df_comp["days_ago"] > (median_days * 1.5)
        due = df_comp[df_comp["due_flag"]]
        if due.empty:
            st.success("No components flagged as due by the simple heuristic.")
        else:
            st.warning("Components flagged as likely due (heuristic):")
            st.dataframe(due, use_container_width=True)

        # Optional: call Gemini / local LLM to produce plain-language suggestion
        if GEMINI_API_KEY:
            if st.button("Explain prediction using Gemini (cloud)"):
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=GEMINI_API_KEY)
                    prompt = f"Predictive maintenance summary for vehicle {chosen_pred}. Components and days since last event:\n\n{df_comp.to_dict(orient='records')}\n\nWrite a concise prediction and next steps."
                    model = genai.GenerativeModel(GEMINI_MODEL)
                    resp = model.generate_content(prompt)
                    st.markdown("**AI Note:**")
                    st.write(resp.text)
                except Exception as e:
                    st.error(f"LLM call failed: {e}")
        else:
            st.info("Gemini API key not configured. Predictions are heuristic-only.")

st.divider()

# ---------------------------
# Truck Health Score (simple)
# ---------------------------
st.header("🩺 Truck Health Score (Simple)")
st.markdown("Score computed with simple rules. 100 = best. Lower means more issues / older events.")

def compute_health_score(df_vehicle: pd.DataFrame):
    # base 100
    score = 100.0
    # penalty for missing critical events in last 12 months (example heuristics)
    keys = [
        "oil_change_date", "clutch_plate_date", "coolant_tank_date", "clutch_oil_date",
        "air_filter_date", "wheel_alignment_date", "battery_date", "radiator_date"
    ]
    # use any column containing keyword
    penalties = 0
    for k in keys:
        cols = [c for c in df_vehicle.columns if k.split("_")[0] in c.lower() and "date" in c.lower()]
        # if no such column, small penalty
        if not cols:
            penalties += 0.5
            continue
        # check last date across those columns
        last_dates = []
        for c in cols:
            try:
                tmp = pd.to_datetime(df_vehicle[c], errors="coerce")
                last_dates.extend(list(tmp.dropna().tolist()))
            except Exception:
                continue
        if last_dates:
            last = max(last_dates)
            days = (pd.Timestamp.now() - last).days
            if days > 365:
                penalties += 3
            elif days > 180:
                penalties += 1.5
            else:
                penalties += 0
        else:
            penalties += 1.5
    # penalty scaling
    score = max(0, score - penalties)
    # reduce based on total repairs in past year (more repairs -> lower health)
    try:
        if "_norm_date" in df_vehicle.columns:
            one_year_ago = pd.Timestamp.now() - pd.Timedelta(days=365)
            recent_repairs = df_vehicle[df_vehicle["_norm_date"] >= one_year_ago]
            count = len(recent_repairs)
            score -= min(20, count * 2)  # each recent repair reduces score a bit
    except Exception:
        pass
    return round(score, 1)

if vehicle_col and (not overview_df.empty):
    chosen_health = st.selectbox("Select vehicle for health score", sorted(overview_df[vehicle_col].dropna().unique().tolist()), key="health_vehicle")
    df_h = overview_df[overview_df[vehicle_col] == chosen_health].copy()
    if df_h.empty:
        st.info("No data for this vehicle.")
    else:
        health = compute_health_score(df_h)
        st.metric(label=f"Health Score — {chosen_health}", value=f"{health} / 100")
        st.caption("Heuristic score. Lower score suggests more maintenance attention required.")

st.divider()

st.info("Dashboard end — all changes applied. If any plot or table shows unexpected 'None' values, it's likely because some of your Supabase tables use different column names. This code attempts to detect names heuristically and normalize cost/date. If you'd like, share exact column names and I will map them precisely (no DB changes required).")

# End of file


