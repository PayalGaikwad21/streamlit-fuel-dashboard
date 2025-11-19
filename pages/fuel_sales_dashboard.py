
# ==========================
# FUEL SALES DASHBOARD (Enhanced + Gemini AI Auto-Detection)
# ==========================

import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai

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


# ---- SUPABASE CONNECTION ----
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

from supabase import create_client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==== Streamlit Setup ====
st.set_page_config(page_title="⛽ Fuel Sales Dashboard", layout="wide")
st.title("📊 Fuel Sales and Consumption Dashboard")
st.caption("Real transaction data from Bharat Petroleum SmartFleet — powered by Supabase.")

# ==== Fetch Data ====
@st.cache_data
def load_data():
    response = supabase.table("fuel_sales_transactions").select("*").execute()
    return pd.DataFrame(response.data)

try:
    df = load_data()
    st.success(f"✅ Loaded {len(df)} transactions successfully!")
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()





# ==========================
# 🔹 SEMANTIC ADAPTABILITY SECTION (FINAL STABLE + SAFE FALLBACKS)
# ==========================

import json
import google.generativeai as genai

# ==== Configure Gemini ====
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ==== Known column mapping dictionary ====
column_map = {
    "quantity_litres": ["litres", "fuel_qty", "total_litres", "fuel_quantity"],
    "total_amount_rs": ["amount", "fuel_cost", "total_price", "cost_rs", "price"],
    "vehicle_number": ["vehicle_no", "truck_id", "veh_number", "vehicleid"],
    "transaction_date": ["date", "fuel_date", "txn_date", "purchase_date"],
    "station_name": ["station", "pump", "pump_name", "fuel_station", "stationname"],
    "route_km": ["route", "distance_km", "route_distance"],
    "avg_mileage": ["avg_mileage_kmpl", "mileage", "fuel_efficiency"]
}

# ==== Manage Session State for Source ====
if "data_source" not in st.session_state:
    st.session_state.data_source = "supabase"

st.sidebar.header("🧩 Data Options")

# ==========================
# 🔀 DASHBOARD SWITCHER
# ==========================
if "current_dashboard" not in st.session_state:
    st.session_state.current_dashboard = "fuel"

st.sidebar.markdown("## 🗂 Dashboards")

dash_choice = st.sidebar.radio(
    "Select Dashboard",
    ["Fuel Dashboard", "Maintenance Dashboard"],
    index=0 if st.session_state.current_dashboard == "fuel" else 1
)

# Switch dashboard
if dash_choice == "Maintenance Dashboard":
    st.session_state.current_dashboard = "maintenance"
    st.switch_page("pages/maintenance_dashboard.py")

elif dash_choice == "Fuel Dashboard":
    st.session_state.current_dashboard = "fuel"



uploaded_file = st.sidebar.file_uploader("📂 Browse a new CSV file", type=["csv"])




# ==========================
# 🔹 Data Loading Logic
# ==========================
if uploaded_file:
    st.session_state.data_source = "csv"
    df_uploaded = pd.read_csv(uploaded_file)
    df_uploaded.columns = df_uploaded.columns.str.strip().str.lower()
    st.sidebar.success(f"✅ File uploaded with {len(df_uploaded)} rows")

    # ---- STEP A: Map columns from known dictionary ----
    rename_map = {}
    for std_col, aliases in column_map.items():
        for col in df_uploaded.columns:
            if col in aliases or std_col in col:
                rename_map[col] = std_col
    df_uploaded.rename(columns=rename_map, inplace=True)

    # ---- STEP B: Identify missing columns ----
    expected_cols = list(column_map.keys())
    missing_cols = [col for col in expected_cols if col not in df_uploaded.columns]

    if missing_cols:
        st.sidebar.warning(f"⚠️ Missing standard columns: {', '.join(missing_cols)} — trying Gemini AI...")

        try:
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            prompt = f"""
            You are a data analyst.
            The CSV columns are: {list(df_uploaded.columns)}.
            Expected fields are: {expected_cols}.
            Return a valid JSON mapping like {{"col_in_csv": "expected_col"}} for any columns that match logically.
            """
            response = model.generate_content(prompt)
            raw_text = response.text.strip()

            json_start = raw_text.find("{")
            json_end = raw_text.rfind("}")
            if json_start != -1 and json_end != -1:
                json_text = raw_text[json_start:json_end + 1]
                mapping_from_ai = json.loads(json_text)
                df_uploaded.rename(columns=mapping_from_ai, inplace=True)
                st.sidebar.success("✅ Gemini AI standardized column names successfully!")
            else:
                st.sidebar.warning("⚠️ Gemini AI returned invalid JSON, applying fallback mapping.")
        except Exception as e:
            st.sidebar.error(f"⚠️ Gemini AI mapping failed, fallback applied. Error: {e}")

        # Fallback fix — if still missing key column names
        if "station_name" not in df_uploaded.columns:
            pump_cols = [c for c in df_uploaded.columns if "pump" in c or "station" in c]
            if pump_cols:
                df_uploaded.rename(columns={pump_cols[0]: "station_name"}, inplace=True)
                st.sidebar.info(f"✅ Fallback mapped '{pump_cols[0]}' → 'station_name'")
        if "route_km" not in df_uploaded.columns:
            df_uploaded["route_km"] = df_uploaded["quantity_litres"] * 2
        if "avg_mileage" not in df_uploaded.columns:
            df_uploaded["avg_mileage"] = 8.5

    # ✅ Safe dataframe assignment
    df = df_uploaded.copy()
    st.sidebar.info("📊 Displaying dashboard using uploaded CSV data")

else:
    # Default Supabase data
    if st.session_state.data_source == "supabase":
        response = supabase.table("fuel_sales_transactions").select("*").limit(500).execute()
        df = pd.DataFrame(response.data)
        st.sidebar.info("📡 Using Supabase live data")








# ==== KPI Metrics ====
total_fuel = df["quantity_litres"].sum()
total_spent = df["total_amount_rs"].sum()
unique_vehicles = df["vehicle_number"].nunique()
avg_price_per_litre = round(total_spent / total_fuel, 2)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Fuel (L)", f"{total_fuel:,.2f}")
col2.metric("Total Spend (₹)", f"{total_spent:,.2f}")
col3.metric("Unique Vehicles", unique_vehicles)
col4.metric("Avg Price/Litre (₹)", avg_price_per_litre)

st.divider()


# ==========================
# SECTION 2: SALES ANALYTICS OVERVIEW
# ==========================
st.subheader("📊 Fuel Sales Overview")

tab1, tab2, tab3 = st.tabs(["🚛 Fuel by Vehicle", "📅 Daily Trend", "🛣️ Fuel Stations"])

with tab1:
    st.write("### 🚛 Total Spend and Fuel Consumption by Vehicle")
    fuel_by_vehicle = (
        df.groupby("vehicle_number")[["quantity_litres", "total_amount_rs"]]
        .sum().reset_index()
        .sort_values("total_amount_rs", ascending=False)
    )
    fig1 = px.bar(fuel_by_vehicle, x="vehicle_number", y="total_amount_rs",
                  title="Total Spend by Vehicle", hover_data=["quantity_litres"],
                  color="total_amount_rs", color_continuous_scale="Blues")
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.write("### 📅 Daily Fuel Purchase Trend")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    daily = df.groupby("transaction_date")[["quantity_litres", "total_amount_rs"]].sum().reset_index()
    fig2 = px.line(daily, x="transaction_date", y="total_amount_rs",
                   title="Daily Total Spend Trend", markers=True, color_discrete_sequence=["#FF6600"])
    fig2.update_traces(line=dict(width=3))
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.write("### 🛣️ Spend Distribution by Fuel Station")
    station_summary = (
        df.groupby("station_name")[["total_amount_rs"]]
        .sum().reset_index()
        .sort_values("total_amount_rs", ascending=False)
    )
    fig3 = px.pie(station_summary, values="total_amount_rs", names="station_name",
                  title="Spend Distribution by Station", hole=0.3,
                  color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ==========================
# VEHICLE LEVEL + P&L ANALYSIS
# ==========================
st.subheader("📈 Vehicle-Level Analysis")

df = df.copy()
if "transaction_date" in df.columns:
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

needed = {"route_km", "avg_mileage", "quantity_litres"}
missing_for_expected = needed - set(df.columns)
if not missing_for_expected:
    df["expected_fuel_liters"] = (df["route_km"] / df["avg_mileage"]).round(2)
    df["expected_fuel_liters"].replace([float("inf"), 0], pd.NA, inplace=True)
    df["expected_fuel_liters"].fillna(df["expected_fuel_liters"].mean(), inplace=True)
    df["variance_pct"] = ((df["quantity_litres"] - df["expected_fuel_liters"]) /
                          df["expected_fuel_liters"]).round(3) * 100

if "total_amount_rs" in df.columns and "quantity_litres" in df.columns and df["quantity_litres"].sum() > 0:
    global_avg_price_per_litre = df["total_amount_rs"].sum() / df["quantity_litres"].sum()
else:
    global_avg_price_per_litre = None

v_tab1, v_tab2 = st.tabs(["🚘 Vehicle-Level Analysis", "📉 P&L (Profit & Loss)"])


# ----------------------------------
# TAB 1 — Vehicle-Level Analysis
# ----------------------------------
with v_tab1:

    st.write("Select a vehicle and metrics to visualize (time series + pie summary).")
    vehicles = ["All"] + sorted(df["vehicle_number"].dropna().unique().tolist())
    sel_vehicle = st.selectbox("Choose vehicle", vehicles, index=0)

    possible_metrics = ["total_amount_rs", "quantity_litres", "variance_pct", "route_km", "avg_mileage"]
    sel_metrics = st.multiselect("Select metrics to plot (time series)", possible_metrics, default=["total_amount_rs"])

    df_v = df.copy()
    if sel_vehicle != "All":
        df_v = df_v[df_v["vehicle_number"] == sel_vehicle]

    # ----- Plot -----
    if not df_v.empty and "transaction_date" in df_v.columns:
        ts = df_v.sort_values("transaction_date")
        fig_ts = px.line(ts, x="transaction_date", y=sel_metrics,
                         title=f"Time Series for {sel_vehicle}", markers=True)
        st.plotly_chart(fig_ts, use_container_width=True)

    # ----- AI Summary FOR TAB 1 ONLY -----
    if "product_name" in df_v.columns:
        st.markdown("### 🧠 AI Summary for Fuel Spend")

        summary_df = df_v.groupby("product_name")["total_amount_rs"].sum().reset_index()

        prompt = f"""
        You are an expert fleet analyst. 
        Summarize the spending pattern for this vehicle/group.

        Data:
        {summary_df.to_dict(orient='records')}

        Write a simple, non-technical business summary.
        """

        try:
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            resp = model.generate_content(prompt)
            st.success(resp.text)
        except:
            st.warning("AI not available — showing manual summary instead.")
            for _, row in summary_df.iterrows():
                st.write(f"• **{row['product_name']}** → ₹{row['total_amount_rs']:,}")


# ----------------------------------
# TAB 2 — P&L Analysis
# ----------------------------------
with v_tab2:

    st.markdown("### 🧠 AI Summary — P&L (Profit & Loss)")

    # --- VEHICLE DROPDOWN FOR P&L ---
    vehicles_pnl = ["All"] + sorted(df["vehicle_number"].dropna().unique().tolist())
    sel_vehicle_pnl = st.selectbox("Choose vehicle for P&L", vehicles_pnl, index=0)

    # filter dataframe
    df_pnl = df.copy()
    if sel_vehicle_pnl != "All":
        df_pnl = df_pnl[df_pnl["vehicle_number"] == sel_vehicle_pnl]

    if df_pnl.empty:
        st.info("No data available for summary.")
    else:
        pnl_summary = df_pnl.groupby("transaction_date")["total_amount_rs"].sum().reset_index()

        prompt = f"""
        You are a transport business analyst.
        Create a simple and easy P&L summary for **{sel_vehicle_pnl}**.

        Data:
        {pnl_summary.to_dict(orient='records')}

        Explain:
        - Whether spending is rising or falling
        - Any spikes/dips
        - Monthly trend behavior
        - Final conclusion (simple language)
        """

        try:
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            resp = model.generate_content(prompt)
            st.success(resp.text)

        except:
            st.warning("AI not available — showing manual summary instead.")
            total = pnl_summary["total_amount_rs"].sum()
            st.write(f"• Total spending: ₹{total:,}")
            st.write(f"• Rows analysed: {len(pnl_summary)}")

st.divider()





# ----------------------------
# Hybrid LLM wrapper: Local (Ollama) <-> Gemini toggle
# ----------------------------
import os, json, subprocess
from typing import Optional
import diskcache
import google.generativeai as genai
from langchain_community.llms import Ollama
  # optional, if installed

# configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# small disk cache for computed aggregates (optional)
cache = diskcache.Cache("./.cache_hybrid")

# choose default model names
# choose default model names
GEMINI_MODEL = "models/gemini-2.5-flash"
LOCAL_OLLAMA_MODEL = "phi3:mini"


               # model name you pulled with ollama

# UI toggle in sidebar
st.sidebar.markdown("## 🧠 AI Mode")
mode = st.sidebar.radio("Choose LLM mode", ["Gemini (Cloud)", "Local (Ollama)"])

# Optional: show current selected model
st.sidebar.caption(f"Mode: {mode}")

def compute_derived(df_in: pd.DataFrame) -> pd.DataFrame:
    """Compute expected cost, profit, variance — use caching for speed."""
    # create a cache key based on rowcount + last tx date
    key = f"derived_{len(df_in)}_{df_in['transaction_date'].max() if 'transaction_date' in df_in.columns else 'no_date'}"
    if key in cache:
        return cache[key]
    dfc = df_in.copy()
    # safe conversions
    if "transaction_date" in dfc.columns:
        dfc["transaction_date"] = pd.to_datetime(dfc["transaction_date"], errors="coerce")
    # expected fuel
    if {"route_km", "avg_mileage"}.issubset(dfc.columns):
        dfc["expected_fuel_liters"] = (dfc["route_km"] / dfc["avg_mileage"]).round(2)
    else:
        # heuristic fallback: expected fuel ~ route_km / 8.5 or proportionate to quantity
        if "route_km" in dfc.columns:
            dfc["expected_fuel_liters"] = (dfc["route_km"] / 8.5).round(2)
        else:
            dfc["expected_fuel_liters"] = dfc.get("quantity_litres", 0) * 0.95

    # average price per litre
    qty_sum = dfc.get("quantity_litres", pd.Series([0])).sum()
    total_sum = dfc.get("total_amount_rs", pd.Series([0])).sum()
    avg_price = (total_sum / max(qty_sum, 1)) if qty_sum > 0 else 0
    dfc["expected_cost_rs"] = (dfc["expected_fuel_liters"] * avg_price).round(2)
    if "total_amount_rs" in dfc.columns:
        dfc["profit_rs"] = (dfc["expected_cost_rs"] - dfc["total_amount_rs"]).round(2)
    else:
        dfc["profit_rs"] = dfc["expected_cost_rs"] * 0.0

    cache.set(key, dfc, expire=300)  # cached 5 minutes
    return dfc

def call_gemini(prompt: str) -> str:
    model = genai.GenerativeModel(GEMINI_MODEL)
    resp = model.generate_content(prompt)
    return resp.text

def call_local_ollama_langchain(prompt: str, model_name: str = LOCAL_OLLAMA_MODEL):
    """Use only safe Ollama CLI call."""
    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=60
        )

        output = result.stdout.decode("utf-8", errors="ignore")
        error = result.stderr.decode("utf-8", errors="ignore")

        return output.strip() if output else error
    except Exception as e:
        return f"[Local LLM error] {e}"




def get_response(user_question: str, df_source: pd.DataFrame, mode_choice: str = "Gemini (Cloud)"):
    # 1) compute derived metrics locally
    dfc = compute_derived(df_source)

    # 2) prepare a concise context (sample + summary)
    sample_rows = dfc.head(25).to_dict(orient="records")   # keep small

    for row in sample_rows:
        for key, value in row.items():
            if isinstance(value, (pd.Timestamp, )):
                row[key] = str(value)

    summary = {
        "total_rows": int(len(dfc)),
        "total_spend": float(dfc["total_amount_rs"].sum()) if "total_amount_rs" in dfc.columns else 0.0,
        "total_fuel": float(dfc["quantity_litres"].sum()) if "quantity_litres" in dfc.columns else 0.0,
        "avg_price_per_litre": float((dfc["total_amount_rs"].sum() / max(dfc["quantity_litres"].sum(),1)) if "quantity_litres" in dfc.columns else 0.0)
    }

    # 3) construct prompt (clear and instructive)
    prompt = f"""
You are a professional data analyst for a transport company. Use the supplied summary + sample to answer precisely.
SUMMARY: {json.dumps(summary)}
COLUMNS: {list(dfc.columns)}
SAMPLE_ROWS (first 25): {json.dumps(sample_rows)}
INSTRUCTION: If user asks for profit or ranking, compute and use 'profit_rs' column (derived locally as expected_cost_rs - total_amount_rs).
User question: {user_question}
Respond in short, clear English. Provide top-k results if asked. If dates are mentioned, filter by transaction_date.
"""

    # 4) call appropriate model
    if mode_choice == "Gemini (Cloud)":
        return call_gemini(prompt)
    else:
        return call_local_ollama_langchain(prompt, model_name=LOCAL_OLLAMA_MODEL)

# Sidebar chatbox example using wrapper
st.sidebar.markdown("## 💬 Ask the hybrid AI")
q = st.sidebar.text_input("Ask about profit / trucks / dates")
if st.sidebar.button("Send (Hybrid)"):
    if not q:
        st.sidebar.warning("Type a question first.")
    else:
        with st.spinner("Analyzing..."):
            # fetch live supabase data for chat (but not necessary if df already is live)
            try:
                resp = supabase.table("fuel_sales_transactions").select("*").limit(1000).execute()
                df_live = pd.DataFrame(resp.data)
            except Exception:
                df_live = df.copy()

            answer = get_response(q, df_live, mode_choice=mode)
            st.sidebar.markdown(f"**AI ({mode}) Answer:**\n\n{answer}")



# ==========================
# SECTION 3: ADVANCED ANALYTICS
# ==========================

tab4, tab5, tab6 = st.tabs([
    "📊 Variance & Efficiency",
    "💰 Cost Distribution",
    "🧾 Trip-Level Analysis"
])

# ---------------- Tab 4: Variance & Efficiency ----------------
with tab4:
    st.subheader("⛽ Variance & Fuel Efficiency Analysis")

    required_cols = {"route_km", "avg_mileage", "quantity_litres"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        st.warning(f"⚠️ Missing columns in Supabase data: {', '.join(missing_cols)}")
    else:
        df["expected_fuel_liters"] = (df["route_km"] / df["avg_mileage"]).round(2)
        df["expected_fuel_liters"].replace([float("inf"), 0], pd.NA, inplace=True)
        df["expected_fuel_liters"].fillna(df["expected_fuel_liters"].mean(), inplace=True)

        df["variance_pct"] = (
            (df["quantity_litres"] - df["expected_fuel_liters"]) / df["expected_fuel_liters"]
        ).round(3) * 100

        # Histogram
        fig_var = px.histogram(
            df, x="variance_pct", nbins=40,
            title="Fuel Variance % Distribution",
            color_discrete_sequence=["#F97316"]
        )
        st.plotly_chart(fig_var, use_container_width=True)

        # KPI
        st.metric("Average Variance (%)", round(df["variance_pct"].mean(), 2))

        # Comparison grid
        st.dataframe(
            df[["vehicle_number","quantity_litres","expected_fuel_liters","variance_pct"]]
            .head(10),
            use_container_width=True
        )

    # -----------------------------
    # ✅ AI SUMMARY INSIDE TAB ONLY
    # -----------------------------
    if not df.empty and "variance_pct" in df.columns:
        st.markdown("### 🧠 AI Summary — Variance & Fuel Efficiency Insights")

        summary_data = {
            "avg_variance_pct": float(df["variance_pct"].mean()),
            "max_variance": float(df["variance_pct"].max()),
            "min_variance": float(df["variance_pct"].min()),
            "high_variance_count": int((df["variance_pct"] > 20).sum()),
            "low_variance_count": int((df["variance_pct"] < -20).sum()),
            "total_records": len(df)
        }

        prompt = f"""
        Provide a very simple 5–7 line summary for a transport owner.
        Data:
        {summary_data}
        """

        try:
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            resp = model.generate_content(prompt)
            st.success(resp.text)
        except:
            st.warning("AI not available — showing manual summary instead.")




# ---------------- Tab 5: Cost Distribution ----------------
with tab5:
    st.subheader("💰 Cost Distribution Insights")

    fig_station = px.bar(
        df.groupby("station_name")["total_amount_rs"].sum().reset_index(),
        x="station_name",
        y="total_amount_rs",
        title="Total Spend per Station"
    )
    st.plotly_chart(fig_station, use_container_width=True)

    # ---------------------------------------
    # ✅ AI SUMMARY (INSIDE TAB 5 ONLY)
    # ---------------------------------------
    if not df.empty and "total_amount_rs" in df.columns and "product_name" in df.columns:

        st.markdown("### 🧠 AI Summary — Cost Distribution Insights")

        summary_df = df.groupby("product_name")["total_amount_rs"].sum().reset_index()
        summary_dict = summary_df.to_dict(orient="records")

        prompt = f"""
        Write a simple and clear fuel-cost summary.
        Data:
        {summary_dict}
        """

        try:
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            resp = model.generate_content(prompt)
            st.success(resp.text)
        except:
            st.warning("AI unavailable — manual summary:")
            for _, row in summary_df.iterrows():
                st.write(f"• **{row['product_name']}** → ₹{row['total_amount_rs']:,}")




# ---------------- Tab 6: Trip-Level Detailed Data ----------------
with tab6:
    st.subheader("🧾 Vehicle / Trip-Level Data Explorer")

    vehicles = ["All"]
    if "vehicle_number" in df.columns:
        vehicles += sorted(df["vehicle_number"].dropna().unique().tolist())

    if "product_name" in df.columns:
        product_types = ["All"] + sorted(df["product_name"].dropna().unique().tolist())
    else:
        product_types = ["All"]

    selected_vehicle = st.selectbox("Select Vehicle", vehicles)
    selected_product = st.selectbox("Select Product", product_types)

    filtered_df = df.copy()

    if "vehicle_number" in df.columns and selected_vehicle != "All":
        filtered_df = filtered_df[filtered_df["vehicle_number"] == selected_vehicle]
    if "product_name" in df.columns and selected_product != "All":
        filtered_df = filtered_df[filtered_df["product_name"] == selected_product]

    st.dataframe(filtered_df, use_container_width=True)
        # ---------------------------------------
    # 🧠 AI Summary for Trip-Level Analysis
    # ---------------------------------------
    if not filtered_df.empty:

        st.markdown("### 🧠 AI Summary — Trip-Level Insights")

        # Prepare small useful aggregated data
        summary_trip = {
            "total_trips": len(filtered_df),
            "avg_spend": float(filtered_df["total_amount_rs"].mean()) if "total_amount_rs" in filtered_df else 0,
            "max_spend_trip": float(filtered_df["total_amount_rs"].max()) if "total_amount_rs" in filtered_df else 0,
            "min_spend_trip": float(filtered_df["total_amount_rs"].min()) if "total_amount_rs" in filtered_df else 0,
            "most_used_station": (
                filtered_df["station_name"].mode()[0]
                if "station_name" in filtered_df and not filtered_df["station_name"].isna().all()
                else "N/A"
            ),
            "top_vehicle": (
                filtered_df.groupby("vehicle_number")["total_amount_rs"].sum().idxmax()
                if "vehicle_number" in filtered_df and "total_amount_rs" in filtered_df
                else "N/A"
            )
        }

        prompt = f"""
        You are an expert transport trip analyst.
        Generate a very short, simple and clear summary (5–6 bullet points only)
        based on trip-level data.

        Data:
        {summary_trip}

        Explain ONLY:
        - Total number of trips
        - Average fuel spend per trip
        - Highest & lowest spend trip
        - Most used fuel station
        - Which vehicle is spending the most overall
        - One simple recommendation for the transport owner

        Keep the language extremely simple.
        """

        try:
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            resp = model.generate_content(prompt)
            st.success(resp.text)

        except:
            st.warning("AI unavailable — showing basic summary instead.")
            st.write(f"• Total trips analysed: {summary_trip['total_trips']}")
            st.write(f"• Average spend per trip: ₹{summary_trip['avg_spend']:.2f}")
            st.write(f"• Highest single-trip spend: ₹{summary_trip['max_spend_trip']:.2f}")
            st.write(f"• Lowest single-trip spend: ₹{summary_trip['min_spend_trip']:.2f}")
            st.write(f"• Most used station: {summary_trip['most_used_station']}")
            st.write(f"• Top spending vehicle: {summary_trip['top_vehicle']}")


    # ✅ Correct indentation for download button
    st.download_button(
        "📥 Download Filtered Data (CSV)",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_sales_data.csv",
        mime="text/csv"
    )



# ==========================
# INSIGHTS SUMMARY
# ==========================
st.divider()
st.subheader("🧠 Automated Insights")

avg_spent = round(df["total_amount_rs"].mean(), 2)
top_vehicle = df.groupby("vehicle_number")["total_amount_rs"].sum().idxmax()
top_spent = df.groupby("vehicle_number")["total_amount_rs"].sum().max()

st.write(f"🚛 Vehicle **{top_vehicle}** spent the most on fuel — ₹{top_spent:,.2f}.")
st.write(f"⚙️ Average transaction amount is ₹{avg_spent}.")
if "variance_pct" in df.columns:
    st.write(f"⛽ Overall variance trend: {round(df['variance_pct'].mean(), 2)}%.")


