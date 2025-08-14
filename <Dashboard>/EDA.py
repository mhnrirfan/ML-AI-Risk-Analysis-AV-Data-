# EDA.py — Hybrid Plotly + Matplotlib Streamlit dashboard
# Run with:  streamlit run EDA.py
# If you run with plain python you’ll see warnings; Streamlit features require `streamlit run`.

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

# to run  streamlit run "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Dashboard>/EDA.py"
# Optional deps (used in UK choropleth fallback)
try:
    import geopandas as gpd
    from shapely.geometry import Point
    _HAS_GEOPANDAS = True
except Exception:
    _HAS_GEOPANDAS = False

# -----------------------------
# Page / Styling
# -----------------------------
st.set_page_config(page_title="AV Incident EDA Dashboard", layout="wide")
st.title("🚗 Autonomous Vehicle Incident EDA Dashboard")
st.caption("Hybrid: Plotly where possible, Matplotlib/GeoPandas fallback where needed.")

# -----------------------------
# Sidebar: Data Sources & Options
# -----------------------------
with st.sidebar:
    st.header("Data Sources")
    st.markdown("Provide paths to your cleaned CSV files. You can also use the default paths.")

    default_uk = "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK-cleaned_data.csv"
    default_us = "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/US-cleaned_data.csv"

    uk_path = st.text_input("UK CSV path", value=default_uk)
    us_path = st.text_input("US CSV path", value=default_us)

    st.divider()
    st.header("Geospatial (UK)")
    st.markdown("Path to the **UK LAD** dataset folder (shapefile or GeoPackage).\n\nFor example, a directory that contains *.shp, *.dbf, *.shx, *.prj*, or a single *.geojson*/*gpkg* file.")
    default_uk_lads = "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/LAD_MAY_2025_UK_BFC_2360005762104150824"
    uk_lads_path = st.text_input("UK LAD path (dir/file)", value=default_uk_lads)

    st.divider()
    region = st.radio("Active Region", ["UK", "US"], index=0)
    top_k = st.slider("Top K categories/cities to display", 5, 50, 20, 1)

# -----------------------------
# Utilities
# -----------------------------
NUM_FMT = "{:,}"

def safe_exists(path: str) -> bool:
    if not path:
        return False
    if os.path.isdir(path) or os.path.isfile(path):
        return True
    return False

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    if not safe_exists(path):
        raise FileNotFoundError(f"File or directory not found: {path}")
    # Mixed-type ID columns: keep as string to avoid DtypeWarning
    try:
        df = pd.read_csv(
            path,
            low_memory=False,
            dtype={"Report ID": "string", "Report Version": "string"},
        )
    except Exception:
        # Fallback without dtype hints
        df = pd.read_csv(path, low_memory=False)
    return df

@st.cache_data(show_spinner=False)
def prep_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Parse incident date
    if "Incident Date" in df.columns:
        df["Incident Date"] = pd.to_datetime(df["Incident Date"], errors="coerce")
        df["Year"] = df["Incident Date"].dt.year
        df["Month"] = df["Incident Date"].dt.month
        df["Weekday"] = df["Incident Date"].dt.day_name()
    # Parse time → hour
    time_col = "Incident Time (24:00)"
    if time_col in df.columns:
        # Handle formats like "01:15:00", "1:15", integers, etc.
        # Try to_datetime first; fallback to extracting hour via string split
        s = df[time_col]
        # If already datetime-like, keep hour; else coerce
        if not pd.api.types.is_datetime64_any_dtype(s):
            # Try to parse as datetime (no date)
            try:
                parsed = pd.to_datetime(s, errors="coerce").dt.hour
            except Exception:
                parsed = pd.Series([pd.NA] * len(s), index=s.index)
            # Where parsing failed, try to extract first token before ':'
            needs_fill = parsed.isna()
            if needs_fill.any():
                # Convert to string and split
                tmp = s.astype(str).str.strip()
                # Replace possible artifacts
                tmp = tmp.str.replace("\\.", ":", regex=True)
                # Extract hour safely
                hh = tmp.str.extract(r"^(\d{1,2})")[0]
                hh = pd.to_numeric(hh, errors="coerce")
                parsed = parsed.fillna(hh)
            df["Incident Hour"] = pd.to_numeric(parsed, errors="coerce")
        else:
            df["Incident Hour"] = s.dt.hour
    return df

@st.cache_data(show_spinner=False)
def split_dtypes(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number", "datetime"]).columns.tolist()
    dt_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
    # Ensure common important cols appear if present
    for c in ["Highest Injury Severity Alleged", "State", "City", "SV Contact Area", "Model Year"]:
        if c in df.columns and c not in cat_cols and c not in num_cols:
            cat_cols.append(c)
    return num_cols, cat_cols, dt_cols

# -----------------------------
# Load data
# -----------------------------
load_error = None
try:
    UK_data = load_csv(uk_path)
    UK_data = prep_dates(UK_data)
except Exception as e:
    UK_data = pd.DataFrame()
    load_error = f"UK: {e}"

try:
    US_data = load_csv(us_path)
    US_data = prep_dates(US_data)
except Exception as e:
    US_data = pd.DataFrame()
    load_error = (load_error + "\n" if load_error else "") + f"US: {e}"

if load_error:
    st.warning(load_error)

# Active df
df = UK_data if region == "UK" else US_data

if df.empty:
    st.stop()

num_cols, cat_cols, dt_cols = split_dtypes(df)

# -----------------------------
# Helper plot functions
# -----------------------------

def fig_missingness(data: pd.DataFrame):
    missing = data.isnull().mean().mul(100).reset_index()
    missing.columns = ["Feature", "Missing %"]
    missing = missing.sort_values("Missing %", ascending=False)
    fig = px.bar(missing, x="Feature", y="Missing %", title="Missingness per Column (%)")
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def fig_missing_vs_severity(data: pd.DataFrame, severity_col: str):
    # Build long-form counts of missingness vs severity for columns with NA
    cols_with_na = [c for c in data.columns if data[c].isna().any()]
    records = []
    for c in cols_with_na:
        miss = data[c].isna().astype(int)
        grp = (
            data.assign(_miss=miss)
            .groupby(["_miss", severity_col], dropna=False)
            .size()
            .reset_index(name="Count")
        )
        grp["Feature"] = c
        records.append(grp)
    if not records:
        return px.bar(pd.DataFrame({"Feature": [], "_miss": [], "Count": [], severity_col: []}))
    out = pd.concat(records, ignore_index=True)
    out["Missing"] = out["_miss"].map({0: "Present", 1: "Missing"})
    fig = px.bar(
        out,
        x="Feature",
        y="Count",
        color=severity_col,
        facet_col="Missing",
        barmode="stack",
        category_orders={"Missing": ["Present", "Missing"]},
        title="Missingness vs. Severity (stacked)"
    )
    fig.update_layout(xaxis_tickangle=-45, showlegend=True)
    return fig



def fig_boxplot(data: pd.DataFrame, column: str):
    return px.box(data, y=column, points="all", title=f"Boxplot: {column}")


def fig_hist_density_compare(uk_df: pd.DataFrame, us_df: pd.DataFrame, column: str):
    # Overlay density via normalized histogram
    uk = uk_df[[column]].assign(Region="UK").dropna()
    us = us_df[[column]].assign(Region="US").dropna()
    comb = pd.concat([uk, us], ignore_index=True)
    fig = px.histogram(
        comb, x=column, color="Region", histnorm="probability density", barmode="overlay",
        opacity=0.45, title=f"Distribution (normalized): {column} — UK vs US"
    )
    return fig


def fig_topk_bar(data: pd.DataFrame, column: str, k: int):
    counts = data[column].value_counts(dropna=False).head(k).reset_index()
    counts.columns = [column, "Count"]
    fig = px.bar(counts, x=column, y="Count", title=f"Top {k}: {column}")
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def explode_contact_area_for_counts(data: pd.DataFrame, col: str, k: int) -> pd.DataFrame:
    ser = data[col].dropna().astype(str)
    areas = ser.str.split(",").explode().str.strip()
    vc = areas.value_counts().head(k).reset_index()
    vc.columns = [col, "Count"]
    return vc


def fig_severity_stack(data: pd.DataFrame, cat_col: str, severity_col: str, k: int):
    # top-k categories by total counts
    totals = data[cat_col].value_counts().head(k).index
    sub = data[data[cat_col].isin(totals)]
    grouped = sub.groupby([cat_col, severity_col], dropna=False).size().reset_index(name="Count")
    fig = px.bar(grouped, x=cat_col, y="Count", color=severity_col, barmode="stack",
                 title=f"Severity by {cat_col} (Top {k})")
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def fig_us_state_choropleth(us_df: pd.DataFrame):
    if "State" not in us_df.columns:
        return px.bar(pd.DataFrame({"State": [], "Count": []}), x="State", y="Count")
    state_counts = us_df["State"].astype(str).str.strip().str.upper().value_counts().reset_index()
    state_counts.columns = ["State", "Count"]
    total = int(state_counts["Count"].sum()) if not state_counts.empty else 0
    title = f"US Incidents by State — Total: {NUM_FMT.format(total)}"
    fig = px.choropleth(
        state_counts,
        locations="State",
        locationmode="USA-states",
        color="Count",
        scope="usa",
        title=title
    )
    return fig


def fig_ca_top_cities(us_df: pd.DataFrame, k: int):
    if "State" not in us_df.columns or "City" not in us_df.columns:
        return px.bar(pd.DataFrame({"City": [], "Count": []}), x="City", y="Count")
    ca = us_df[us_df["State"].astype(str).str.strip().str.upper() == "CA"]
    city_counts = ca["City"].value_counts().head(k).reset_index()
    city_counts.columns = ["City", "Count"]
    fig = px.bar(city_counts, x="City", y="Count", title=f"Top {k} California Cities by Incident Count")
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def fig_radial_time_matplotlib(data: pd.DataFrame):
    # Matplotlib fallback for radial time plot
    if "Incident Hour" not in data.columns:
        return None
    counts = (
        data["Incident Hour"].dropna().astype(int).clip(0, 23).value_counts().reindex(range(24), fill_value=0).sort_index()
    )
    theta = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    width = 2 * np.pi / 24
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.bar(theta, counts.values, width=width, edgecolor="black")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(theta)
    ax.set_xticklabels([f"{h}:00" for h in range(24)])
    ax.set_title("Incident Time Distribution (Radial)")
    fig.tight_layout()
    return fig


def fig_uk_choropleth_matplotlib(uk_df: pd.DataFrame, uk_lads_path: str):
    if not _HAS_GEOPANDAS:
        st.error("GeoPandas/Shapely not available — cannot render UK choropleth fallback.")
        return None
    if not safe_exists(uk_lads_path):
        st.error("UK LAD path not found.")
        return None
    if not {"latitude", "longitude"}.issubset(uk_df.columns):
        st.error("UK data must have 'latitude' and 'longitude' columns.")
        return None

    # Load LADs (dir with shapefile components or a single file)
    try:
        if os.path.isdir(uk_lads_path):
            # Find a file inside the directory to read (e.g., .shp, .geojson, .gpkg)
            candidates = [
                f for f in os.listdir(uk_lads_path)
                if f.lower().endswith((".shp", ".geojson", ".gpkg"))
            ]
            if not candidates:
                st.error("No .shp/.geojson/.gpkg found in the LAD directory.")
                return None
            lad_file = os.path.join(uk_lads_path, candidates[0])
        else:
            lad_file = uk_lads_path
        uk_lads = gpd.read_file(lad_file)
    except Exception as e:
        st.error(f"Failed to read LAD data: {e}")
        return None

    # Build points GeoDataFrame
    try:
        geometry = [Point(xy) for xy in zip(uk_df["longitude"], uk_df["latitude"])]
        gdf_points = gpd.GeoDataFrame(uk_df.copy(), geometry=geometry, crs="EPSG:4326")
        # Reproject to match uk_lads
        if uk_lads.crs is None:
            # Assume WGS84 if missing
            uk_lads = uk_lads.set_crs("EPSG:4326")
        gdf_points = gdf_points.to_crs(uk_lads.crs)
        # Spatial join
        joined = gpd.sjoin(gdf_points, uk_lads, how="left", predicate="within")
        # Try to find a LAD name column
        lad_name_col = None
        for cand in ["LAD25NM", "LAD23NM", "LAD22NM", "LAD19NM", "LAD_NAME", "lad_name"]:
            if cand in joined.columns:
                lad_name_col = cand
                break
        if lad_name_col is None:
            # If no standard column, look for any object dtype column from uk_lads
            possible = [c for c in uk_lads.columns if uk_lads[c].dtype == object]
            lad_name_col = possible[0] if possible else uk_lads.columns[0]
        counts = joined.groupby(lad_name_col).size().reset_index(name="incident_count")
        choropleth_gdf = uk_lads.merge(counts, how="left", left_on=lad_name_col, right_on=lad_name_col)
        choropleth_gdf["incident_count"] = choropleth_gdf["incident_count"].fillna(0)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 12))
        choropleth_gdf.plot(
            column="incident_count",
            cmap="Blues",
            linewidth=0.6,
            ax=ax,
            edgecolor="lightgrey",
            legend=True,
        )
        total = int(choropleth_gdf["incident_count"].sum())
        max_count = int(choropleth_gdf["incident_count"].max())
        title = f"UK Incidents by Local Authority\nTotal: {NUM_FMT.format(total)} | Peak: {NUM_FMT.format(max_count)}"
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Failed to render UK choropleth: {e}")
        return None

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs([
    "📉 Missingness",
    "🔢 Numerical",
    "🔤 Categorical",
    "🗓️ Datetime",
    "🗺️ Geospatial",
    "🧱 Severity",
])

# 1) Missingness
with tabs[0]:
    st.subheader("Missing Data Overview")
    st.plotly_chart(fig_missingness(df), use_container_width=True)

    # Missingness vs Severity (if severity col exists)
    sev_col = "Highest Injury Severity Alleged"
    if sev_col in df.columns:
        st.plotly_chart(fig_missing_vs_severity(df, sev_col), use_container_width=True)
    else:
        st.info("Severity column not found: 'Highest Injury Severity Alleged'")

# 2) Numerical
with tabs[1]:
    st.subheader("Numerical EDA")
    if not num_cols:
        st.info("No numerical columns detected.")
    else:
        sel_num = st.selectbox("Select numerical column", num_cols, index=min(0, len(num_cols)-1))
        st.plotly_chart(fig_boxplot(df, sel_num), use_container_width=True)
        # If both datasets present and column exists, allow UK vs US compare
        if sel_num in UK_data.columns and sel_num in US_data.columns and not UK_data.empty and not US_data.empty:
            st.markdown("**UK vs US distribution (normalized histogram)**")
            st.plotly_chart(fig_hist_density_compare(UK_data, US_data, sel_num), use_container_width=True)

# 3) Categorical
with tabs[2]:
    st.subheader("Categorical EDA")
    # Limit candidate columns to reasonable non-numeric ones
    if not cat_cols:
        st.info("No categorical columns detected.")
    else:
        sel_cat = st.selectbox("Select categorical column", cat_cols, index=min(0, len(cat_cols)-1))
        if sel_cat == "SV Contact Area":
            vc = explode_contact_area_for_counts(df, sel_cat, top_k)
            fig = px.bar(vc, x=sel_cat, y="Count", title=f"Top {top_k} {sel_cat}")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(fig_topk_bar(df, sel_cat, top_k), use_container_width=True)

# 4) Datetime
with tabs[3]:
    st.subheader("Datetime Distributions")
    # Year
    if "Year" in df.columns:
        st.plotly_chart(px.histogram(df, x="Year", title="Incidents by Year"), use_container_width=True)
    # Month
    if "Month" in df.columns:
        st.plotly_chart(px.histogram(df, x="Month", nbins=12, title="Incidents by Month"), use_container_width=True)
    # Weekday (ordered)
    if "Weekday" in df.columns:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        wd_counts = df["Weekday"].value_counts().reindex(order).fillna(0).reset_index()
        wd_counts.columns = ["Weekday", "Count"]
        st.plotly_chart(px.bar(wd_counts, x="Weekday", y="Count", title="Incidents by Weekday"), use_container_width=True)

    st.markdown("### Radial Time Plot (Matplotlib fallback)")
    fig = fig_radial_time_matplotlib(df)
    if fig is not None:
        st.pyplot(fig)
    else:
        st.info("No time data available for radial plot.")

# 5) Geospatial
with tabs[4]:
    st.subheader("Geospatial Analysis")
    if region == "US":
        st.plotly_chart(fig_us_state_choropleth(US_data), use_container_width=True)
        st.plotly_chart(fig_ca_top_cities(US_data, top_k), use_container_width=True)
    else:
        st.markdown("**UK Choropleth (GeoPandas + Matplotlib fallback)**")
        fig = fig_uk_choropleth_matplotlib(UK_data, uk_lads_path)
        if fig is not None:
            st.pyplot(fig)

# 6) Severity
with tabs[5]:
    st.subheader("Severity by Categorical Variable")
    sev_col = "Highest Injury Severity Alleged"
    if sev_col not in df.columns:
        st.info("Severity column not found: 'Highest Injury Severity Alleged'")
    else:
        # Pick a categorical column for stacking
        candidate_cats = [c for c in cat_cols if c != sev_col]
        if not candidate_cats:
            st.info("No suitable categorical columns for severity breakdown.")
        else:
            sel_cat = st.selectbox("Select categorical column", candidate_cats)
            st.plotly_chart(fig_severity_stack(df, sel_cat, sev_col, top_k), use_container_width=True)

st.markdown("---")
st.caption("Tip: If you see 'missing ScriptRunContext' warnings, make sure you launched with `streamlit run EDA.py`.")
