import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Point
# EDA.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_csv(path):
    return pd.read_csv(path)

def prep_dates(df, date_columns):
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def fig_missingness(df):
    fig = sns.heatmap(df.isna(), cbar=False)
    return fig

def fig_boxplot(df, col):
    fig = sns.boxplot(x=df[col])
    return fig

# ---------------------------
# UK Choropleth Map
# ---------------------------
def plot_uk_choropleth(UK_data, uk_shapefile_path):
    geometry = [Point(xy) for xy in zip(UK_data['longitude'], UK_data['latitude'])]
    gdf_points = gpd.GeoDataFrame(UK_data, geometry=geometry, crs="EPSG:4326")
    uk_lads = gpd.read_file(uk_shapefile_path)
    gdf_points = gdf_points.to_crs(uk_lads.crs)
    gdf_joined = gpd.sjoin(gdf_points, uk_lads, how="left", predicate="within")
    lad_counts = gdf_joined.groupby('LAD25NM').size().reset_index(name='incident_count')
    choropleth_gdf = uk_lads.merge(lad_counts, how='left', on='LAD25NM').fillna(0)
    
    fig = px.choropleth(
        choropleth_gdf,
        geojson=choropleth_gdf.geometry,
        locations=choropleth_gdf.index,
        color='incident_count',
        color_continuous_scale="Blues",
        title=f"UK Incidents by Local Authority (Total: {int(choropleth_gdf['incident_count'].sum()):,})"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    return fig

# ---------------------------
# US State Choropleth
# ---------------------------
def plot_us_state_choropleth(US_data):
    US_data['State'] = US_data['State'].astype(str).str.strip().str.upper()
    state_counts = US_data['State'].value_counts().reset_index()
    state_counts.columns = ['State', 'Count']
    
    fig = px.choropleth(
        state_counts,
        locations='State',
        locationmode='USA-states',
        color='Count',
        color_continuous_scale='Blues',
        scope='usa',
        title=f'US Incidents by State (Total: {state_counts["Count"].sum():,})'
    )
    return fig

# ---------------------------
# California Cities Bar Chart
# ---------------------------
def plot_california_cities(US_data, top_n=10):
    ca_data = US_data[US_data['State'].str.strip().str.upper() == 'CA']
    city_counts = ca_data['City'].value_counts().head(top_n).reset_index()
    city_counts.columns = ['City', 'Count']
    
    fig = px.bar(
        city_counts,
        x='City',
        y='Count',
        color='Count',
        color_continuous_scale='Blues',
        title=f'Top {top_n} California Cities by Incidents'
    )
    return fig

# ---------------------------
# Severity Analysis for Categorical Columns
# ---------------------------
def plot_severity_stacked(df, categorical_columns, severity_col='Highest Injury Severity Alleged', top_n=10, title_prefix='Data'):
    figs = []
    for col in categorical_columns:
        if col not in df.columns or severity_col not in df.columns:
            continue
        # Handle comma-separated categories
        if df[col].dtype == object and df[col].str.contains(',').any():
            all_rows = []
            for _, row in df[[col, severity_col]].dropna().iterrows():
                for val in row[col].split(','):
                    all_rows.append({col: val.strip(), severity_col: row[severity_col]})
            plot_df = pd.DataFrame(all_rows)
        else:
            plot_df = df[[col, severity_col]].dropna()
        grouped = plot_df.groupby([col, severity_col]).size().reset_index(name='Count')
        top_items = grouped.groupby(col)['Count'].sum().nlargest(top_n).index
        grouped_top = grouped[grouped[col].isin(top_items)]
        fig = px.bar(grouped_top, x=col, y='Count', color=severity_col, title=f'{title_prefix}: {col} (Top {top_n})')
        figs.append(fig)
    return figs
