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
import geopandas as gpd
from shapely.geometry import Point
import plotly.express as px

def plot_uk_choropleth():
    # Example choropleth figure
    fig = px.choropleth(...)  # your actual plotting code here
    
    # Save figure to file
    fig.write_image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Dashboard>/UK_incidents_choropleth.png")
    
    # Return the figure object
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
    )
    return fig

# ---------------------------
# Severity Analysis for Categorical Columns
# ---------------------------
def plot_severity_stacked(df, categorical_columns, severity_col='Highest Injury Severity Alleged', top_n=10, title_prefix='Data'):
    import plotly.express as px
    import pandas as pd
    import seaborn as sns

    # Convert all categorical columns to string
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Define a consistent green palette for severity
    severity_levels = df[severity_col].dropna().unique()
    severity_levels_sorted = sorted(severity_levels)  # optional: sort for consistent order
    green_palette = sns.light_palette("green", n_colors=len(severity_levels_sorted), reverse=False).as_hex()
    severity_color_map = dict(zip(severity_levels_sorted, green_palette))

    figs = []

    for col in categorical_columns:
        if col not in df.columns or severity_col not in df.columns:
            continue

        # Handle comma-separated categories
        if df[col].str.contains(',').any():
            all_rows = []
            for _, row in df[[col, severity_col]].dropna().iterrows():
                for val in row[col].split(','):
                    all_rows.append({col: val.strip(), severity_col: row[severity_col]})
            plot_df = pd.DataFrame(all_rows)
        else:
            plot_df = df[[col, severity_col]].dropna()

        # Group by category and severity
        grouped = plot_df.groupby([col, severity_col]).size().reset_index(name='Count')

        # Keep top_n categories
        top_items = grouped.groupby(col)['Count'].sum().nlargest(top_n).index
        grouped_top = grouped[grouped[col].isin(top_items)]

        # Create stacked bar chart
        fig = px.bar(
            grouped_top,
            x=col,
            y='Count',
            color=severity_col,  # stacked by severity
            color_discrete_map=severity_color_map,  # consistent green shades
            custom_data=[grouped_top[col], grouped_top[severity_col], grouped_top['Count']]  # for hover / click
        )

        # Add hover info
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>Severity: %{customdata[1]}<br>Count: %{customdata[2]}<extra></extra>'
        )

        # Update layout: white background + font for clarity
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='black'),
            legend_title_text='Severity'
        )

        figs.append(fig)

    return figs




def plot_adas_ads_pie(df, dataset_label, st, chart_height=400):
    if 'Automation System Engaged?' not in df.columns:
        st.warning(f"'Automation System Engaged?' column not found in {dataset_label} dataset.")
        return

    system_counts = df['Automation System Engaged?'].value_counts()
    pastel_colors = sns.color_palette("pastel", len(system_counts)).as_hex()

    fig = px.pie(
        names=system_counts.index,
        values=system_counts.values,
        color=system_counts.index,
        color_discrete_sequence=pastel_colors,
        hole=0.3
    )

    fig.update_traces(
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
    )

    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black'),
        height=chart_height  # <-- Added chart height
    )

    st.plotly_chart(fig, use_container_width=True)
