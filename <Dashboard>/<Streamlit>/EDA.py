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

# ----- SU
import pandas as pd
import plotly.express as px

def get_imputation_df():
    """Return the hardcoded imputation comparison dataframe."""
    data = {
        "Column": [
            "Posted Speed Limit (MPH)", "Roadway Type", "Lighting", 
            "Roadway Surface", "Weather", "SV Pre-Crash Movement", 
            "Crash With", "SV Contact Area", "Highest Injury Severity Alleged"
        ],
        "Mode_Accuracy": [30.9, 36.1, 58.1, 87.6, 74.4, 55.2, 31.0, 36.4, 86.5],
        "RF_Accuracy": [68.6, 75.8, 65.1, 93.7, 81.1, 70.5, 52.8, 36.4, 89.0],
        "LOCF_Accuracy": [43.2, 49.0, 46.6, 80.6, 59.4, 48.6, 22.4, 32.4, 81.2],
        "XGB_Accuracy": [68.2, 74.1, 65.0, 96.9, 80.7, 69.1, 46.5, 55.8, 89.8],
        "Mode_Jaccard": [3.4, 6.0, 11.6, 21.9, 9.3, 4.6, 2.1, 44.9, 17.3],
        "RF_Jaccard": [25.3, 32.3, 18.9, 40.2, 25.1, 19.2, 24.3, 44.9, 36.6],
        "LOCF_Jaccard": [13.4, 15.3, 21.3, 22.3, 7.8, 7.9, 5.3, 45.0, 28.1],
        "XGB_Jaccard": [28.2, 36.5, 21.3, 54.9, 31.3, 18.0, 19.3, 65.5, 42.4]
    }
    return pd.DataFrame(data)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px

def plot_bar_side_by_side(df):
    """
    Creates a single figure with Accuracy and Jaccard bar charts side by side.
    """
    df_acc = df.melt(id_vars="Column", 
                     value_vars=["Mode_Accuracy", "RF_Accuracy", "LOCF_Accuracy", "XGB_Accuracy"],
                     var_name="Imputer", value_name="Value")
    
    df_jac = df.melt(id_vars="Column", 
                     value_vars=["Mode_Jaccard", "RF_Jaccard", "LOCF_Jaccard", "XGB_Jaccard"],
                     var_name="Imputer", value_name="Value")

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy (%)", "Jaccard (%)"))

    # Accuracy bars
    for imputer in df_acc['Imputer'].unique():
        df_tmp = df_acc[df_acc['Imputer'] == imputer]
        fig.add_trace(
            go.Bar(x=df_tmp['Column'], y=df_tmp['Value'], name=imputer, text=df_tmp['Value']),
            row=1, col=1
        )

    # Jaccard bars (do not duplicate legend)
    for imputer in df_jac['Imputer'].unique():
        df_tmp = df_jac[df_jac['Imputer'] == imputer]
        fig.add_trace(
            go.Bar(x=df_tmp['Column'], y=df_tmp['Value'], name=imputer, text=df_tmp['Value'], showlegend=True, legendgroup=imputer),
            row=1, col=2
        )

    fig.update_layout(
    title=dict(
        text="Imputation Accuracy & Jaccard Comparison",
        font=dict(color="black", size=20)
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="black"),
    barmode='group',
    xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
    xaxis2=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
    yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
    yaxis2=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
    legend=dict(title=dict(text="Imputer", font=dict(color="black")), font=dict(color="black"))
)


    return fig

def plot_line_side_by_side(df):
    """
    Creates a shaded line chart for Accuracy & Jaccard trends.
    """
    df_acc = df.melt(id_vars="Column", 
                     value_vars=["Mode_Accuracy", "RF_Accuracy", "LOCF_Accuracy", "XGB_Accuracy"],
                     var_name="Imputer", value_name="Value")
    df_acc["Metric"] = "Accuracy"

    df_jac = df.melt(id_vars="Column", 
                     value_vars=["Mode_Jaccard", "RF_Jaccard", "LOCF_Jaccard", "XGB_Jaccard"],
                     var_name="Imputer", value_name="Value")
    df_jac["Metric"] = "Jaccard"

    df_all = pd.concat([df_acc, df_jac])

    fig = px.line(df_all, x="Column", y="Value", color="Imputer", facet_row="Metric", markers=True)
    fig.update_traces(mode="lines+markers", line_shape="spline", fill='tozeroy')

    fig.update_layout(
        height=800,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black"),
        title=dict(
            text="Accuracy & Jaccard Trends by Imputer",
            font=dict(color="black", size=20)
        ),
        legend=dict(
            title=dict(text="Imputer", font=dict(color="black")),
            font=dict(color="black")
        ),
        xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
        xaxis2=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
        yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
        yaxis2=dict(title_font=dict(color="black"), tickfont=dict(color="black"))
    )

    return fig


import streamlit as st
import pandas as pd

def imputer_overall_summary(df):
    """
    Returns a summary of the best imputer model based on average Accuracy and Jaccard.
    """
    acc_cols = ['Mode_Accuracy', 'RF_Accuracy', 'LOCF_Accuracy', 'XGB_Accuracy']
    jac_cols = ['Mode_Jaccard', 'RF_Jaccard', 'LOCF_Jaccard', 'XGB_Jaccard']
    
    # Compute average per model
    avg_acc = df[acc_cols].mean()
    avg_jac = df[jac_cols].mean()
    
    # Find best model for each metric
    best_acc_model = avg_acc.idxmax().replace('_Accuracy','')
    best_jac_model = avg_jac.idxmax().replace('_Jaccard','')
    
    # Create summary table
    summary = pd.DataFrame({
        'Metric': ['Accuracy', 'Jaccard'],
        'Best_Model': [best_acc_model, best_jac_model],
        'Average_Value': [avg_acc.max(), avg_jac.max()]
    })
    
    # Style the table: bold header, light grey background
    styled_summary = summary.style.set_properties(
        **{
            'background-color': '#f5f5f5',  # light grey for all cells
            'color': 'black'
        }
    ).set_table_styles([
        {'selector': 'th', 'props': [('font-weight', 'bold'), ('background-color', '#d0d0d0'), ('color', 'black')]}
    ])
    
    st.table(styled_summary)  # single static table

    return summary
