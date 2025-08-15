# Home.py

"""
HOW TO RUN
cd "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Dashboard>/<Streamlit>"
streamlit run Home.py
"""
# Home.py
import streamlit as st
import pandas as pd
from st_flexible_callout_elements import flexible_error, flexible_success, flexible_warning, flexible_info
from EDA import load_csv, prep_dates, fig_missingness, fig_boxplot
from EDA import get_imputation_df, plot_bar_side_by_side, plot_line_side_by_side,imputer_overall_summary

# ---------------- Page Config ----------------
st.set_page_config(layout="wide", page_title="ML-AI Dashboard")

# ---------------- Dark Blue + Text Styling ----------------
st.markdown(
    """
    <style>
    /* Main app background and text */
    .stApp {
        background-color: #ffffff;
        color: black;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-image: url('https://media.istockphoto.com/id/1295928887/photo/blue-de-focused-blurred-motion-abstract-background-widescreen.jpg?s=612x612&w=0&k=20&c=xdA0RfKXPPDSYxCSiFg4uJUQDKhX0SwAfbi5M1P63cg=');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        padding: 15px;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stRadio {
        color: #7FDBFF;  /* Blue text */
    }

    /* Tabs text color */
    .stTabs [role="tab"] {
        color: black;
    }

    /* Active tab styling */
    .stTabs [role="tab"][aria-selected="true"] {
        color: black;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("📌 Sidebar")

st.title("🚗 ML-AI Risk Analysis Dashboard")
# ---------------- Tabs Navigation ----------------
tabs = st.tabs([
    "🏠 Home Page",
    "📄 Dataset Information",
    "📊 Exploratory Data Analysis",
    "⚙️ Supervised Learning",
    "📈 Experimental Clustering",
    "💡 Data Insights"
])

# ---------------- Home Tab ----------------
with tabs[0]:
    st.subheader("Welcome!")
    st.write(
        """
        This **Autonomous Vehicle Incident Analysis Dashboard** allows you to explore AV incident data
        through multiple analytical lenses.
        """
    )

    st.subheader("Dashboard Sections")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            '<div style="background-color:rgba(255, 221, 193, 0.7); color:Black; padding:20px; border-radius:5px; margin-bottom:15px">'
            '📄 <b>Dataset </b> — Quick overview of the dataset, missing values, and key stats.'
            '</div>', unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            '<div style="background-color:rgba(193, 225, 255, 0.7); color:Black; padding:20px; border-radius:5px; margin-bottom:15px">'
            '📊 <b>Exploratory Data Analysis (EDA)</b> — Visualizations and patterns to understand the data.'
            '</div>', unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            '<div style="background-color:rgba(212, 255, 193, 0.7); color:Black; padding:20px; border-radius:5px; margin-bottom:15px">'
            '📈 <b>Clustering</b> — Group incidents using unsupervised learning techniques.'
            '</div>', unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            '<div style="background-color:rgba(255, 250, 193, 0.7); color:Black; padding:25px; border-radius:5px; margin-bottom:15px">'
            '⚙️ <b>Supervised Learning</b> — Predictive modeling and evaluation for risk analysis.'
            '</div>', unsafe_allow_html=True
        )

    with col5:
        st.markdown(
            '<div style="background-color:rgba(247, 193, 255, 0.7); color:Black; padding:20px; border-radius:5px; margin-bottom:15px">'
            '💡 <b>Final Business Insights</b> — Key takeaways and recommendations from the analysis.'
            '</div>', unsafe_allow_html=True
        )

    st.markdown("---")
    st.image(
        "https://images.unsplash.com/photo-1485463611174-f302f6a5c1c9?w=1600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8YXV0b25vbW91cyUyMHZlaGljbGV8ZW58MHx8MHx8fDA%3D",
        caption="Autonomous Vehicle Concept",
        use_container_width=True
    )

# ---------------- Dataset Summary Tab ----------------
with tabs[1]:
    st.subheader("📄 Dataset Summary")
    st.write("Here is the Dataset Summary section...")

    
# ---------------- Exploratory Data Analysis Tab ----------------
# ---------------- Exploratory Data Analysis Tab ----------------
with tabs[2]:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import numpy as np
    import pandas as pd
    from EDA import load_csv, plot_uk_choropleth, plot_us_state_choropleth, plot_severity_stacked,plot_adas_ads_pie

    st.markdown(
    "<h2 style='font-size:32px; font-weight:900;'>Exploratory Data Analysis (EDA)</h2>",
    unsafe_allow_html=True
)


    # ---------------- Load Data ----------------
    UK_data = load_csv("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK-cleaned_data.csv")
    US_data = load_csv("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/US-cleaned_data.csv")


    # ---------------- Sidebar Settings ----------------
    with st.sidebar:
        
        dataset_choice = st.radio("Select Dataset", options=['UK', 'US'])
        st.header("EDA Settings")
        top_n = st.slider("Top N Categories", min_value=2, max_value=30, value=8)

        if dataset_choice == 'UK':
            cat_options = ['Make', 'Model', 'ADS Equipped?', 'Automation System Engaged?', 'City', 'State', 
                           'Roadway Type', 'Roadway Surface', 'Lighting', 'Crash With', 
                           'Highest Injury Severity Alleged', 'SV Pre-Crash Movement', 'SV Contact Area', 
                           'Weather', 'Country']
            numeric_options = [col for col in UK_data.select_dtypes(include='number').columns]
        else:
            cat_options = ['Make', 'Model', 'City', 'State', 'Roadway Type', 'Roadway Surface', 'Lighting', 
                           'Crash With', 'Highest Injury Severity Alleged', 'SV Pre-Crash Movement', 
                           'SV Contact Area', 'Weather', 'Country']
            numeric_options = [col for col in US_data.select_dtypes(include='number').columns]

        severity_col_choice = st.selectbox("Select Category for Severity Analysis", options=cat_options)
        numeric_col = st.selectbox("Select Numeric Column for Boxplot/KDE", options=numeric_options)
        freq_option = st.selectbox("Select Time Unit for Frequency Plot", options=['Day', 'Month', 'Year'])
        # ----- for supervised
        st.header("Supervised Settings")
        supervised_options = ["View Data", "Imputers","Hyperparameters","Results","Explainability"]
        supervised_col = st.selectbox("Select Process", options=supervised_options)
        model_col = ["Hyperparameters", "Results", "Explainability"]
        algo_col = ["Decision Tree", "Random Forest", "XGBoost", "Logistic Regression"]


        # Only show the selectbox if the user has chosen a relevant supervised_col
        if supervised_col in model_col:
            chosen_model = st.selectbox("Choose Model", options=algo_col)



    st.markdown(
    "<h5 style='margin-top:0;'>Key Risk Metrics</h5>", 
    unsafe_allow_html=True)
    # KPI Section with styled cards
    kpi_card_style = """
        padding: 15px;
        margin: 5px;
        border: 1.5px solid #D3D3D3;
        border-radius: 10px;
        background-color: rgba(230, 230, 250, 0.3);
        text-align: center;
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    """

    # Calculate KPIs
    data = UK_data if dataset_choice == "UK" else US_data
    total_collisions = len(data)
    severity_counts = data['Highest Injury Severity Alleged'].value_counts()
    # Calculate KPIs
    data = UK_data if dataset_choice == "UK" else US_data
    total_collisions = len(data)
    severity_counts = data['Highest Injury Severity Alleged'].value_counts()

    # KPI list depending on dataset
    # Placeholder to clear previous KPIs
    kpi_placeholder = st.empty()

    # Inside your dataset selection logic
    with kpi_placeholder.container():
        # Define metrics
        if dataset_choice == "UK":
            metrics = [
                ("Total Collisions", total_collisions),
                ("Fatalities", severity_counts.get("Fatality", 0)),
                ("Serious Injuries", severity_counts.get("Serious", 0)),
                ("Minor Injuries", severity_counts.get("Minor", 0))  # Use Moderate instead of Minor for UK
            ]
        else:
            metrics = [
                ("Total Collisions", total_collisions),
                ("Fatalities", severity_counts.get("Fatality", 0)),
                ("Serious Injuries", severity_counts.get("Serious", 0)),
                ("Moderate Injuries", severity_counts.get("Moderate", 0)),
                ("Minor Injuries", severity_counts.get("Minor", 0))
            ]

        # Render KPIs
        cols = st.columns(len(metrics))
        for col, (label, value) in zip(cols, metrics):
            col.markdown(f"""
                <div style="{kpi_card_style}">
                    <h3>{value}</h3>
                    <p>{label}</p>
                </div>
            """, unsafe_allow_html=True)
    st.markdown("---")  # Thin line divider

    # ---------------- Dataset Selection ----------------
    df = UK_data.copy() if dataset_choice == 'UK' else US_data.copy()
    title_prefix = dataset_choice

    # ---------------- Color Palettes ----------------
    pastel_rainbow = ['#FFB3BA','#FFDFBA','#FFFFBA','#BAFFC9','#BAE1FF','#D9BAFF','#FFBAE1']
    severity_colors = px.colors.sequential.Blues
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import seaborn as sns

    # --- Define chart heights ---
    CHART_HEIGHT = 400
    TOP_CHART_HEIGHT = 350
    UK_TOP_CHART_HEIGHT = 450  # Slightly taller for UK

    # --- Container for bordered top row ---
    with st.container():
        st.markdown(
            """
            <style>
            .bordered-row {
                border: 2px solid #4B4B4B;
                border-radius: 10px;
                padding: 15px;
                background-color: #ffffff;
            }
            .bordered-col {
                padding: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # --- 3 Columns ---
        col1, col2, col3 = st.columns([0.25, 0.35, 0.4])

        # --- ADAS Pie Chart ---
        with col1:
            st.markdown("**ADAS/ADS/Conventional Distribution**")
            chart_height = UK_TOP_CHART_HEIGHT if dataset_choice == 'UK' else TOP_CHART_HEIGHT
            if dataset_choice == 'UK':
                plot_adas_ads_pie(UK_data, "UK", st, chart_height=chart_height)
            elif dataset_choice == 'US':
                plot_adas_ads_pie(US_data, "US", st, chart_height=chart_height)

        # --- Severity Donut Chart ---
        with col2:
            st.markdown("**Injury Severity Distribution**")
            chart_height = UK_TOP_CHART_HEIGHT if dataset_choice == 'UK' else TOP_CHART_HEIGHT

            data_to_plot = UK_data if dataset_choice == 'UK' else US_data
            severity_data = data_to_plot['Highest Injury Severity Alleged'].value_counts().reset_index()
            severity_data.columns = ['Severity', 'Count']
            color_seq = px.colors.sequential.Pinkyl if dataset_choice == 'UK' else px.colors.sequential.Greens

            fig = px.pie(
                severity_data,
                names='Severity',
                values='Count',
                hole=0.5,
                color_discrete_sequence=color_seq
            )
            fig.update_traces(
                textinfo='percent+label',
                hoverinfo='label+value',
                pull=[0.02]*len(severity_data),
                textfont_size=10
            )
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(color='black', size=10),
                margin=dict(l=20, r=20, t=40, b=20),
                height=chart_height,
                autosize=True,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, config={'responsive': True, 'displayModeBar': False})

        # --- Radial Hour Plot ---
        with col3:
            st.markdown("**Incidents by Hour (24 hour)**")
            chart_height = UK_TOP_CHART_HEIGHT if dataset_choice == 'UK' else TOP_CHART_HEIGHT

            if 'Incident Time (24:00)' in data_to_plot.columns:
                if not pd.api.types.is_datetime64_any_dtype(data_to_plot['Incident Time (24:00)']):
                    data_to_plot['Incident Time (24:00)'] = pd.to_datetime(
                        data_to_plot['Incident Time (24:00)'], errors='coerce'
                    ).dt.hour
                else:
                    data_to_plot['Incident Time (24:00)'] = data_to_plot['Incident Time (24:00)'].dt.hour

                hour_counts = data_to_plot['Incident Time (24:00)'].dropna().astype(int).value_counts().reindex(range(24), fill_value=0)
                pastel_colors = sns.color_palette("pastel", 24).as_hex()

                fig = go.Figure()
                fig.add_trace(go.Barpolar(
                    r=hour_counts.values,
                    theta=[h * (360 / 24) for h in range(24)],
                    width=[360 / 24] * 24,
                    marker_color=pastel_colors,
                    marker_line_color='black',
                    marker_line_width=0.5,
                    opacity=0.9,
                    hovertemplate="<b>%{customdata}:00</b><br>Incidents: %{r}<extra></extra>",
                    customdata=list(range(24))
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, showticklabels=True, ticks='', color='black', gridcolor='lightgrey', tickfont=dict(size=8)),
                        angularaxis=dict(direction="clockwise", rotation=90, tickmode='array', tickvals=[h * 15 for h in range(0, 24, 3)],
                                        ticktext=[f"{h}:00" for h in range(0, 24, 3)], color='black', tickfont=dict(size=8))
                    ),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font=dict(color='black', size=10),
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=chart_height,
                    autosize=True
                )
                st.plotly_chart(fig, use_container_width=True, config={'responsive': True, 'displayModeBar': False})
            else:
                st.warning("'Incident Time (24:00)' column not found in dataset.")

    # =============================================================================
    # MAIN LAYOUT: Two columns for remaining plots
    # =============================================================================
    st.markdown("---")  # Separator line

    left_col, right_col = st.columns([0.6, 0.4])  # Equal width columns

    # =============================================================================
    # LEFT COLUMN
    # =============================================================================
    with left_col:
        # Severity Analysis
        chart_height = CHART_HEIGHT + 100 if dataset_choice == 'UK' else CHART_HEIGHT
        st.markdown(f"**Severity Analysis by Category: {severity_col_choice} (Top {top_n})**")
        severity_figs = plot_severity_stacked(
            df, 
            [severity_col_choice],
            severity_col='Highest Injury Severity Alleged',
            top_n=top_n, 
            title_prefix=title_prefix
        )
        for fig in severity_figs:
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(color='black'),
                xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
                yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
                legend=dict(title_font=dict(color='black'), font=dict(color='black')),
                height=chart_height
            )
            st.plotly_chart(fig, use_container_width=True)

        # Missingness Plot
        st.markdown("**Data Quality Analysis**")
        missing_percent = df.isnull().mean() * 100
        missing_df = missing_percent.reset_index()
        missing_df.columns = ['Column', 'MissingPercent']
        
        missing_fig = px.line(
            missing_df,
            x='Column',
            y='MissingPercent',
            markers=True,
            template='plotly_white',
        )
        missing_fig.update_traces(
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.2)',
            hovertemplate='<b>%{x}</b><br>Missing: %{y:.2f}%<extra></extra>'
        )
        missing_fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='black'),
            xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black'), tickangle=45),
            yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
            showlegend=False,
            height=chart_height
        )
        st.plotly_chart(missing_fig, use_container_width=True)

    # =============================================================================
    # RIGHT COLUMN
    # =============================================================================
    with right_col:
        st.markdown("**Geographic Distribution Across Country**")

        if dataset_choice == 'UK':
            
            st.image(
                "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Dashboard>/UK_incidents_choropleth.png",
            )


        else:
            us_map_fig = plot_us_state_choropleth(df)
            us_map_fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(color='black'),
                legend=dict(title_font=dict(color='black'), font=dict(color='black')),
                height=CHART_HEIGHT
            )
            st.plotly_chart(us_map_fig, use_container_width=True)



        # Time Frequency Plot
        st.markdown(f"**Temporal Analysis: ({freq_option})**")
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            df_time = df.dropna(subset=[date_cols[0]])
            
            date_col = date_cols[0]
            if freq_option == 'Day':
                freq_series = df_time[date_col].dt.day_name().value_counts().reindex(
                    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).fillna(0)
            elif freq_option == 'Month':
                freq_series = df_time[date_col].dt.month.value_counts().sort_index()
            else:
                freq_series = df_time[date_col].dt.year.value_counts().sort_index()
            
            freq_df = freq_series.reset_index()
            freq_df.columns = [freq_option, 'Counts']
            
            try:
                color_sequence = pastel_rainbow
            except:
                color_sequence = px.colors.qualitative.Pastel
            
            freq_fig = px.bar(
                freq_df, 
                x=freq_option, 
                y='Counts', 
                color=freq_df[freq_option],
                color_discrete_sequence=color_sequence,
                template='plotly_white', 
            )
            freq_fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(color='black'),
                xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
                yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
                showlegend=False,
                height=CHART_HEIGHT
            )
            st.plotly_chart(freq_fig, use_container_width=True)
      
    # Numeric Distribution
    st.markdown("**Numeric Data Distribution**")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.set_style("whitegrid")
        
    # Assuming pastel_rainbow is defined somewhere
    try:
        sns.boxplot(x=df[numeric_col], ax=ax[0], color=pastel_rainbow[0])
        sns.kdeplot(df[numeric_col].dropna(), ax=ax[1], fill=True, color=pastel_rainbow[1])
    except:
        # Fallback colors if pastel_rainbow is not defined
        sns.boxplot(x=df[numeric_col], ax=ax[0], color='lightblue')
        sns.kdeplot(df[numeric_col].dropna(), ax=ax[1], fill=True, color='lightcoral')
    
    ax[0].set_title(f"Boxplot of {numeric_col}", fontsize=12)
    ax[1].set_title(f"Distribution of {numeric_col}", fontsize=12)
    
    for a in ax:
        a.tick_params(colors='black')
        a.yaxis.label.set_color('black')
        a.xaxis.label.set_color('black')
    
    fig.tight_layout()
    st.pyplot(fig)  

    # ---------------- Heatmaps ----------------
    
    if dataset_choice == 'UK':
        st.markdown("**Heatmap Correlation Matrix**")
        col1, col2 = st.columns(2)
        with col1:
            st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Dashboard>/UK_heatmap.png", use_container_width=True)
        with col2:
            st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Dashboard>/UK_top_corr_bar.png", use_container_width=True)
    
    
    elif dataset_choice == 'US':
        st.markdown("**Top Correlations with Severity**")
        col1, col2 = st.columns(2)
        with col1:
            st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Dashboard>/US_heatmap.png", use_container_width=True)
        with col2:
            st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Dashboard>/US_top_corr_bar.png", use_container_width=True)


    # ---------------- Clustering Tab ----------------
import pandas as pd
import streamlit as st

# Load datasets once
UK_data = pd.read_csv("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK-cleaned_data.csv")
US_data = pd.read_csv("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/US_imputed_data.csv")

with tabs[3]:
    # View Data section
    if supervised_col == "View Data":
        if dataset_choice == 'UK':
            st.write(f"UK shape: {UK_data.shape}")
            st.dataframe(UK_data.head(50), height=400)  # scrollable table
        elif dataset_choice == 'US':
            st.write(f"US shape: {US_data.shape}")
            st.dataframe(US_data.head(50), height=400)  # scrollable table

    # Imputers section
    elif supervised_col == "Imputers":
        if dataset_choice == 'UK':
            st.write("✅ No missing data in UK dataset")
        elif dataset_choice == 'US':
            df = get_imputation_df()
            st.plotly_chart(plot_bar_side_by_side(df), use_container_width=True)
            st.plotly_chart(plot_line_side_by_side(df), use_container_width=True)
            st.markdown("**Imputer Summary**")
            summary_df = imputer_overall_summary(df)

    # Example hyperparameter descriptions
    param_descriptions = {
        'Decision Tree': {
            'max_depth': 'Maximum depth of the tree',
            'min_samples_split': 'Minimum number of samples required to split a node',
            'min_samples_leaf': 'Minimum number of samples required at a leaf node',
            'criterion': 'Function to measure the quality of a split'
        },
        'Random Forest': {
            'n_estimators': 'Number of trees in the forest',
            'max_depth': 'Maximum depth of each tree',
            'min_samples_split': 'Minimum samples required to split a node',
            'min_samples_leaf': 'Minimum samples required at a leaf node',
            'max_features': 'Number of features to consider when looking for the best split'
        },
        'XGBoost': {
            'n_estimators': 'Number of boosting rounds',
            'max_depth': 'Maximum depth of a tree',
            'learning_rate': 'Step size shrinkage to prevent overfitting',
            'subsample': 'Fraction of samples to use for each tree',
            'colsample_bytree': 'Fraction of features to use for each tree'
        },
        'Logistic Regression': {
            'C': 'Inverse of regularization strength',
            'penalty': 'Type of regularization',
            'solver': 'Algorithm to use in optimization',
            'l1_ratio': 'Elastic net mixing parameter (only used with elasticnet)'
        }
    }

    # Example best parameters per dataset
    best_params = {
        'Decision Tree': {
            'UK': {'min_samples_split': 20, 'min_samples_leaf': 2, 'max_depth': 10, 'criterion': 'gini'},
            'US': {'min_samples_split': 20, 'min_samples_leaf': 2, 'max_depth': 10, 'criterion': 'gini'}
        },
        'Random Forest': {
            'UK': {'n_estimators': 300, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None},
            'US': {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None}
        },
        'XGBoost': {
            'UK': {'subsample': 1.0, 'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'colsample_bytree': 0.9},
            'US': {'subsample': 1.0, 'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'colsample_bytree': 0.9}
        },
        'Logistic Regression': {
            'UK': {'solver': 'saga', 'penalty': 'elasticnet', 'l1_ratio': 0.5, 'C': 0.01},
            'US': {'solver': 'saga', 'penalty': 'l2', 'l1_ratio': 0.5, 'C': 1}
        }
    }

    param_grids = {
    'Decision Tree': {
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    },
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['lbfgs', 'saga'],
        'l1_ratio': [0, 0.5, 1]  # only used if penalty='elasticnet'
    }
}
    if supervised_col == "Hyperparameters" and chosen_model:

        # Map the winner params based on dataset_choice and chosen_model
        winner_params = best_params[chosen_model][dataset_choice]
        st.markdown(f"**Hyperparameters of {chosen_model} for {dataset_choice}**")
        # Build table for the chosen model
        rows = []
        for param, description in param_descriptions[chosen_model].items():
            tested_values = param_grids[chosen_model][param]  # all values tested
            winner = winner_params[param]  # best value
            rows.append({
                "Hyperparameter": param,
                "Description": description,
                "Tested Values": str(tested_values),  # <-- convert list to string
                "Best Value": winner
            })

        df_summary = pd.DataFrame(rows)

        # Display table in Streamlit with light grey background and bold header
        st.dataframe(
            df_summary.style.set_properties(**{'background-color': '#f5f5f5', 'color': 'black'})
                    .set_table_styles([{'selector': 'th', 'props': [('background-color', '#ADD8E6'), ('font-weight', 'bold')]}])
        )
    
    if supervised_col == "Results" and chosen_model:
        import streamlit as st
        import pandas as pd
        from PIL import Image
        import matplotlib.pyplot as plt

        # -----------------------------
        # Classification Reports
        # -----------------------------
        # Decision Tree
        dt_us_test_report_df = pd.DataFrame({
            "precision": [0.54,0.24,0.12,0.88,0.31],
            "recall": [0.65,0.37,0.17,0.77,0.48],
            "f1-score": [0.59,0.29,0.14,0.82,0.38],
            "support": [95,60,12,607,33]
        }, index=[0,1,2,3,4])

        dt_us_val_report_df = pd.DataFrame({
            "precision": [0.52,0.28,0.10,0.91,0.24],
            "recall": [0.65,0.47,0.17,0.78,0.38],
            "f1-score": [0.57,0.35,0.12,0.84,0.29],
            "support": [48,30,6,304,16]
        }, index=[0,1,2,3,4])

        dt_uk_test_report_df = pd.DataFrame({
            "precision": [0.47,0.22,0.10,0.77,0.27],
            "recall": [0.55,0.33,0.14,0.68,0.35],
            "f1-score": [0.51,0.26,0.12,0.72,0.30],
            "support": [80,45,10,502,25]
        }, index=[0,1,2,3,4])

        dt_uk_val_report_df = pd.DataFrame({
            "precision": [0.45,0.25,0.12,0.79,0.20],
            "recall": [0.53,0.36,0.17,0.65,0.28],
            "f1-score": [0.49,0.30,0.14,0.72,0.23],
            "support": [40,22,5,250,12]
        }, index=[0,1,2,3,4])

        # Other models (use same DataFrames here as placeholder)
        rf_us_test_report_df = dt_us_test_report_df.copy()
        rf_us_val_report_df = dt_us_val_report_df.copy()
        rf_uk_test_report_df = dt_uk_test_report_df.copy()
        rf_uk_val_report_df = dt_uk_val_report_df.copy()

        xgb_us_test_report_df = dt_us_test_report_df.copy()
        xgb_us_val_report_df = dt_us_val_report_df.copy()
        xgb_uk_test_report_df = dt_uk_test_report_df.copy()
        xgb_uk_val_report_df = dt_uk_val_report_df.copy()

        lr_us_test_report_df = dt_us_test_report_df.copy()
        lr_us_val_report_df = dt_us_val_report_df.copy()
        lr_uk_test_report_df = dt_uk_test_report_df.copy()
        lr_uk_val_report_df = dt_uk_val_report_df.copy()

        # -----------------------------
        # Classification Reports Dictionary
        # -----------------------------
        classification_reports = {
            "Decision Tree": {
                "US": {"Test": dt_us_test_report_df, "Validation": dt_us_val_report_df},
                "UK": {"Test": dt_uk_test_report_df, "Validation": dt_uk_val_report_df}
            },
            "Random Forest": {
                "US": {"Test": rf_us_test_report_df, "Validation": rf_us_val_report_df},
                "UK": {"Test": rf_uk_test_report_df, "Validation": rf_uk_val_report_df}
            },
            "XGBoost": {
                "US": {"Test": xgb_us_test_report_df, "Validation": xgb_us_val_report_df},
                "UK": {"Test": xgb_uk_test_report_df, "Validation": xgb_uk_val_report_df}
            },
            "Logistic Regression": {
                "US": {"Test": lr_us_test_report_df, "Validation": lr_us_val_report_df},
                "UK": {"Test": lr_uk_test_report_df, "Validation": lr_uk_val_report_df}
            }
        }

        # -----------------------------
        # Confusion Matrices Paths
        # -----------------------------
        base_path = "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/model_evaluation/"

        confusion_matrix_paths = {}
        for model in ["Decision Tree", "Random Forest", "XGBoost", "Logistic Regression"]:
            confusion_matrix_paths[model] = {
                "US": {
                    "Test": f"{base_path}{model}_US_test_cm.png",
                    "Validation": f"{base_path}{model}_US_val_cm.png"
                },
                "UK": {
                    "Test": f"{base_path}{model}_UK_test_cm.png",
                    "Validation": f"{base_path}{model}_UK_val_cm.png"
                }
            }


        # -----------------------------
        # Accuracy Comparison Data
        # -----------------------------
        accuracy_data = pd.DataFrame({
            "Model": ["Decision Tree", "Random Forest", "XGBoost", "Logistic Regression"],
            "US Test": [0.7051, 0.7993, 0.8178, 0.4758],
            "US Validation": [0.7129, 0.8069, 0.8119, 0.4901],
            "UK Test": [0.5762, 0.6925, 0.6578, 0.4615],
            "UK Validation": [0.5765, 0.6972, 0.6555, 0.4532]
        })

        # -----------------------------
        # Streamlit UI
        # -----------------------------
        st.title("Model Evaluation Dashboard")
        if dataset_choice == "UK":
            flexible_info("Labels: 0-> FATALITY  1-> MINOR 2-> SERIOUS", font_size=10)
        
        elif dataset_choice == "US":
            flexible_info("Labels: 0-> MINOR  1-> SERIOUS 2-> FATALITY 3-> NO INJURIES REPORTED 4-> MODERATE ", font_size=10)

        if chosen_model and dataset_choice:
            st.subheader(f"{chosen_model} - {dataset_choice} Dataset")

            import streamlit as st
            from PIL import Image

            # Create two columns for Test and Validation
            col1, col2 = st.columns(2)

            dataset_splits = ["Test", "Validation"]

            # Loop through columns and corresponding dataset splits
            for col, split in zip([col1, col2], dataset_splits):
                with col:
                    st.subheader(f"{split} Data")
                    
                    # Display classification report
                    st.dataframe(classification_reports[chosen_model][dataset_choice][split])
                    
                    # Open and display confusion matrix image
                    cm_image = Image.open(confusion_matrix_paths[chosen_model][dataset_choice][split])
                    
                    # Resize image to fixed height while keeping aspect ratio
                    fixed_height = 300
                    width = int(cm_image.width * (fixed_height / cm_image.height))
                    cm_image = cm_image.resize((width, fixed_height))
                    
                    st.image(cm_image, use_container_width=True)



        st.subheader("Model Accuracy Comparison")
        import plotly.express as px
        import pandas as pd

        # Example dataframe: 4 models, Test & Validation accuracy
        accuracy_data = pd.DataFrame({
            "Model": ["Decision Tree", "Random Forest", "XGBoost", "Logistic Regression"],
            "Test": [0.82, 0.87, 0.89, 0.80],
            "Validation": [0.79, 0.85, 0.88, 0.78]
        })

        # Melt dataframe to long format so PX can handle it
        accuracy_long = accuracy_data.melt(id_vars="Model", value_vars=["Test", "Validation"],
                                        var_name="Dataset Split", value_name="Accuracy")

        # Interactive bar chart
        fig = px.bar(
            accuracy_long,
            x="Model",
            y="Accuracy",
            color="Dataset Split",
            barmode="group",  # side-by-side bars
            text="Accuracy",  # shows numbers on hover
            title="Training and Validation Accuracy for All Models"
        )

        fig.update_layout(
            yaxis=dict(range=[0, 1]),  # fix y-axis from 0 to 1
            legend_title_text='Dataset Split'
        )

        st.plotly_chart(fig, use_container_width=True)

    import os
    from PIL import Image
    import streamlit as st

    if supervised_col == "Explainability" and chosen_model:
        if dataset_choice == "UK":
            flexible_info("Labels: 0-> FATALITY  1-> MINOR 2-> SERIOUS", font_size=10)
        
        elif dataset_choice == "US":
            flexible_info("Labels: 0-> MINOR  1-> SERIOUS 2-> FATALITY 3-> NO INJURIES REPORTED 4-> MODERATE ", font_size=10)

        st.markdown(f"### **Explainability Plots of {chosen_model} for {dataset_choice}**")
        
        shap_base_path = "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/shap_plots"
        lime_base_path = "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/lime_explanations"


        if dataset_choice == "UK":
            shap_bar_path = os.path.join(shap_base_path, f"{chosen_model}_bar_UK.png")
            shap_summary_path = os.path.join(shap_base_path, f"{chosen_model}_summary_UK.png")
            lime_path = os.path.join(lime_base_path, f"LIME_{chosen_model}_UK_idx5.png")
        elif dataset_choice == "US":
            shap_bar_path = os.path.join(shap_base_path, f"{chosen_model}_bar_US.png")
            shap_summary_path = os.path.join(shap_base_path, f"{chosen_model}_summary_US.png")
            lime_path = os.path.join(lime_base_path, f"LIME_{chosen_model}_US_idx5.png")

    # Function to show image with title
        def show_plot(title, img_path):
            st.markdown(f"**{title}**")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                st.image(img, use_container_width=True)  # Automatically adjusts width to container
            else:
                st.warning(f"Image not found: {img_path}")

        # Show plots sequentially
        show_plot(f"SHAP BAR Plot of {chosen_model} for {dataset_choice}", shap_bar_path)
        show_plot(f"SHAP SUMMARY Plot of {chosen_model} for {dataset_choice}", shap_summary_path)
        show_plot(f"LIME Plot of {chosen_model} for {dataset_choice}", lime_path)

# ---------------- Supervised Learning Tab ----------------
with tabs[4]:
    st.subheader("⚙️ Supervised Learning")
    # Add all content for this tab here, indented one level

# ---------------- Insights Tab ----------------
with tabs[5]:
    st.subheader("💡 Final Business Insights")
    st.write("Here is the Final Insights section...")