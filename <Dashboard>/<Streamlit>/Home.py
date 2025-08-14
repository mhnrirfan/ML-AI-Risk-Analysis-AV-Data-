# Home.py

"""
HOW TO RUN
cd "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Dashboard>/<Streamlit>"
streamlit run Home.py
"""
# Home.py
import streamlit as st
import pandas as pd
from EDA import load_csv, prep_dates, fig_missingness, fig_boxplot

# ---------------- Page Config ----------------
st.set_page_config(layout="wide", page_title="ML-AI Dashboard")

# ---------------- Dark Blue + Text Styling ----------------
st.markdown(
    """
    <style>
    /* Main app background and text */
    .stApp {
        background-color: #f9f9f9;
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
    "📈 Experimental Clustering",
    "⚙️ Supervised Learning",
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

    st.subheader("📊 Exploratory Data Analysis (EDA)")

    # ---------------- Load Data ----------------
    UK_data = load_csv("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK-cleaned_data.csv")
    US_data = load_csv("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/US-cleaned_data.csv")


    # ---------------- Sidebar Settings ----------------
    with st.sidebar:
        st.header("EDA Settings")
        
        dataset_choice = st.radio("Select Dataset", options=['US', 'UK'])
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

    # ---------------- Dataset Selection ----------------
    df = UK_data.copy() if dataset_choice == 'UK' else US_data.copy()
    title_prefix = dataset_choice

    # ---------------- Color Palettes ----------------
    pastel_rainbow = ['#FFB3BA','#FFDFBA','#FFFFBA','#BAFFC9','#BAE1FF','#D9BAFF','#FFBAE1']
    severity_colors = px.colors.sequential.Blues
    # -- IMPROVED DASHBOARD LAYOUT
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import seaborn as sns
    import matplotlib.pyplot as plt

    # --- Define a fixed height for all charts ---
    CHART_HEIGHT = 400
    TOP_CHART_HEIGHT = 350  # Slightly smaller for top row
    col1, col2, col3 = st.columns([0.25, 0.35, 0.4])  # Equal width columns

    # --- ADAS Pie Chart ---
    with col1:
        st.markdown("**ADAS/ADS Distribution**")
        if dataset_choice == 'UK':
            plot_adas_ads_pie(UK_data, "UK", st, chart_height=TOP_CHART_HEIGHT)
        elif dataset_choice == 'US':
            plot_adas_ads_pie(US_data, "US", st, chart_height=TOP_CHART_HEIGHT)

    # --- Severity Donut Chart ---
    with col2:
        st.markdown("**Injury Severity**")
        if dataset_choice == 'UK':
            data_to_plot = UK_data
        elif dataset_choice == 'US':
            data_to_plot = US_data
        
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
            textfont_size=10  # Smaller text to fit better
        )
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='black', size=10),
            margin=dict(l=20, r=20, t=40, b=20),  # More balanced margins
            height=TOP_CHART_HEIGHT,
            width=None,  # Let it auto-size width
            showlegend=False,
            autosize=True  # Enable auto-sizing
        )
        st.plotly_chart(fig, use_container_width=True, config={'responsive': True, 'displayModeBar': False})

    # --- Radial Hour Plot ---
    with col3:
        st.markdown("**Incidents by Hour**")
        if 'Incident Time (24:00)' in data_to_plot.columns:
            # Convert to hour
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
                marker_line_width=0.5,  # Thinner lines
                opacity=0.9,
                hovertemplate="<b>%{customdata}:00</b><br>Incidents: %{r}<extra></extra>",
                customdata=list(range(24))
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, 
                        showticklabels=True, 
                        ticks='', 
                        color='black', 
                        gridcolor='lightgrey',
                        tickfont=dict(size=8)  # Smaller tick labels
                    ),
                    angularaxis=dict(
                        direction="clockwise", 
                        rotation=90,
                        tickmode='array',
                        tickvals=[h * (360 / 24) for h in range(0, 24, 3)],  # Show every 3rd hour
                        ticktext=[f"{h}:00" for h in range(0, 24, 3)],
                        color='black',
                        tickfont=dict(size=8)  # Smaller labels
                    )
                ),
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(color='black', size=10),
                margin=dict(l=20, r=20, t=40, b=20),  # Consistent margins
                height=TOP_CHART_HEIGHT,
                width=None,  # Let it auto-size width
                autosize=True  # Enable auto-sizing
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
        st.markdown("### 📈 Severity Analysis by Category")
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
                height=CHART_HEIGHT
            )
            st.plotly_chart(fig, use_container_width=True)

        # Missingness Plot
        st.markdown("### 🔍 Data Quality Analysis")
        missing_percent = df.isnull().mean() * 100
        missing_df = missing_percent.reset_index()
        missing_df.columns = ['Column', 'MissingPercent']
        
        missing_fig = px.line(
            missing_df,
            x='Column',
            y='MissingPercent',
            markers=True,
            template='plotly_white',
            title=f"Missing Data by Column ({title_prefix})"
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
            height=CHART_HEIGHT
        )
        st.plotly_chart(missing_fig, use_container_width=True)

    # =============================================================================
    # RIGHT COLUMN
    # =============================================================================
    with right_col:
        # Map
        st.markdown("### 🗺️ Geographic Distribution")
        if dataset_choice == 'UK':
            st.markdown("**UK Incident Map by Local Authority**")
            # Placeholder for UK map (uncomment when shapefile is available)
            # uk_map_fig = plot_uk_choropleth(df, uk_shapefile_path)
            # uk_map_fig.update_layout(
            #     paper_bgcolor='white',
            #     plot_bgcolor='white',
            #     font=dict(color='black'),
            #     legend=dict(title_font=dict(color='black'), font=dict(color='black')),
            #     height=CHART_HEIGHT
            # )
            # st.plotly_chart(uk_map_fig, use_container_width=True)
            st.info("UK map will be displayed when shapefile is available")
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
        st.markdown(f"### ⏰ Temporal Analysis ({freq_option})")
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
                title=f"Incident Frequency by {freq_option} ({title_prefix})"
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
    st.markdown("### 📊 Numeric Data Distribution")
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
# ---------------- Clustering Tab ----------------
with tabs[3]:
    st.subheader("📈 Clustering")
    st.write("Here is the Clustering section...")

# ---------------- Supervised Learning Tab ----------------
with tabs[4]:
    st.subheader("⚙️ Supervised Learning")
    st.write("Here is the Supervised Learning section...")

# ---------------- Insights Tab ----------------
with tabs[5]:
    st.subheader("💡 Final Business Insights")
    st.write("Here is the Final Insights section...")
