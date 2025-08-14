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
        background-color: white;
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
    from EDA import load_csv, plot_uk_choropleth, plot_us_state_choropleth, plot_severity_stacked

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

    # ---------------- Dataset Selection ----------------
    df = UK_data.copy() if dataset_choice == 'UK' else US_data.copy()
    title_prefix = dataset_choice

    # ---------------- Color Palettes ----------------
    pastel_rainbow = ['#FFB3BA','#FFDFBA','#FFFFBA','#BAFFC9','#BAE1FF','#D9BAFF','#FFBAE1']
    severity_colors = px.colors.sequential.Blues

    # ---------------- Layout ----------------
    col1, col2 = st.columns([0.55, 0.45])

    # ---------------- Left Column ----------------
    with col1:
        # Severity Analysis
        st.markdown("### Severity Analysis by Category")
        severity_figs = plot_severity_stacked(df, [severity_col_choice],
                                              severity_col='Highest Injury Severity Alleged',
                                              top_n=top_n, title_prefix=title_prefix)
        for fig in severity_figs:
            fig.update_traces(marker=dict(color=severity_colors))
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(color='black'),
                xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
                yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
                legend=dict(title_font=dict(color='black'), font=dict(color='black'))
            )
            st.plotly_chart(fig, use_container_width=True)

        # Missingness Plot
        st.markdown("### Missingness per Column")
        missing_percent = df.isnull().mean() * 100
        missing_df = missing_percent.reset_index()
        missing_df.columns = ['Column', 'MissingPercent']
        missing_fig = px.bar(missing_df, x='Column', y='MissingPercent', color='Column',
                             color_discrete_sequence=pastel_rainbow,
                             template='plotly_white', title=f"Percentage Missing per Column ({title_prefix})")
        missing_fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='black'),
            xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
            yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
            showlegend=False
        )
        st.plotly_chart(missing_fig, use_container_width=True)

        # Numeric Boxplot & KDE
        st.markdown("### Distribution of Numeric Column")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.set_style("whitegrid")
        sns.boxplot(x=df[numeric_col], ax=ax[0], color=pastel_rainbow[0])
        ax[0].set_title(f"Boxplot of {numeric_col}", fontsize=12)
        sns.kdeplot(df[numeric_col].dropna(), ax=ax[1], fill=True, color=pastel_rainbow[1])
        ax[1].set_title(f"KDE of {numeric_col}", fontsize=12)
        for a in ax:
            a.tick_params(colors='black')
            a.yaxis.label.set_color('black')
            a.xaxis.label.set_color('black')
        fig.tight_layout()
        st.pyplot(fig)

    # ---------------- Right Column ----------------
    with col2:
        # Map
        if dataset_choice == 'UK':
            st.markdown("### UK Incident Map by Local Authority")
            uk_shapefile_path = "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/LAD_MAY_2025_UK_BFC_2360005762104150824"
            uk_map_fig = plot_uk_choropleth(df, uk_shapefile_path)
            uk_map_fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(color='black'),
                legend=dict(title_font=dict(color='black'), font=dict(color='black'))
            )
            st.plotly_chart(uk_map_fig, use_container_width=True)
        else:
            st.markdown("### US Incident Map by State")
            us_map_fig = plot_us_state_choropleth(df)
            us_map_fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(color='black'),
                legend=dict(title_font=dict(color='black'), font=dict(color='black'))
            )
            st.plotly_chart(us_map_fig, use_container_width=True)

        # Radial Hour Plot
        st.markdown("### Hourly Incident Distribution (Clock)")
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            df_time = df.dropna(subset=[date_cols[0]])
            df_time['hour'] = df_time[date_cols[0]].dt.hour
            hour_counts = df_time['hour'].value_counts().reindex(range(24), fill_value=0)

            theta = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
            radii = hour_counts.values
            width = 2 * np.pi / 24

            colors = plt.cm.viridis(radii / radii.max()) if radii.max() > 0 else plt.cm.viridis(np.zeros_like(radii))
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
            bars = ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.8)

            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
            ax.set_xticklabels(range(24), color='black')
            ax.yaxis.label.set_color('black')
            ax.tick_params(colors='black')
            ax.set_title(f"Incidents by Hour ({title_prefix})", va='bottom', color='black')
            fig.tight_layout()
            st.pyplot(fig)

        # Time Frequency Plot
        st.markdown(f"### Time Frequency Plot ({freq_option})")
        if date_cols:
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
            freq_fig = px.bar(freq_df, x=freq_option, y='Counts', color=freq_df[freq_option],
                              color_discrete_sequence=pastel_rainbow,
                              template='plotly_white', title=f"Incident Frequency by {freq_option} ({title_prefix})")
            freq_fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(color='black'),
                xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
                yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
                showlegend=False
            )
            st.plotly_chart(freq_fig, use_container_width=True)

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
