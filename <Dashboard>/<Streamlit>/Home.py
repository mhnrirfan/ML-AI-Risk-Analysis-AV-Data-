# Home.py

"""
HOW TO RUN
cd "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Dashboard>/<Streamlit>"
streamlit run Home.py
"""
# Home.py
import streamlit as st
import pandas as pd
import os
from PIL import Image
import streamlit as st
from st_flexible_callout_elements import flexible_error, flexible_success, flexible_warning, flexible_info
from Functions.py import load_csv, prep_dates, fig_missingness, fig_boxplot
from Functions.py import get_imputation_df, plot_bar_side_by_side, plot_line_side_by_side,imputer_overall_summary



# ---------------- Load Data ----------------
UK_data = load_csv("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK-cleaned_data.csv")
US_data = load_csv("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/US-cleaned_data.csv")


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
    with st.sidebar:
        
        dataset_choice = st.radio("Select Dataset", options=['UK', 'US'])
        st.header("EDA Settings")
        top_n = st.slider("Top N Categories", min_value=2, max_value=30, value=8)

        if dataset_choice == 'US':
            cat_options = ['Make', 'Model', 'ADS Equipped?', 'Automation System Engaged?', 'City', 'State', 
                           'Roadway Type', 'Roadway Surface', 'Lighting', 'Crash With', 
                            'SV Pre-Crash Movement', 'SV Contact Area', 
                           'Weather']
            numeric_options = ["Posted Speed Limit (MPH)", "Report Version","Model Year"]
        else:
            cat_options = ['Make', 'Model', 'City', 'State', 'Roadway Type', 'Roadway Surface', 'Lighting', 
                           'Crash With', 'SV Pre-Crash Movement', 
                           'SV Contact Area', 'Weather']
            numeric_options = ["Posted Speed Limit (MPH)", "Report Version","Model Year"]

        severity_col_choice = st.selectbox("Select Category for Severity Analysis", options=cat_options)
        numeric_col = st.selectbox("Select Numeric Column for Boxplot/KDE", options=numeric_options)
        freq_option = st.selectbox("Select Time Unit for Frequency Plot", options=['Day', 'Month', 'Year'])
        # ----- for supervised
        st.header("Supervised Settings")
        supervised_options = ["View Data", "Imputers","Hyperparameters","Results"]
        supervised_col = st.selectbox("Select Process", options=supervised_options)
        model_col = ["Hyperparameters", "Results", "Explainability"]
        algo_col = ["Decision Tree", "Random Forest", "XGBoost", "Logistic Regression"]


        # Only show the selectbox if the user has chosen a relevant supervised_col
        if supervised_col in model_col:
            chosen_model = st.selectbox("Choose Model", options=algo_col)
        
        st.header("Clustering Setting")
        clustering_options = ["View Clustered Data", "K-Means","PCA","TSNE"]
        clustering_col = st.selectbox("Select Process", options=clustering_options)

st.title("🚗 ML-AI Risk Analysis Dashboard")
# ---------------- Tabs Navigation ----------------
tabs = st.tabs([
    "🏠 Home Page",
    "📄 Dataset Information",
    "📊 Exploratory Data Analysis",
    "⚙️ Supervised Learning",
    "📈 Experimental Clustering",
])

# ---------------- Home Tab ----------------
with tabs[0]:
    st.markdown("<h2 style='color:#1f77b4;'>Welcome!</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        This **Autonomous Vehicle Incident Analysis Dashboard** allows you to explore AV incident data
        through multiple analytical lenses. Dive into risk analysis, model explainability, and insightful visualizations.
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h3 style='color:#ff7f0e;'>👩🏻‍🦱 About Me</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        - **Name:** Mahnoor Iqbal  
        - **University:** University of Liverpool  
        - **Background:** BEng Civil Engineering  
        - **Current:** Final semester MSc Data Science and AI student  
        - **Projects & Notebooks:** All project Jupyter notebooks and resources can be found on [GitHub](https://github.com/mhnrirfan)  
        """,
        unsafe_allow_html=True
    )
    st.subheader("Dashboard Sections")
    col1, col2, col3, col4, col5 = st.columns(5)

    sections = [
        ("🏠 Home Page", "Introduction to Project and Motivation.", "rgba(247, 193, 255, 0.7)"),
        ("📄 Dataset", "Quick overview of the dataset, missing values, and key stats.", "rgba(255, 221, 193, 0.7)"),
        ("📊 Exploratory Data Analysis (EDA)", "Visualizations and patterns to understand the data.", "rgba(193, 225, 255, 0.7)"),
        ("📈 Clustering", "Group incidents using unsupervised learning techniques.", "rgba(212, 255, 193, 0.7)"),
        ("⚙️ Supervised Learning", "Predictive modeling and evaluation for risk analysis.", "rgba(255, 250, 193, 0.7)")
    ]

    for col, (title, desc, color) in zip([col1, col2, col3, col4, col5], sections):
        col.markdown(
            f'<div style="background-color:{color}; color:black; padding:15px; border-radius:5px; margin-bottom:10px; height:150px">'
            f'<b>{title}</b><br>{desc}'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("<h2 style='color:#9467bd;'>🚗 About this Project: Risk Analysis of Autonomous Vehicle Accidents</h2>", unsafe_allow_html=True)
        # 5 images side by side
    img_urls = [
        "https://images.unsplash.com/photo-1485463611174-f302f6a5c1c9?w=500",
        "https://plus.unsplash.com/premium_photo-1741723515540-16a4e71b7d49?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8Y2FyJTIwJTIwYXV0b25vbW91cyUyMHZlaGljbGUlMjBpbWFnZXN8ZW58MHx8MHx8fDA%3D",
        "https://plus.unsplash.com/premium_photo-1741466913893-f66b4100cb0d?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8OXx8YWklMjBjYXJ8ZW58MHx8MHx8fDA%3D",
        "https://plus.unsplash.com/premium_photo-1741723515483-363a0857707b?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1pbi1zYW1lLXNlcmllc3wxfHx8ZW58MHx8fHx8",
        "https://images.unsplash.com/photo-1685984352141-2d66e772b7b1?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTV8fGNhciUyMGF1dG9ub21vdXN8ZW58MHx8MHx8fDA%3D"
    ]

    cols = st.columns(5)
    for c, img_url in zip(cols, img_urls):
        c.image(img_url, use_container_width=True)
    # Background & Context
    st.markdown("<h3 style='color:#1f77b4;'>🌍 Background & Context</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='background-color:#f2f2f2; padding:15px; border-radius:5px'>
        - Transportation evolution: footpaths → railways → aviation.<br>
        - Road safety is critical: <b>~1.19M deaths annually</b> ⚠️<br>
        - Passenger vehicles have far higher accident rates than buses, trains, or planes.<br>
        - Human error causes <b>up to 94% of accidents</b> (speeding, DUI, delayed reactions).<br>
        - <b>Autonomous Vehicles (AVs)</b> aim to reduce accidents with:<br>
            - 🛑 Automatic Emergency Braking (AEB)<br>
            - 🚦 Adaptive Cruise Control (ACC)<br>
            - ↔️ Lane switching & Automated Parking Systems (APS)<br>
            - 📸 Sensors: Cameras, LiDAR, Radar, ultrasonic
        </div>
        """, unsafe_allow_html=True
    )
    # Research Gaps
    st.markdown("<h3 style='color:#ff7f0e;'>🔍 Research Gaps</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='background-color:#f2f2f2; padding:15px; border-radius:5px'>
        - Limited <b>real-world AV crash data</b>; many studies use synthetic datasets.<br>
        - Comparison needed: <b>ADS (fully autonomous)</b> vs <b>ADAS (assisted driving)</b>.<br>
        - <b>SAE Levels of Automation</b>:<br>
            - 0️⃣  Conventional vehicle<br>
            - 1️⃣–3️⃣ ADAS: some automation, driver in control<br>
            - 4️⃣–5️⃣ ADS: fully autonomous, no driver
        </div>
        """, unsafe_allow_html=True
    )
    # Key Steps
    st.markdown("<h3 style='color:#9467bd;'>🛠️ Key Steps</h3>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background-color:#f2f2f2; padding:15px; border-radius:5px'>
        <b>1. 📚 Literature Review </b>– AV features, accident causes, autonomy levels, ML techniques
       <br><b>2. 🧹 Data Cleaning</b> – remove outliers, impute missing values, normalize
        <br><b>3. 📊 Exploratory Data Analysis</b> – distributions, correlations, patterns
       <br><b> 4. 🤖 Supervised & Unsupervised Models</b> – implement and evaluate
       <br><b> 5. 🔍 Explainable AI</b> – SHAP & LIME to interpret results
       <br><b> 6. 📈 Dashboard</b> – summarize findings with insights & recommendations
        </div>
        """, unsafe_allow_html=True)
    # Project Aim
    st.markdown("<h3 style='color:#d62728;'>🎯 Project Aim</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='background-color:#f2f2f2; padding:15px; border-radius:5px'>
        - Risk analysis of <b>real-world AV accident data</b> (ADS vs ADAS).<br>
        - Apply <b>supervised & unsupervised learning</b> to:<br>
            - Identify accident causes & severity factors<br>
            - Evaluate with confusion matrices, precision & recall<br>
            - Provide insights through an <b>interactive dashboard</b><br>
            - Employ <b>XAI methods</b> to interpret model decisions
        </div>
        """, unsafe_allow_html=True)
    # Methodology & Explainability
    st.markdown("<h3 style='color:#2ca02c;'>🧠 Methodology & Explainability</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='background-color:#f2f2f2; padding:15px; border-radius:5px'>
        - <b>ML models predict accident severity:</b><br>
            - 🌳 Decision Tree<br>
            - 🌲 Random Forest<br>
            - 💡 XGBoost<br>
            - 📈 Logistic Regression<br>
        - 🧩 Clustering Methods<br>
        - Black-box models lack transparency → risky for public safety.<br>
        - <b>XAI techniques:</b><br>
            - 🔹 SHAP (SHaply Additive exPlanations)<br>
            - 🔹 LIME (Local Interpretable Model-agnostic Explanations)
        </div>
        """, unsafe_allow_html=True)


with tabs[1]:
    st.subheader("📁 Dataset Information")

    if dataset_choice == "UK":
        st.markdown("""
        The UK Road Casualty Dataset is based on **STATS19 police-reported personal injury collisions**.  
        - Annual & provisional statistics available  
        - Open dataset: [data.gov.uk](https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-accidents-safety-data)  
        - Detailed data guide: [DfT Road Casualty Statistics Data Guide 2024](https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2024.xlsx)  
        """)
    else:
        st.markdown("""
        The US NHTSA SGO Dataset contains crash data for vehicles with ADS/ADAS systems.  
        - NHTSA dataset: [NHTSA SGO Dataset](https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting)  
        - Data element definitions: [SGO 2021-01 Data Element Definitions PDF](https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/Archive-2021-2025/SGO-2021-01_Data_Element_Definitions.pdf)  
        """)

        
    with st.expander("📝 Column Definitions", expanded=False):
            st.markdown("""
            ## Dataset Columns
            Notes: Both datasets have different columns however named in different ways or merged upon several columns or datasets hence here a unified column list 
            is made to accurate compare the dataset with clarity using python scripts listed below:
            Defintions of each columns:
            - Report ID: Unique Identifier of the Accident
            - Report Version: Version kept as can be multiple reports made adding more information to the incident
            - Make: Make of the vehicle
            - Model: Model of the Make
            - Model Year: Vehicles orginal release year
            - ADS Equipped?: Whether ADAS (Driver assisted), ADS (Fully Autonomous) or Conventional vehicle
            - Automation System Engaged?: ADS engaged or not
            - Incident Date: Date in DD/MM/YYYY format
            - Incident Time (24:00): Time in HH:MM:SS
            - City: City Location
            - State: State (and for clarity: England, Scotland, Ireland, Wales in UK data)
            - Roadway Type: Type of road eg: freeway, dual carriageaway, street etc
            - Roadway Surface: Wet, Icy, Slush, Clear etc
            - Posted Speed Limit (MPH): Speed Limit listed on Road
            - Lighting: Level of light: daylight, foggy, dark, dusk etc
            - Crash With: Other item involved eg: Passenger, Car, Pole, Tree etc
            - Highest Injury Severity Alleged: eg: No injury, Severe, Moderate, Fatality etc
            - SV Pre-Crash Movement: What vehicle was doing eg: proceesing straight, turning left, parked
            - Weather: Cloudy, Raining, Clear, Fine winds etc
            - SV Contact Area: Area hit on the vehicle
            - Country: US or UK
            """)

    # -----------------------
    # Data Limitations
    # -----------------------
    with st.expander("⚠️ Data Limitations", expanded=False):
        if dataset_choice == "UK":
            st.markdown("""
            - Minor injuries may be underreported  
            - Subjective officer judgment in some fields  
            - Sensitive info withheld (e.g., exact location, breath test results)  
            - Some fields may be missing or redacted
            """)
        else:
            st.markdown("""
            - Reporting differences across manufacturers  
            - Some fields redacted (confidential / PII)  
            - Data may be incomplete or unverified  
            - Same crash may appear multiple times
            """)

    # -----------------------
    # Toggle to show Python cleaning code
    # -----------------------
    with st.expander("🧹 Show Data Cleaning Script", expanded=False):
        if dataset_choice == "UK":
            with open("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Python Scripts>/UK_Cleaning.py") as f:
                code = f.read()
            st.code(code, language="python")
        else:
            with open("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Python Scripts>/US_Cleaning.py") as f:
                code = f.read()
            st.code(code, language="python")

    if dataset_choice == "UK":
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                "https://media.istockphoto.com/id/2148752786/photo/car-crash-with-two-vehicles-collided-at-traffic-accident-site-on-american-street.jpg?s=612x612&w=0&k=20&c=KSagw-918ODp_uoACRnYTSifKA6GR3yhqGb9sgpdwoM=",
                use_container_width=True
            )
        with col2:
            st.image(
                "https://media.istockphoto.com/id/1398986965/photo/heavy-traffic-jam-next-to-bus-lane-in-england-uk.webp?a=1&b=1&s=612x612&w=0&k=20&c=HP8QNQc7axASGywLluGWC8T21s9bfgxeKkaZlhW-6tY=",
                use_container_width=True
            )
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                "https://media.istockphoto.com/id/523830845/photo/traffic-mishap.webp?a=1&b=1&s=612x612&w=0&k=20&c=kTtaUOZfzanT3W0qq_G5kk2dpnvhUoPoGQAKfgceUpI=",
                use_container_width=True
            )
        with col2:
            st.image(
                "https://media.istockphoto.com/id/1446301560/photo/modern-vehicle-with-ai-assisted-sensors-for-movement.webp?a=1&b=1&s=612x612&w=0&k=20&c=CtPWpGPlua1LTTPts5rCm50A0fxqDU6BxMaw1noIVTI=",
                use_container_width=True
            )
    
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
    # ---------------- Sidebar Settings ----------------
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
            with st.expander("ℹ️ Insights"):
                st.write("**UK** Vehicles are all conventional and not stated as ADS/ADAS within data collection pack as hard to know if jaguar is a ads or normal even if model is given ")
                st.write("**US:** Only ADAS and ADS vehicles, therefore smaller dataset size ")
            chart_height = UK_TOP_CHART_HEIGHT if dataset_choice == 'UK' else TOP_CHART_HEIGHT
            if dataset_choice == 'UK':
                plot_adas_ads_pie(UK_data, "UK", st, chart_height=chart_height)
            elif dataset_choice == 'US':
                plot_adas_ads_pie(US_data, "US", st, chart_height=chart_height)

        # --- Severity Donut Chart ---
        with col2:
            st.markdown("**Injury Severity Distribution**")
            with st.expander("ℹ️ Insights"):
                st.write("**UK** Collsions are classed as minor whilst which mean no injury reported in not collected or classed as minor")
                st.write("**US:** Not each accident ends in an injury only 16.9% which makes an already small 4353 dataset even smaller for targeted injury only accidents")
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
            with st.expander("ℹ️ Insights"):
                st.write("**UK** Much Clearer pattern with schooltime/work rush and Post work rush shwoing accidents occuring due to busy traffic and signalling through these areas")
                st.write("**US:** Generally Higher for Peak times (Morning, Lunch and After work) suprising high for 9pm to 12pm which can still be busier in cities however as the drop from 12am to 6am it is unlikely accidents due to lighting is occuring")
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
        if dataset_choice =="UK":
            with st.expander("ℹ️ Insights"):
                st.markdown(
                    """
            - **Make and Model**: Tesla, Jaguar and Ford have the highest fatality rate, whilst Jaguar and Cruise have the most non-injury crashes.  
            This shows the high proportion of Jaguar vehicles in this dataset.  

            - **ADS Equipped?**: Whether ADAS (Driver Assisted), ADS (Fully Autonomous), or Conventional vehicle  

            - **Automation System Engaged?**: Whether ADS was engaged or not  

            - **City**: City location  

            - **State**: State (and for clarity: England, Scotland, Ireland, Wales in UK data)  

            - **Roadway Type**: Type of road e.g., freeway, dual carriageway, street, etc.  

            - **Roadway Surface**: Wet, Icy, Slush, Clear, etc.  

            - **Lighting**: Level of light: daylight, foggy, dark, dusk, etc.  

            - **Crash With**: Other item involved e.g., Passenger, Car, Pole, Tree, etc.  

            - **SV Pre-Crash Movement**: What vehicle was doing e.g., proceeding straight, turning left, parked  

            - **Weather**: Cloudy, Raining, Clear, Fine winds, etc.  

            - **SV Contact Area**: Area hit on the vehicle  

            - **Country**: US or UK  
            """
                )

        else:
            with st.expander("ℹ️ Insights"):
                st.markdown(
                        """
                - **Make and Model**: Tesla, Jaguar and Ford have the highest fatality rate, whilst Jaguar and Cruise have the most non-injury crashes.  
                This shows the high proportion of Jaguar vehicles - Ipace which have most severe crashes and least severe crashes.

                - **City and State**: This dataset only have 4 fatalities and one is from Sanfransisco but the rest are nan which is suprising given the importance of those cases as given if state is known the city, road and loaction is neccessary for analysis
                Other than California, Arizona and Texas also have higher accident rates compared to the other states

                - **State**: State (and for clarity: England, Scotland, Ireland, Wales in UK data)  

                - **Roadway Type**: Type of road e.g., freeway, dual carriageway, street, etc.  

                - **Roadway Surface**: Wet, Icy, Slush, Clear, etc.  

                - **Lighting**: Level of light: daylight, foggy, dark, dusk, etc.  

                - **Crash With**: Other item involved e.g., Passenger, Car, Pole, Tree, etc.  

                - **SV Pre-Crash Movement**: What vehicle was doing e.g., proceeding straight, turning left, parked  

                - **Weather**: Cloudy, Raining, Clear, Fine winds, etc.  

                - **SV Contact Area**: Area hit on the vehicle  

                - **Country**: US or UK  
                """
                    )
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
        with st.expander("ℹ️ Insights"):
                st.write("**UK** No Missing data for these columns (as dropped the columns due to only 0.1% being missing values) ")
                st.write("""**US:** High missing value especially for the severity which is the main target value for the machine learning models therefore must be imputed with high accuracy or the dataset will reduce by almost 50%
                            other categorical data can be imputed easily based on data correlations available eg: road surface based on timing, lighting based on time, state based on city""")
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
        with st.expander("ℹ️ Insights"):
                st.write("**UK** Higher Number of accidents occuring in Birmingham which is suprising given how low London is, this can be due to ")
                st.write("**US:** Highest in California as epicentre of autonomous vehicle creation (San Francisco) however every state does have values but far smaller representative showing the bias towards californian conditions")
        if dataset_choice == 'UK':
            
            st.image(
                "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/EDA_FE/UK_incidents_choropleth.png",
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
        with st.expander("ℹ️ Insights"):
            st.write("""**UK** Evenly Distributed for day with peak on friday and suprisingly a dip on the weekend", Additionally the summer and autumn months are higher, the highest number of accidents being in 2019 and decreasing during covid years, even in 2023 there is still a decrease however this can be due to the rise of autonomous features deployed in vehicles on the road but also the reduce reporting""")
            st.write("""**US** Extremely high peak in the years for 2025 however this can be due greater reporting and more vehicles on the road instead of due to distribution, for us data summer months are lower compared to december and april, and similalrh the weekend is supring lower than the start of the week """)
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
    if dataset_choice == "UK":
        with st.expander("ℹ️ Insights"):
                st.write(""" 
                - **Posted Speed Limit**: Within this dataset majority of the crashes have occured between 30/40 mph on main roads however outliers such as 60 and 70mphs are 
                suggested with the boxplot method however these are rational accidents which could have occured on the motorway or due to dangerous overspeeding but this also shows 
                less accidents occur on motorways thus roadway type is a big factor in accidents
                - **Model Year**: some outliers can be seen which show that vehicles from the 1940-1995s have in accidents however 
                these should not be removed as the UK law allows for older vehicles to be allowed on the road with MOTs and the number 
                of these vehicles are quite small given the dataset being 290,000+
                """)
    else:
        with st.expander("ℹ️ Insights"):
            st.write(""" 
                - **Report Version**: Most accidents reports only have 1/2 versions however outliers shows that there are accidents that have been amended and added on up to 9 times, whilst this 
                is not a major issue it can show the data collection process can be problematic and given the data qualty and missingness it can affect the data analysis
                - **Posted Speed Limit**: Unlike the UK, the speeds are more spread with higher speeds also having accidents however US infrastructure is likely to have more highways and freeways increasing the chance 
                of an accident occuring (as dataset is drom National Highway Safety Transport Administration)
                - **Model Year**: most vehicles are between 2014-2026 which is reasonable given AVs are relatively new compared to traditional vehicles, additionally model year 2026 is also being rolled out, 
                highest accidents occur for models 2020-2023 which is due to how many AVs have been released and how long they have been on the road
                """)
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
        with st.expander("ℹ️ Insights"):
            st.write("""For UK data, the correlations much lower due to the size of the dataset creating more reliable correlations)
            - Common correlations such as Make/Model, Lighting/Incident Time, Roadsurface/Weather are higher as excepted (these can be factchecked in severity distrubtion by category plot)
            - The data also shows postive correlations for severity with where the vehicle was hit, type of road, which state(country eg: england), and time suggesting accidents in england are higher than other countries and severity can increase based on how damaged the vehicle got
            - Additionally the negative correlations with lighting, city and precrash movementas these can increasing the severity""")
        col1, col2 = st.columns(2)
        with col1:
            st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/EDA_FE/UK_heatmap.png", use_container_width=True)
        with col2:
            st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/EDA_FE/UK_top_corr_bar.png", use_container_width=True)
    
    
    elif dataset_choice == 'US':
        st.markdown("**Top Correlations with Severity**")
        with st.expander("ℹ️ Insights"):
            st.write("""For US data, the correlations are much higher compared to the UK dataset and this is due to the smaller datasize and dependent on 2025 data (up to May)
            creating higher correlations
            - Common correlations such as ADAS/ADS, Make/Model, Lighting/Incident Time, Roadsurface/Weather are high as excepted (these can be factchecked in severity distrubtion by category plot)
            - The data also shows postive correlations with ADAS, Model and Incident Year suggesting the severity can be reduced based on ADAS or ADS selection, and vehicle age and roadway type such as motorways have less accidents than streets
            - Additionally the negative correlations with Speed limit, Make, Surface can increase chances of an accident given higher speed limits, wetter surfaces and certain models have are where more severe accidents occur""")
        col1, col2 = st.columns(2)
        with col1:
            st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/EDA_FE/US_heatmap.png", use_container_width=True)
        with col2:
            st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/EDA_FE/US_top_corr_bar.png", use_container_width=True)


    # ---------------- Clustering Tab ----------------
import pandas as pd
import streamlit as st

# Load datasets once
UK_data = pd.read_csv("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK-cleaned_data.csv")
US_data = pd.read_csv("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/US_imputed_data.csv")

with tabs[3]:
    st.subheader("⚙️ Supervised Learning")
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
            with st.expander("ℹ️ Insights"):
                st.write("""
                - Whilst some imputers such as  mode and last observation carried forward as easy to conduct 
                especially for categorical datasets they are far behind the random forest and XGBoost with most imputing accuracy 
                less than 30% which shows the importance of avoiding mode imputation as an easy way out
                - ML models use the rest of the columns to learn relationships between the columns to impute with an higher accuracy 
                for instance, if city is known then state can be found too
                - The highest performing model on average is XGBoost, whilst random forest may be slightly higher for some columns it does not 
                perform as well for multivalue columns such as contact area which contains "front, back, left" at the same row and due to this 
                XGBoost performs much better for jaccard index (where order of values doesn't matter) creating a higher average of both accuracy and jaccard index
                thus used as the final imputation statergy
                """)
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
        with st.expander("ℹ️ Insights"):
            st.write("""
            - This section shows which hyperparamters where utlised in RandomSearchCv hyperparameter testing, in this case 
            random combinations where chosen of hyperparameters based on best performance
            - Whilst grid search would be more reliable testing every single combination it required a significant more amount of runtime and computational resource
            - Note: Best value is the combination with the other parameters""")
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
            "precision": [0.54, 0.24, 0.12, 0.88, 0.31, 0.71, 0.42, 0.76],
            "recall": [0.65, 0.37, 0.17, 0.77, 0.48, 0.71, 0.49, 0.71],
            "f1-score": [0.59, 0.29, 0.14, 0.82, 0.38, 0.71, 0.44, 0.73],
            "support": [95, 60, 12, 607, 33, 807, 807, 807]
        }, index=["0", "1", "2", "3", "4", "accuracy", "macro avg", "weighted avg"])

        dt_us_val_report_df = pd.DataFrame({
            "precision": [0.52, 0.28, 0.10, 0.91, 0.24, 0.71, 0.41, 0.78],
            "recall": [0.65, 0.47, 0.17, 0.78, 0.38, 0.71, 0.49, 0.71],
            "f1-score": [0.57, 0.35, 0.12, 0.84, 0.29, 0.71, 0.44, 0.74],
            "support": [48, 30, 6, 304, 16, 404, 404, 404]
            }, index=["0", "1", "2", "3", "4", "accuracy", "macro avg", "weighted avg"])

        rf_us_test_report_df = pd.DataFrame({
            "precision": [0.62, 0.39, 0.60, 0.89, 0.57, 0.80, 0.61, 0.80],
            "recall": [0.84, 0.38, 0.25, 0.87, 0.24, 0.80, 0.52, 0.80],
            "f1-score": [0.71, 0.39, 0.35, 0.88, 0.34, 0.80, 0.53, 0.79],
            "support": [95, 60, 12, 607, 33, 807, 807, 807]
        }, index=["0", "1", "2", "3", "4", "accuracy", "macro avg", "weighted avg"])

        rf_us_val_report_df = pd.DataFrame({
            "precision": [0.60, 0.52, 0.50, 0.89, 0.33, 0.81, 0.57, 0.80],
            "recall": [0.73, 0.53, 0.17, 0.89, 0.19, 0.81, 0.50, 0.81],
            "f1-score": [0.66, 0.52, 0.25, 0.89, 0.24, 0.81, 0.51, 0.80],
            "support": [48, 30, 6, 304, 16, 404, 404, 404]
        }, index=["0", "1", "2", "3", "4", "accuracy", "macro avg", "weighted avg"])

        xgb_us_test_report_df = pd.DataFrame({
            "precision": [0.69, 0.36, 0.50, 0.89, 0.63, 0.82, 0.62, 0.81],
            "recall": [0.85, 0.35, 0.25, 0.89, 0.36, 0.82, 0.54, 0.82],
            "f1-score": [0.76, 0.36, 0.33, 0.89, 0.46, 0.82, 0.56, 0.81],
            "support": [95, 60, 12, 607, 33, 807, 807, 807]
        }, index=["0", "1", "2", "3", "4", "accuracy", "macro avg", "weighted avg"])

        xgb_us_val_report_df = pd.DataFrame({
            "precision": [0.62, 0.50, 0.33, 0.90, 0.40, 0.81, 0.55, 0.81],
            "recall": [0.69, 0.50, 0.33, 0.90, 0.25, 0.81, 0.53, 0.81],
            "f1-score": [0.65, 0.50, 0.33, 0.90, 0.31, 0.81, 0.54, 0.81],
            "support": [48, 30, 6, 304, 16, 404, 404, 404]
        }, index=["0", "1", "2", "3", "4", "accuracy", "macro avg", "weighted avg"])

        lr_us_test_report_df = pd.DataFrame({
            "precision": [0.37, 0.19, 0.03, 0.88, 0.15, 0.48, 0.32, 0.73],
            "recall": [0.53, 0.40, 0.33, 0.48, 0.39, 0.48, 0.43, 0.48],
            "f1-score": [0.43, 0.26, 0.06, 0.62, 0.21, 0.48, 0.32, 0.55],
            "support": [95, 60, 12, 607, 33, 807, 807, 807]
        }, index=["0", "1", "2", "3", "4", "accuracy", "macro avg", "weighted avg"])

        lr_us_val_report_df = pd.DataFrame({
            "precision": [0.43, 0.22, 0.02, 0.83, 0.14, 0.49, 0.32, 0.70],
            "recall": [0.54, 0.33, 0.17, 0.51, 0.38, 0.49, 0.39, 0.49],
            "f1-score": [0.48, 0.26, 0.03, 0.63, 0.20, 0.49, 0.32, 0.56],
            "support": [48, 30, 6, 304, 16, 404, 404, 404]
        }, index=["0", "1", "2", "3", "4", "accuracy", "macro avg", "weighted avg"])
    
# --------uk classfication report 
        dt_uk_test_report_df = pd.DataFrame({
            "precision": [0.04, 0.79, 0.22, 0.58, 0.35, 0.66],
            "recall": [0.25, 0.67, 0.25, 0.58, 0.39, 0.58],
            "f1-score": [0.07, 0.73, 0.24, 0.58, 0.34, 0.61],
            "support": [283, 14238, 3862, 18383, 18383, 18383]
        }, index=["0", "1", "2", "accuracy", "macro avg", "weighted avg"])

        dt_uk_val_report_df = pd.DataFrame({
            "precision": [0.04, 0.78, 0.21, 0.58, 0.34, 0.65],
            "recall": [0.26, 0.68, 0.23, 0.58, 0.39, 0.58],
            "f1-score": [0.07, 0.73, 0.22, 0.58, 0.34, 0.61],
            "support": [142, 7119, 1931, 9192, 9192, 9192]
        }, index=["0", "1", "2", "accuracy", "macro avg", "weighted avg"])

        rf_uk_test_report_df = pd.DataFrame({
            "precision": [0.04, 0.78, 0.24, 0.69, 0.35, 0.66],
            "recall": [0.04, 0.85, 0.16, 0.69, 0.35, 0.69],
            "f1-score": [0.04, 0.81, 0.19, 0.69, 0.35, 0.67],
            "support": [283, 14238, 3862, 18383, 18383, 18383]
        }, index=["0", "1", "2", "accuracy", "macro avg", "weighted avg"])

        rf_uk_val_report_df = pd.DataFrame({
            "precision": [0.03, 0.78, 0.25, 0.70, 0.35, 0.66],
            "recall": [0.04, 0.86, 0.16, 0.70, 0.35, 0.70],
            "f1-score": [0.03, 0.82, 0.19, 0.70, 0.35, 0.67],
            "support": [142, 7119, 1931, 9192, 9192, 9192]
            }, index=["0", "1", "2", "accuracy", "macro avg", "weighted avg"])

        xgb_uk_test_report_df = pd.DataFrame({
            "precision": [0.04, 0.79, 0.24, 0.66, 0.36, 0.66],
            "recall": [0.14, 0.80, 0.19, 0.66, 0.38, 0.66],
            "f1-score": [0.06, 0.79, 0.21, 0.66, 0.36, 0.66],
            "support": [283, 14238, 3862, 18383, 18383, 18383]
            }, index=["0", "1", "2", "accuracy", "macro avg", "weighted avg"])

        xgb_uk_val_report_df = pd.DataFrame({
            "precision": [0.04, 0.79, 0.23, 0.66, 0.35, 0.66],
            "recall": [0.16, 0.80, 0.17, 0.66, 0.38, 0.66],
            "f1-score": [0.07, 0.79, 0.20, 0.66, 0.35, 0.66],
            "support": [142, 7119, 1931, 9192, 9192, 9192]
        }, index=["0", "1", "2", "accuracy", "macro avg", "weighted avg"])

        lr_uk_test_report_df = pd.DataFrame({
            "precision": [0.03, 0.79, 0.21, 0.46, 0.34, 0.66],
            "recall": [0.38, 0.50, 0.32, 0.46, 0.40, 0.46],
            "f1-score": [0.06, 0.61, 0.25, 0.46, 0.31, 0.53],
            "support": [283, 14238, 3862, 18383, 18383, 18383]
        }, index=["0", "1", "2", "accuracy", "macro avg", "weighted avg"])

        lr_uk_val_report_df = pd.DataFrame({
            "precision": [0.04, 0.79, 0.20, 0.45, 0.34, 0.65],
            "recall": [0.44, 0.49, 0.31, 0.45, 0.41, 0.45],
            "f1-score": [0.07, 0.60, 0.25, 0.45, 0.31, 0.52],
            "support": [142, 7119, 1931, 9192, 9192, 9192]
        }, index=["0", "1", "2", "accuracy", "macro avg", "weighted avg"])

 
 
        dt_us_test_report_df = dt_us_test_report_df.copy()
        dt_us_val_report_df = dt_us_val_report_df.copy()
        dt_uk_test_report_df = dt_uk_test_report_df.copy()
        dt_uk_val_report_df = dt_uk_val_report_df.copy()

        rf_us_test_report_df = rf_us_test_report_df.copy()
        rf_us_val_report_df = rf_us_val_report_df.copy()
        rf_uk_test_report_df = rf_uk_test_report_df.copy()
        rf_uk_val_report_df = rf_uk_val_report_df.copy()

        xgb_us_test_report_df = xgb_us_test_report_df.copy()
        xgb_us_val_report_df = xgb_us_val_report_df.copy()
        xgb_uk_test_report_df = xgb_uk_test_report_df.copy()
        xgb_uk_val_report_df = xgb_uk_val_report_df.copy()

        lr_us_test_report_df = lr_us_test_report_df.copy()
        lr_us_val_report_df = lr_us_val_report_df.copy()
        lr_uk_test_report_df = lr_uk_test_report_df.copy()
        lr_uk_val_report_df = lr_uk_val_report_df.copy()

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
            "US Test": [0.705081, 0.799257, 0.817844, 0.475836],
            "US Validation": [0.712871, 0.806931, 0.811881, 0.490099],
            "UK Test": [0.576239, 0.692488, 0.657782, 0.461513],
            "UK Validation": [0.576480, 0.697237, 0.655461, 0.453220]
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
            if chosen_model == "Decision Tree" and dataset_choice == "UK":
                with st.expander("ℹ️ Insights"):
                    st.write("""
                    - Decision Tree as seen in the clustering explainabilty works by recursively splitting data based on features based on decisions made by the branches splitting data into leaves. 
                    - Given the hyperparamter testing this decision tree algorithm works by gini impurity which dictates how mixed the data is in the group
                    - The maxmium depth is also chosen as 10 splits which ensures a reduction od overfitting
                    - **Advantages**: They are easy to understand and have fast speeds given the amount of data available 
                    - **Disadvantages**: However they are sensitive to variables, and overfit meaning they perform worse to unseen data thus increasing inaccuracy

                    **Insights**                
                    - The general accuracy of both validation and training dataset is 0.58 which class 1 (minor) being the most accurate compared to the other classes this can also be due to it being majority class
                    - However the fatality and severe class is much lower showcasing the favouring as 138 out of 238 is misclassfied for class 0 (fatality) and 2385 out of 3862  for class 2 (severe),
                    - This can be due to the limitation of only using 1 tree but even with SMOTE the scores is low.
                    """)
            elif chosen_model == "Random Forest" and dataset_choice == "UK":
                with st.expander("ℹ️ Insights"):
                    st.write("""
                    Random Forest uses multiple decision trees to reduce overfitting and increase accuracy as it combines their input
                    Hyperparameter 200/300 trees are used for the US and UK dataset with no maxmium limit on the depth which can be reduced for quicker runtime but randomsearchCV gave them to be the best parameters.
                    -**Advantages** it has higher accuracy and robust to outliers and noise and can capture complex relationships
                    -**Disadvantages**: however they can take an extremely long time to run (the longest in my jupiter notebook compared to other models)

                    **Insights**
                    - Random Forest is highest accuracy compared to the othe models with the least false postive and negatives even so the accuracy is 69.2% for training and 69.7% for validation
                    - Once again the accuracy of class 1 is much higher than the accuracy for class 0 and 2 showcasing this is a common issue as models are not learning as well with SMOTE
                    """)

            elif chosen_model == "XGBoost" and dataset_choice == "UK":
                with st.expander("ℹ️ Insights"):
                    st.write("""
                            XGBoost also uses trees however builds the trees sequentially by correcting the errors of the previous trees which make its much faster than random forest
                            Hyperparameter 0.1 is used as the learning rate and the max depth of each tree is 6 based on tuning which is the same for both dataset
                            -**Advantages** Often outperforms random forest models and contains regularisation to reduce overfitting using L1/L2 regularisation 
                            -**Disadvantages**: Not as easy to understand and explain and careful tuning is needed for the number of trees and step sizes for updating the predictions for learning rate

                            **Insights**
                            - Performs slightly worse than random forest for the UK dataset where the minority class for fatality is better predicted for both validation and testing dataset but class 2 performs much better in random forest than XGBoost
                            """)

            elif chosen_model == "Logistic Regression" and dataset_choice == "UK":
                with st.expander("ℹ️ Insights"):
                    st.write("""
                        Logistic Regression uses the logistic sigmoid equation to predict the probabilty of the point being a class and using parameters to find coefficients to best fit the data
                        Hyperparameter Elasticnet is used for the UK dataset which uses both L1 and L2 for regularisation
                        -**Advantages** Fast to train and coefficients are given for each feature to explain the model results, it also is simple to conduct and overfitting methods can be applied
                        -**Disadvantages**: Highly correlated features, and outliers can reduce the accuracy by alot compared to random forest and XGboost
                        
                        **Insights**
                            - Performs slightly worse than random forest for the UK dataset where the minority class for fatality is better predicted for both validation and testing dataset but class 2 performs much better in random forest than XGBoost
                            """)

            elif chosen_model == "Decision Tree" and dataset_choice == "US":
                with st.expander("ℹ️ Insights"):
                    st.write("""
                    - Decision Tree as seen in the clustering explainabilty works by recursively splitting data based on features based on decisions made by the branches splitting data into leaves. 
                    - Given the hyperparamter testing this decision tree algorithm works by gini impurity which dictates how mixed the data is in the group
                    - The maxmium depth is also chosen as 10 splits which ensures a reduction od overfitting
                    - **Advantages**: They are easy to understand and have fast speeds given the amount of data available 
                    - **Disadvantages**: However they are sensitive to variables, and overfit meaning they perform worse to unseen data thus increasing inaccuracy
                    
                    **Insights**
                    - Decision Trees perform very well with less false postives compared to logistic regression
                    - Classes 4 and 2 struggle with precision and due to less support and values it is harder to learn from and classify
                    """)
            elif chosen_model == "Random Forest" and dataset_choice == "US":
                with st.expander("ℹ️ Insights"):
                    st.write("""
                    Random Forest uses multiple decision trees to reduce overfitting and increase accuracy as it combines their input
                    Hyperparameter 200/300 trees are used for the US and UK dataset with no maxmium limit on the depth which can be reduced for quicker runtime but randomsearchCV gave them to be the best parameters.
                    -**Advantages** it has higher accuracy and robust to outliers and noise and can capture complex relationships
                    -**Disadvantages**: however they can take an extremely long time to run (the longest in my jupiter notebook compared to other models)
                   
                    **Insights**
                    - Whilst coming second to the XGboost models, it only performs slightly worse for all of the clusters and struggles more with class 4 compared to XGBoost
                    - Both RF and XGBoost perform exactly the same for fatality given a very small support of 12 
                    """)

            elif chosen_model == "XGBoost" and dataset_choice == "US":
                with st.expander("ℹ️ Insights"):
                    st.write("""
                            XGBoost also uses trees however builds the trees sequentially by correcting the errors of the previous trees which make its much faster than random forest
                            Hyperparameter 0.1 is used as the learning rate and the max depth of each tree is 6 based on tuning which is the same for both dataset
                            -**Advantages** Often outperforms random forest models and contains regularisation to reduce overfitting using L1/L2 regularisation 
                            -**Disadvantages**: Not as easy to understand and explain and careful tuning is needed for the number of trees and step sizes for updating the predictions for learning rate

                            **Insights**
                            -  XGboost is the best model with a great score of 81.7% for training and 81.2% for validation
                            - The key highlight is as it sequentially builds upon the previous tree whilst still having a high n-estimatior (trees) it learns from the previous tree derving greater results for the 3 most count classes
                                                        """)

            elif chosen_model == "Logistic Regression" and dataset_choice == "US":
                with st.expander("ℹ️ Insights"):
                    st.write("""
                    Logistic Regression uses the logistic sigmoid equation to predict the probabilty of the point being a class and using parameters to find coefficients to best fit the data
                    Hyperparameter Elasticnet is used for the UK dataset which uses both L1 and L2 for regularisation
                    -**Advantages** Fast to train and coefficients are given for each feature to explain the model results, it also is simple to conduct and overfitting methods can be applied
                    -**Disadvantages**: Highly correlated features, and outliers can reduce the accuracy by alot compared to random forest and XGboost

                    **Insights**
                    - Logistic Regression has preformed significantly worse than the rest of the models unable to pick up the complex relationships even when there is a majority class
                    - This is evident with even the majority class (3) having a much lower accuracy rate for both validation and testing 

                            """)


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


        st.title(f"**Explainability Plots of {chosen_model} for {dataset_choice}**")
        if chosen_model == "Decision Tree":
            with st.expander("ℹ️ Insights"):
                st.write("""
                - SHAP 
                - LIME 

                **UK Insights**                


                **US Insights**                
     
                """)







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
        import os
        from PIL import Image
        import streamlit as st

        def show_plot(title, img_path):
            if os.path.exists(img_path):
                st.markdown(f"<h3 style='text-align: center;'>{title}</h3>", unsafe_allow_html=True)
                img = Image.open(img_path)
                st.image(img, use_container_width=True)  # Image scales to container width
            else:
                st.warning(f"Image not found: {img_path}")


        # Show plots sequentially
        show_plot(f"SHAP BAR Plot of {chosen_model} for {dataset_choice}", shap_bar_path)
        show_plot(f"SHAP SUMMARY Plot of {chosen_model} for {dataset_choice}", shap_summary_path)
        show_plot(f"LIME Plot of {chosen_model} for {dataset_choice}", lime_path)

            # ---------------- Supervised Learning Tab ----------------

        
        import plotly.express as px
        import pandas as pd
        if dataset_choice == "US":
            st.title("US Model Accuracy Comparison")
            # Example dataframe: 4 models, Test & Validation accuracy
            accuracy_data = pd.DataFrame({
                "Model": ["Decision Tree", "Random Forest", "XGBoost", "Logistic Regression"],
                "Test": [0.705081, 0.799257, 0.817844, 0.475836],
                "Validation": [0.712871, 0.806931, 0.811881, 0.490099]
            })
        if dataset_choice == "UK":
            st.title("UK Model Accuracy Comparison")
            accuracy_data = pd.DataFrame({
                "Model": ["Decision Tree", "Random Forest", "XGBoost", "Logistic Regression"],
                "Test": [0.576239, 0.692488, 0.657782, 0.461513],
                "Validation": [0.576480, 0.697237, 0.655461, 0.453220]
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
    with tabs[4]:
        import streamlit as st
        from st_flexible_callout_elements import flexible_error, flexible_success, flexible_warning, flexible_info


        st.subheader("📈 Clustering")
        # Dataset-specific settings
        if dataset_choice == "UK":
            flexible_success("UK Silhouette Score = **0.121**  |  Cluster Distribution: (0: 1729, 1: 12414, 2: 4499, 3: 4334, 4: 3)", alignment="center")
            with st.expander("ℹ️ Table Insights"):
                    st.write("""
                                - There are 5 Clusters are chosen due to elbow method recommendation, here the mode can be found for each data cluster to enable cluster labelling 
                                - Cluster 1 is the largest and the only cluster with BMW as a model, given it is almost identical to cluster 2 and together makes up to 72% of the dataset, however given the elbow method and silohuette peaking at 5 it is important to experiment with differents K.
                                - The smallest cluster is clster 4 with only 3 values which is an outlier as the lattitude and longitude was 0,0 mapping the city country to Ghana making it useful to find noise and outliers using these methods 
                                - The other key features acting within this dataset split is city with Bradford, Birmingham and Cardiff across a variety of different times from 5-8pm.
                                - Additionally, cluster 3 is the only cluster split based on weather and wet road surface
                                """)
            if clustering_col == "View Clustered Data":
                csv_path = "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK_cluster_summary.csv"
                df = pd.read_csv(csv_path)
                st.dataframe(df.head())

            elif clustering_col == "K-Means":
                st.markdown("**Showing Optimal K for UK dataset**")
                with st.expander("ℹ️ Elbow Method Insights"):
                    st.write("""
                        - Two methods to determine the ideal clusters other than randomsearchCV and gridsearchCV is using the elbow method and silohuette score.
                        - The optimal k can be found using the part where inertia decreases steadily in with case around 3-5, however during this time it is important to balance it with the average  silohuette score which is mean distance to the closest cluster with scores close to 1 being well clustered, -1 to the incorrect cluster and 0 being close to the nearest boundary
                        - Here the scores are closer to one suggesting a lower need for clusters suggesting less meaningful differences between the clusters. 

                        - **For the UK dataset**, K=5 is chosen, while RandomSearch dicates 3/4 clusters, the silohuette score rapidly peaks at 5 which is a good indicator to experiment
                    """)

                st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/UK Dataset_plot.png")
                st.markdown("**Decision Trees showing Split for K=5**")
                with st.expander("ℹ️ Insights"):
                        st.write("""
                                - When using K-Means data is split into K number of cluster, we can adjust hyperparamters but we do not know why a split has occured and the percentages behind each feature. This is why a decision tree can be plotted to show the split and how each cluster is defined other than reading a table and worrying about how to label the clusters. 
                                - A decision tree works by splititng the data based on a condition (root node) and the more layers the more splits can be made which can help trace the cluster 

                                **UK Dataset Insights**
                                - Here each colour is a cluster with (orange:0),(Teal:2),(Green:1)(pink:4)(purple:3) 
                                - The first split is based on the model with 76.9% of vehicles being Ford Fiestas and the 23.1% being BMW and Rainy Weather
                                - Ford Fiestas are then split based on the state/country (given England and Wales) and then Rainy weather and wet road surface to find cluster 3, given cluster 3 is also found on the right side of the tree shows there is overlap
                                - Given the 4 layers of split the pink cluster 4 is approximated to 0.0% samples as 3/22979 is extremely small
                                - The diagram and feature importance also highlight little variance between roadway type, speed, contact area and movement
                                - Instead the time, date, model and model year are key contenders within the clustering 
                                - Given the high amount of minor accidents the overpowers the clustering modes and increasing the K will reduce the silouette hence clustering may not be as useful given the data is only 10% sampled 

                                **Highest Features**
                                Model: 0.3479
                                Roadway Surface: 0.2508
                                State: 0.1971
                                Weather: 0.1472
                                Lighting: 0.0568
                                Country: 0.0003

                                **Cluster Labelling**
                                Cluster 0: Ford Fiesta accidents in Wales for dry accidents
                                Cluster 1: Wet, Rainy BMW accidents within Bradford for newer accidents
                                Cluster 2: Ford Fiesta accidents in Bradford for dry accidents
                                Cluster 3: Ford Fiesta accidents in Birmingham for dry accidents
                                Cluster 4: Ghana Outliers
                            """)

                st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/UK Dataset_explainability.png")

            elif clustering_col == "PCA":
                st.markdown("### PCA Clustering Results for K=5")
                    # Single insights section for both plots
                with st.expander("ℹ️ Insights"):
                        st.write("""
                                    - PCA plots are visual methods to find clusters by reducing the data into principal components (2 for 2D and 3 for 3D) these components try ro find the mosrt variance in the data and help see how tightly packed the clusters can be seen 
                                    - While some points my be closer together other clusters can be classfied as outliers or how they could be part of a different cluster (with overlaps)

                                    **Insights**
                                    - The densely almost linear packed clusters can be shown However cluster 0 which is 1729 In both 2D and 3D is split across the entire dataset suggesting that these clusters can be merged with their corresponding parts making K=3 an ideal number however then Welsh accidents cluster will be removed 
                                    - Clear cluster between the green, teal and blue can be seen, based on city as blue and teal (both Bradford accidents can be merged however then rainy weather and wet surface will be removed in cluster 1)
                                    - The yellow outliers (ghana) is clearly displayed in this plot showing how useful the outlier detection side of clustering is
                                                        """)
                # Create two columns for side-by-side layout
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**2D PCA for K=5**")
                    st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/UK Dataset_pca_2d.png")
                with col2:
                    st.markdown("**3D PCA for K=5**")
                    st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/UK Dataset_pca_3d.png")
            
            elif clustering_col == "TSNE":
                st.markdown("### T-SNE Clustering Results for K=3")
                with st.expander("ℹ️ Insights"):
                    st.write("""
                    - t-SNE are another way to cluster data based on the local relationships instead of finding maxmium variance the points are placed based on their local neighbourhood relationships 
                    - This could lead to more intricate clusters compared to PCA diagram 
                    - They also help fact check the PCA diagram

                    **Insights**
                    - t-SNE confirms the findings with the PCA plot wit the cluster 5 outliers but the diagram is much more scattered compared to PCA
                    - Here clearer clusters between cluster 3 is found but cluster 1 and 2 seems more interconnected (Bradford accidents)
                """)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**2D TSNE for K=5**")
                    st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/UK Dataset_tsne_2d.png")
                with col2:  # You were missing the `with col2:` statement
                    st.markdown("**3D TSNE for K=5**")
                    st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/UK Dataset_tsne_3d.png")
        
        elif dataset_choice == "US":
            flexible_success("US Silhouette Score = **0.219**  |  Cluster Distribution: 0: 1970, 1: 1674, 2: 388", alignment="center")
            with st.expander("ℹ️ Table Insights"):
                st.write("""
                - There are 3 Clusters are chosen due to elbow method recommendation, here the mode can be found for each data cluster to enable cluster labelling 
                - Clear ADAS/ADS clusters can be found showing high signs of certain characteristics being unlilke to the automation such as Make, Roadway Type, Precrash movement
                - However the ADAS vehicles are broken into cluster itself showing vehicles such as model Y having issues with Wetter roads and Cloudy weather
                - ADS tends to struggle more on roads with smaller speeds and commonly hitting with passengers cars however as precrash movement is stopped and contact area is back showing that 
                they may not be at fault and the crash partner is at fault or issues with breaking 
                """)
            if clustering_col == "View Clustered Data":
                csv_path = "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/US_cluster_summary.csv"
                df = pd.read_csv(csv_path)
                st.dataframe(df.head())

            elif clustering_col == "K-Means":
                st.markdown("**Showing Optimal K for US dataset**")
                with st.expander("ℹ️ Elbow Method Insights"):
                        st.write("""
                        - Two methods to determine the ideal clusters other than randomsearchCV and gridsearchCV is using the elbow method and silohuette score.
                        - The optimal k can be found using the part where inertia decreases steadily in with case around 3-5, however during this time it is important to balance it with the average  silohuette score which is mean distance to the closest cluster with scores close to 1 being well clustered, -1 to the incorrect cluster and 0 being close to the nearest boundary
                        - Here the scores are closer to one suggesting a lower need for clusters suggesting less meaningful differences between the clusters. 
                        - **For the US dataset**, K=3 is indicated by randomsearchCV and higher silohuette score as the optimial choice, within the jupiter notebook the number of k can be experimentally changed. 
                        """)

                st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/US Dataset_plot.png")
                st.markdown("**Decision Trees showing Split for K=3**")
                with st.expander("ℹ️ Insights"):
                        st.write("""
                        - When using K-Means data is split into K number of cluster, we can adjust hyperparamters but we do not know why a split has occured and the percentages behind each feature. This is why a decision tree can be plotted to show the split and how each cluster is defined other than reading a table and worrying about how to label the clusters. 
                        - A decision tree works by splititng the data based on a condition (root node) and the more layers the more splits can be made which can help trace the cluster 
                        - Here each colour is a cluster with (orange:0),(Purple:2),(Green:1) and white nodes if not 100% sure with selection


                        **US Dataset Insight**
                        - As expected the root node is the to do with ADAS and ADS, we view the sample split it is indeed based on 57.7% ADAS and 43.3% ADS which is exactly the same as the percentage split in the EDA giving a great opportunity see the key differences in Automation levels. 
                        - For ADAS Clusters 0 (orange) and 2 (purple) the Speed, Weather, Model, Location are the splits, whilst Tesla Model 3 have greater accidents in dry highways in LA whilst Cluster 2 is also Tesla but Model Y on wetter, cloudy weather.
                        - One the otherside, ADS vehicles like Jaguar I-Pace are not affected as much by weather,  accidents tend to be whilst stopped, on clear dry roads and with passenger car contacted on the back
                        - ADS tends to struggle more on roads with smaller speeds they may not be at fault and the crash partner is at fault or issues with breaking 
                        - While ADAS are travelling straight and hit a fixed object.
                        - The data is pretty skewed towards California too as this is where majority of the data is, additionally within this data there is daylight lighting but accidents occuring at night which shows that cluster labelling based in mode can be skewed
                        - With the severity being high 

                        **Highest Features**
                        ADS Equipped?: 0.7058
                        Roadway Surface: 0.2340
                        Weather: 0.0547
                        City: 0.0026
                        Model: 0.0018
                        Automation System Engaged?: 0.0009
                        Incident Date: 0.0001
                        Posted Speed Limit (MPH): 0.0001

                        **Cluster Labelling**
                        - Cluster 0: ADAS Tesla Model 3 with dry highways, clear weather (Los Angeles)
                        - Cluster 1: ADS Jaguar I-Pace with lower speeds, while parked
                        - Cluster 2: ADAS Tesla Model Y with wet highways, clear weather (San Francisco)

                        """)

                st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/US Dataset_explainability.png")

            elif clustering_col == "PCA":
                st.markdown("### PCA Clustering Results for K=3")
                    # Single insights section for both plots
                with st.expander("ℹ️ Insights"):
                        st.write("""
                            - PCA plots are visual methods to find clusters by reducing the data into principal components (2 for 2D and 3 for 3D) these components try ro find the mosrt variance in the data and help see how tightly packed the clusters can be seen 
                            - While some points my be closer together other clusters can be classfied as outliers or how they could be part of a different cluster (with overlaps)

                            **Insights**
                            - The densely packed ADAS and ADS clusters can been with the green and purple clusters
                            - However whilst the second ADAS (cluster 2)  could be recommended to be merged with cluster 0
                            - The 3D plot helps visualise why the merge should not be merged as the X and Z axis show the road surface and weather is a seperate cluster containing both ADAS and ADS 
                                                """)
                # Create two columns for side-by-side layout
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**2D PCA for K=3**")
                    st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/US Dataset_pca_2d.png")
                with col2:
                    st.markdown("**3D PCA for K=3**")
                    st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/US Dataset_pca_3d.png")
            
            elif clustering_col == "TSNE":
                st.markdown("### T-SNE Clustering Results for K=3")
                with st.expander("ℹ️ Insights"):
                    st.write("""
                        - t-SNE are another way to cluster data based on the local relationships instead of finding maxmium variance the points are placed based on their local neighbourhood relationships 
                        - This could lead to more intricate clusters compared to PCA diagram 
                        - They also help fact check the PCA diagram

                        **Insights**
                        - t-SNE confirms the findings with the PCA plot showing distinct but not overfitted clusters, once again the data is largely split based on ADAS and ADS and Roadway/Weather conditions 
                        - For the Yellow (Cluster 2) more overlap can be shown with the ADAS pink (Cluster:0)
                """)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**2D TSNE for K=3**")
                    st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/US Dataset_tsne_2d.png")
                with col2:  # You were missing the `with col2:` statement
                    st.markdown("**3D TSNE for K=3**")
                    st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/US Dataset_tsne_3d.png")