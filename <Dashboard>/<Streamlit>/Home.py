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
        supervised_options = ["View Data", "Imputers","Hyperparameters","Results","Explainability"]
        supervised_col = st.selectbox("Select Process", options=supervised_options)
        model_col = ["Hyperparameters", "Results", "Explainability"]
        algo_col = ["Decision Tree", "Random Forest", "XGBoost", "Logistic Regression"]


        # Only show the selectbox if the user has chosen a relevant supervised_col
        if supervised_col in model_col:
            chosen_model = st.selectbox("Choose Model", options=algo_col)
        
        st.header("Clustering")
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
            st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Dashboard>/UK_heatmap.png", use_container_width=True)
        with col2:
            st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Dashboard>/UK_top_corr_bar.png", use_container_width=True)
    
    
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
                st.write(f"This chart shows the top {top_n} categories for **{severity_col_choice}** severity analysis.")
                st.write("You can add explanations, methodology, or links here.")
            st.plotly_chart(plot_bar_side_by_side(df), use_container_width=True)
            st.plotly_chart(plot_line_side_by_side(df), use_container_width=True)
            st.markdown("**Imputer Summary**")
            with st.expander("ℹ️ Insights"):
                st.write(f"This chart shows the top {top_n} categories for **{severity_col_choice}** severity analysis.")
                st.write("You can add explanations, methodology, or links here.")
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
            st.write(f"This chart shows the top {top_n} categories for **{severity_col_choice}** severity analysis.")
            st.write("You can add explanations, methodology, or links here.")
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
                    with st.expander("ℹ️ Insights"):
                        st.write(f"This chart shows the top {top_n} categories for **{severity_col_choice}** severity analysis.")
                        st.write("You can add explanations, methodology, or links here.")            
                    st.dataframe(classification_reports[chosen_model][dataset_choice][split])
                    
                    # Open and display confusion matrix image
                    with st.expander("ℹ️ Insights"):
                        st.write(f"This chart shows the top {top_n} categories for **{severity_col_choice}** severity analysis.")
                        st.write("You can add explanations, methodology, or links here.")
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
        with st.expander("ℹ️ Insights"):
            st.write(f"This chart shows the top {top_n} categories for **{severity_col_choice}** severity analysis.")
            st.write("You can add explanations, methodology, or links here.")

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
            with st.expander("ℹ️ Insights"):
                st.write(f"This chart shows the top {top_n} categories for **{severity_col_choice}** severity analysis.")
                st.write("You can add explanations, methodology, or links here.")
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
import streamlit as st
import pandas as pd

with tabs[4]:
    import streamlit as st
    from st_flexible_callout_elements import flexible_error, flexible_success, flexible_warning, flexible_info


    st.subheader("📈 Clustering")
    # Dataset-specific settings
    if dataset_choice == "UK":
        flexible_success("UK Silhouette Score = **0.114**  |  Cluster Distribution: 0 → 4565, 1 → 4734, 2 → 13589", alignment="center")
        with st.expander("ℹ️ Table Insights"):
                st.write("""There are 3 Clusters are chosen due to elbow method recommendation, here the mode can be found for each data cluster to enable cluster labelling 
                
                - Cluster 2 is a much bigger datset containing BMW crashes as the mode however cluster 1 contains identical catergoristics however is with a ford fiesta instead
                these clusters can potentially be merged.
                
                - Cluster 0 is contains ford fiestas too however with older models, affecting birmingham which is a peak area (As seen in EDA map) affecting we roads and weather,
                
                - For all clusters the contact area is front and movement is going ahead with minor injuries and at low 30mph, there is no main cluster for severe accidents potentially
                other than the location, weather and model most accidents are occuring at the same way hence clustering may not be as beneficial.
                """)
        if clustering_col == "View Clustered Data":
            csv_path = "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK_cluster_summary.csv"
            df = pd.read_csv(csv_path)
            st.dataframe(df.head())

        elif clustering_col == "K-Means":
            st.markdown("**Showing Optimal K for UK dataset**")
            with st.expander("ℹ️ Elbow Method Insights"):
                st.write("""
                - The Elbow method shows a steeper drop from cluster 3 to 4, suggesting that 3–4 clusters are meaningful  
                - Selecting 3 clusters provides meaningful clustering, especially around severity-based grouping  
                - 3 clusters also give the highest silhouette score, and since closer to 1 is better, this supports the choice  
                - Overall, the silhouette score is not high enough to be ideal, but it is still worth experimenting with  
                """)

            st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/UK Dataset_plot.png")
            st.markdown("**Decision Trees showing Split for K=3**")
            with st.expander("ℹ️ Insights"):
                    st.write("""
                        - Using Severity as a Target variable for Kmeans, 4 splits are used to help visual the split and validate if meaningful splits based on domain knowledge
                        - ADS equipment is the strongest split: when ADS <= 0.132, about 1,709 samples are classified, mostly toward one cluster  
                        - Weather conditions drive outcomes when ADS > 0.132: for example, weather <= 0.153 covers around 1,933 cases  
                        - Roadway surface appears multiple times as a deciding factor, affecting more than 2,000 samples across branches  
                        - Posted speed limit (<= 35 mph) plays a role in smaller groups, splitting around 479 samples  
                        - Vehicle make and model repeatedly influence cluster assignment, seen in splits of around 200–500 cases  
                        - Automation system engagement shows up in about 1,380 samples, underlining its importance in accident outcomes  
                        - Leaf nodes reveal how accidents distribute across the three clusters, e.g., some nodes show values like [0.003, 0.001, 0.996], indicating very strong purity for a single cluster  
                        - Overall, the model distinguishes three accident clusters:  
                            1. Cluster linked with **low ADS equipment** and **roadway/speed conditions**  
                            2. Cluster shaped by **environmental factors** such as weather and road surface  
                            3. Cluster influenced by **automation engagement errors** and **city/time-related context**  
                        """)

            st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/UK Dataset_explainability.png")

        elif clustering_col == "PCA":
            st.markdown("### PCA Clustering Results for K=3")
                # Single insights section for both plots
            with st.expander("ℹ️ Insights"):
                    st.write("""
                    - The data groups into three clear clusters of accidents which is well-separated clusters, suggesting that K=3 is a suitable choice for this dataset or possible overfitting
                    - This is as the 3D PCA plot highlights separation even more clearly: clusters 0 and 2 appear dense, while cluster 1 spreads across a wider range  
                    - Given the explainabilty diagram, yellow and purple cluster could be merged based on FORD as make
                    """)
            # Create two columns for side-by-side layout
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**2D PCA for K=3**")
                st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/UK Dataset_pca_2d.png")
            with col2:
                st.markdown("**3D PCA for K=3**")
                st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/UK Dataset_pca_3d.png")
        
        elif clustering_col == "TSNE":
            st.markdown("### T-SNE Clustering Results for K=3")
            with st.expander("ℹ️ Insights"):
                st.write("""
                - 3D maps the 2D t-SNE visualization improves cluster separation, confirming that incident conditions dominate the dataset structure
                - Explainabilty show that there is three natural clusters identified (K=3) based on incident features like vehicle make, roadway conditions, and weather
                - Cluster 0: compact, consistent incidents; often associated with wet roads and rain (e.g., Birmingham); mostly conventional vehicles on single carriageways
                - Cluster 1: shows overlap with Cluster 2; incidents mostly under clear/dry weather (e.g., Bradford); contains common low–mid range vehicles (e.g., Ford Fiesta)
                - Cluster 2: largest and most spread-out cluster; incidents under varied conditions but primarily conventional vehicles without ADS; roadway features (single carriageway, 30 MPH) strongly influence clustering
                - Primary drivers of clustering: weather and road surface conditions (rain/wet vs. clear/dry)
                - Secondary driver: vehicle type has less influence than environmental and roadway factors
               """)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**2D TSNE for K=3**")
                st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/UK Dataset_tsne_2d.png")
            with col2:  # You were missing the `with col2:` statement
                st.markdown("**3D TSNE for K=3**")
                st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/UK Dataset_tsne_3d.png")
    
    elif dataset_choice == "US":
        flexible_success("US Silhouette Score = **0.114**  |  Cluster Distribution: 0 → 1970, 1 → 1674, 2 → 233", alignment="center")
        with st.expander("ℹ️ Table Insights"):
            st.write("""There are 3 Clusters are chosen due to elbow method recommendation, here the mode can be found for each data cluster to enable cluster labelling 
            
            - Clear ADAS/ADS clusters can be found showing high signs of certain characteristics being unlilke to the automation such as Make, Roadway Type, Precrash movement
            
            - However the ADAS vehicles are broken into cluster itself showing vehicles such as model 3 having issues with Wetter roads and Cloudy weather
            
            - ADS tends to struggle more on roads with smaller speeds and commonly hitting with passengers cars however as precrash movement is stopped and contact area is back showing that 
            they may not be at fault and the crash partner is at fault or issues with breaking 
            """)
        if clustering_col == "View Clustered Data":
            csv_path = "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/US_cluster_summary.csv"
            df = pd.read_csv(csv_path)
            st.dataframe(df.head())

        elif clustering_col == "K-Means":
            st.markdown("**Showing Optimal K for UK dataset**")
            with st.expander("ℹ️ Elbow Method Insights"):
                    st.write("""
                    - The Elbow method shows a steeper drop from cluster 3 to 4, suggesting that 3–4 clusters are meaningful  
                    - Inertia decreases steadily as the number of clusters increases  
                    - The steepest drop occurs between clusters 2 to 4, suggesting meaningful groupings around k=3 or k=4  
                    - After k=4, the curve flattens, showing diminishing returns from adding more clusters  
                    - The highest silhouette score is at k=3, indicating the best cluster separation  
                    - Scores remain relatively low overall, suggesting clusters are not strongly separated but still provide useful structure  
                    """)

            st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/US Dataset_plot.png")
            st.markdown("**Decision Trees showing Split for K=3**")
            with st.expander("ℹ️ Insights"):
                    st.write("""
                    - ADS equipment is the strongest split: when ADS <= 0.12, around 1,805 samples are classified, mostly toward one cluster  
                    - Weather conditions drive outcomes when ADS > 0.12: for example, weather <= 0.14 covers about 1,933 cases  
                    - Roadway surface appears multiple times as a deciding factor, affecting more than 2,000 samples across branches  
                    - Speed limit (<= 49 mph) plays a role in smaller groups, splitting around 258 samples  
                    - Maneuver type repeatedly influences cluster assignment, seen in splits of around 225–500 cases  
                    - Automation system error shows up in about 1,380 samples, underlining its importance in accident outcomes  
                    - Leaf nodes reveal how accidents distribute across the three clusters, e.g., some nodes show values like [0.003, 0.001, 0.996], indicating very strong purity for a single cluster  
                    - Overall, the model distinguishes three accident clusters:  
                    1. cluster linked with low ADS equipment and roadway/speed conditions  
                    2. cluster shaped by environmental factors such as weather and road surface  
                    3.  cluster influenced by automation errors and crash presence  
                    """)

            st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/US Dataset_explainability.png")

        elif clustering_col == "PCA":
            st.markdown("### PCA Clustering Results for K=3")
                # Single insights section for both plots
            with st.expander("ℹ️ Insights"):
                    st.write("""
                    - The 2D plot shows that the smaller yellow cluster could possibly be noise and merged with the purple and green cluster
                    creating a simple split of adas and ads
                    - However the 3d plot shows that yellow are clustered in the z axis too showcasing a possible 
                    smaller cluster in this case the model and weather seperating cluster 0 (yellow/adas) from cluster 2 (purple/adas) while cluster 1
                    is independently ads and clearly distinct
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
                - 3D maps the 2D t-SNE visualization improves cluster separation, confirming that incident conditions dominate the dataset structure
                - Purple and Teal clusters are far showcases that majority of adas/ads have less variation making the root split highly important 
                - Yellow is bimodal shape where both ADAS and ADS have similar behaviour as weather and roadtype dominate so high that it can be considered 
                as a seperate cluster
               """)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**2D TSNE for K=3**")
                st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/US Dataset_tsne_2d.png")
            with col2:  # You were missing the `with col2:` statement
                st.markdown("**3D TSNE for K=3**")
                st.image("/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Jupiter Notebooks>/clustering_plots/US Dataset_tsne_3d.png")
    