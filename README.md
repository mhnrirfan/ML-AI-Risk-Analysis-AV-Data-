### Introduction 
A Machine Learning Study on Accident Risk Patterns in Autonomous and Conventional Vehicles Across U.S. and U.K. Datasets with Model Explainability 
AVAILABLE@ https://mahnooriqbal.streamlit.app
**Derived from COMP702**
- There are many statistical datasets created by public bodies that could have additional uses.
- For example, there are over 1000 available at https://www.gov.uk/government/statistical-data-sets. These include very diverse topics, such as salmon counts in the River Tees, pollution levels and banana prices. Other large-scale data exists, such as from traffic accidents, environmental readings, biological experiments.
- The datasets are often difficult to use as they may be created for human interpretation, are not easily machine-readable, or have missing data and inconsistencies.
- The aim of this project is to develop tools or analyses that can make a practical use of any sets of public data that you are interested in.

**US Dataset**

Datasets are from:  https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting
Including three different datasets
* ADAS = Advanced Driver Assistance Systems
* ADS = Automated Driving Systems
* OTHER = Misclassed/No Available Information

**UK Dataset**
Datasets are from:  https://www.gov.uk/government/publications/road-accidents-and-safety-statistics-notes-and-definitions/road-casualty-statistics-overview-and-coverage
Including two different datasets
* Vehicle Incident Data
* Collision Incident Data
This included collsion and vehicle data which is split up into 2 dataset hence a merge based on accident reference is needed to ensure the cells in the US dataset can be found too


### Aims
- Data Cleaning and Improvement of Data Quality: finding a real-world accident dataset with varying degrees of autonomy and relevant features. Then, cleaning the data, removing outliers, imputing missing values, removing duplicates, encoding and normalising values.
- Exploratory Data Analysis: creating distribution graphs, performing numerical statistical techniques, creating correlation heatmaps to uncover the basic patterns in the dataset
- Use of Supervised and Unsupervised Models: research and deploying useful models for both learning methods, comparing outputs, results from testing and evaluation.
- Applying Explainability Artificial Intelligence techniques: using SHAP and LIME methods on the supervised learning techniques in the project
- Insightful Dashboard: creating a dashboard of the summary results of the project with key findings, what it means and possible recommendations for reducing


### Main structure
- **Jupiter Notebooks:** contains the data science element and plots
    - *US_Cleaning.ipynb*
        - Load and Merge Datasets based Primary Key
        - Get basic info and missingness
        - Reduce to latest report version
        - Connect and merge interconnected columns
        - Convert datatypes
        - Remap any values

    - *UK_Cleaning.ipynb*
        - Load and Merge Datasets based Primary Key
        - Get basic info and missingness, filter for Cars only 
        - Remap/decode values from numbers to words
        - reduce to latest report version
        - Convert longitude and lattitude to  city, state, country
        - Convert datatypes, split columns and calculate age
        - Rename and Map values to match the US dataset

    - *EDA.ipynb (Exploratory Data analysis)*
        - Loading the datasets
        - Look at the shape of the cleaned dataframes
        - Split Columns based on Datatypes
        - Percentage missingness per column and does missingness correlate with severity
        - Numerical: Boxplot, Outliers, EDA
        - Catergorical: Barplots and KDEs (Top 20 so legible)
        - Datetime: Radial plot, months, year, day
        - UK and US map 
        - Severity stacked bar plot

    - *FE.ipynb (feature engineering)*
        - Loading the cleaned datasets
        - Splitting Columns into Datatypes
        - Displaying Missing Percentage per column 
        - Dropping Values with less than 5%
        - Defining which columns to impute
        - Converting Numerical Speed Limit into bins of 10s
        - Mode Imputation
        - LOCF Imputation
        - Random Forest Imputation *
        - XGBoost Imputation *
        - Jaccard Score and Accuracy *
        - Evaluate all the imputation and choose best 

    - *Clustering.ipynb*
        - Stratfied sampling uk dataset
        - target/ordinal/frequency encoding *
        - K means RandomSearchCv for hyperparameter 
        - Silohuette and Elbow methods *
        - Decision Tree for K-Means Explainabilty
        - PCA and TSNE *
        - Cluster Labelling
        
    - *Supervised.ipynb*
        - Load and sample the dataset 
        - Apply SMOTE *
        - target/ordinal/frequency encoding
        - Hyperparameter dictionaries for randomsearch cv*
        - Preparation and Store function* 
        - Decision Tree
        - Random Forest
        - XGBoost
        - Logistic Regression
        - Plot the accuracy comparison (testing/validation/F1/overfitting)*
        - SHAP
        - LIME

    **Folders to store plots**
    - clustering_plots
    - EDA_FE
    - lime_explainations
    - shap_plots
    - model_evalations

- **Python Scripts:** To clean and plot side by side EDA plots
    - US_Cleaning.py
    - UK_Cleaning.py
    - EDA.py

- **Dashboard:** Streamlit dashboard code
    - Home.py: the main code for the dashboard
    - Functions.py Experimenting functions importing into home

- **.gitignore:** code to not upload datasets onto git (memory and storage cap)

- **Datasets**: Contains raw, encoded, imputed version of dataset for each stage
    - UK: Contains the RAW downloaded public files from STATS19
    - US:Contains the RAW downloaded public files from NHSTA
    - LAD: contains shape files for plotting the UK Local Area District Map
    - UK-cleaned_data.csv:  From UK_Cleaning.ipynb
    - US-cleaned_data.csv: From US_Cleaning.ipynb
    - US_imputed_data.csv From FE.ipynb
    - US_scaled_data.csv From FE.ipynb
    - UK_scaled_data.csv From FE.ipynb
    - UK_cluster_summary.csv From Clustering.ipynb
    - US_cluster_summary.csv From Clustering.ipynb
