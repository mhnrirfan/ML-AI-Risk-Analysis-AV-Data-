# Imports
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import numpy as np
import plotly.express as px
from prettytable import PrettyTable
import sys

# Add your cleaning scripts path for import
sys.path.append('/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Python Scripts Cleaning')

# Load raw datasets
collisions = pd.read_csv('/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK/dft-road-casualty-statistics-collision-last-5-years.csv')
vehicles = pd.read_csv('/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK/dft-road-casualty-statistics-vehicle-last-5-years.csv')
adas = pd.read_csv('/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/US/SGO-2021-01_Incident_Reports_ADAS.csv')
ads = pd.read_csv('/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/US/SGO-2021-01_Incident_Reports_ADS.csv')

# Import cleaning functions and clean data
from US_LoadClean import US_Data_Cleaning
US_data = US_Data_Cleaning(adas, ads)

from UK_LoadClean import UK_Data_Cleaning
UK_data = UK_Data_Cleaning(collisions, vehicles)

# Basic Data Inspection
print(f"UK Data Shape: {UK_data.shape}")
print(f"US Data Shape: {US_data.shape}\n")

print("UK Data Head:")
print(UK_data.head(), "\n")

print("US Data Head:")
print(US_data.head(), "\n")

# Columns for visualization
columns_to_plot = [
    'Lighting',
    'Crash With',
    'Highest Injury Severity Alleged',
    'SV Pre-Crash Movement',
    'Weather',
    'Roadway Type',
    'Roadway Surface',
    'SV Contact Area'
]

# Visualization setup
fig, axes = plt.subplots(len(columns_to_plot), 2, figsize=(14, len(columns_to_plot)*4), sharey=False)

for idx, column in enumerate(columns_to_plot):
    # Check column existence to avoid KeyErrors
    if column not in UK_data.columns:
        print(f"Warning: '{column}' not in UK data columns. Skipping.")
        continue
    if column not in US_data.columns:
        print(f"Warning: '{column}' not in US data columns. Skipping.")
        continue
    
    # UK Data Plot
    uk_counts = UK_data[column].value_counts(dropna=False)
    uk_counts.plot(kind='bar', ax=axes[idx, 0], color='lightcoral')
    axes[idx, 0].set_title(f'UK Data: {column}')
    axes[idx, 0].set_xlabel(column)
    axes[idx, 0].set_ylabel('Count')
    axes[idx, 0].tick_params(axis='x', rotation=45)
    axes[idx, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # US Data Plot
    if column == 'SV Contact Area':
        # Special handling for comma-separated multi-values
        area_series = US_data[column].dropna()
        all_areas = []
        for val in area_series:
            parts = [part.strip() for part in val.split(',')]
            all_areas.extend(parts)
        area_counts = Counter(all_areas)
        area_counts_df = pd.Series(area_counts).sort_values(ascending=False)
        area_counts_df.plot(kind='bar', ax=axes[idx, 1], color='lightblue')
    else:
        us_counts = US_data[column].value_counts(dropna=False)
        us_counts.plot(kind='bar', ax=axes[idx, 1], color='lightblue')

    axes[idx, 1].set_title(f'US Data: {column}')
    axes[idx, 1].set_xlabel(column)
    axes[idx, 1].tick_params(axis='x', rotation=45)
    axes[idx, 1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

# Save plot
output_path = '/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Python Scripts Cleaning/visualization_output.png'
plt.savefig(output_path)
plt.show()
