#!/usr/bin/env python3
"""
Exploratory Data Analysis for Autonomous Vehicle Incident Data

Purpose: Conducting EDA analysis on UK and US autonomous vehicle incident data,
including numerical stats, distributions, target analysis and geographical views.

Sections:
1. Loading and Merging Dataset
2. Assessing Missingness
3. EDA for Numerical Values
4. EDA for Categorical Values
5. EDA for DateTime Values
6. Choropleth Maps
7. Severity Stacked Bar Plot
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
from collections import Counter
import plotly.express as px
import geopandas as gpd
from shapely.geometry import Point
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Display all columns of a pandas DataFrame when printed
pd.set_option('display.max_columns', None)

class EDAClass:
    """Class for performing EDA on Autonomous Vehicle incident data"""
    
    def __init__(self, us_data_path, uk_data_path):
        """Initialize with data paths"""
        self.us_data_path = us_data_path
        self.uk_data_path = uk_data_path
        self.US_data = None
        self.UK_data = None
        
        # Define column categories
        self.numerical_columns = ['Posted Speed Limit (MPH)']
        self.categorical_columns = [
            'Make', 'Model', 'Model Year', 'ADS Equipped?',
            'Automation System Engaged?', 'City', 'State', 'Roadway Type', 
            'Roadway Surface', 'Lighting', 'Crash With', 'Highest Injury Severity Alleged',
            'SV Pre-Crash Movement', 'SV Contact Area', 'Weather', 'Country'
        ]
        self.datetime_columns = ['Incident Date', 'Incident Time (24:00)']
        self.indexing_columns = ['Report ID', 'Report Version']
    
    def load_data(self):
        """Load the cleaned datasets"""
        print("Loading datasets...")
        self.US_data = pd.read_csv(self.us_data_path)
        self.UK_data = pd.read_csv(self.uk_data_path)
        
        print("Dataset shapes:")
        print(f"UK data: {self.UK_data.shape}")
        print(f"US data: {self.US_data.shape}")
        
        print("\nDataset columns:")
        print(f"UK columns: {list(self.UK_data.columns)}")
        print(f"US columns: {list(self.US_data.columns)}")
    
    def display_missing_values(self, dataset, dataset_name):
        """Display missing values in any passed dataset per column"""
        features_with_na = [feature for feature in dataset.columns if dataset[feature].isnull().sum() > 0]
        
        table = PrettyTable()
        table.field_names = ["Feature", "Percentage of Missing Values"]
        
        for feature in features_with_na:
            missing_percentage = np.round(dataset[feature].isnull().mean() * 100, 2)
            table.add_row([feature, f"{missing_percentage} %"])
        
        print(f"Missing Values in {dataset_name}:")
        print(table)
    
    def plot_missing_correlation(self, dataset, dataset_name, target_variable, color_map):
        """Plot correlation between missing values and target variable"""
        features_with_na = [feature for feature in dataset.columns if dataset[feature].isnull().sum() > 0]
        num_features = len(features_with_na)

        cols = 2
        rows = (num_features + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 5))
        axes = axes.flatten()

        for i, feature in enumerate(features_with_na):
            data = dataset.copy()
            data[feature + "_missing"] = np.where(data[feature].isnull(), 1, 0)
            grouped = data.groupby([feature + "_missing", target_variable]).size().unstack(fill_value=0)

            grouped.plot(kind='bar', stacked=True, ax=axes[i], colormap=color_map)
            axes[i].set_title(f"{dataset_name}: Missingness in '{feature}'")
            axes[i].set_xlabel("Missing (0 = present, 1 = missing)")
            axes[i].set_ylabel("Count")

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
    
    def preprocess_numeric_column(self, df, column_name):
        """Ensure the column is numeric and drop missing values"""
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        df = df.dropna(subset=[column_name])
        return df

    def detect_outliers_iqr(self, df, column_name):
        """Detect outliers using the IQR method"""
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (df[column_name] < lower_bound) | (df[column_name] > upper_bound)

    def plot_boxplot_and_kde(self, data, column, region, color):
        """Plot both boxplot and KDE for a given region and column"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Boxplot
        sns.boxplot(x=data[column], color=color, flierprops=dict(marker='x', markersize=6),
                    boxprops=dict(alpha=0.6), medianprops=dict(color='black'), ax=ax1)
        ax1.set_title(f"{region} Dataset: {column} Boxplot", fontsize=14)
        ax1.set_xlabel(column)
        ax1.grid(True)
        
        # KDE plot
        sns.kdeplot(data[column], color=color, fill=True, alpha=0.6, ax=ax2)
        ax2.set_title(f"{region} Dataset: {column} KDE Distribution", fontsize=14)
        ax2.set_xlabel(column)
        ax2.set_ylabel("Density")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_combined_kde(self, uk_data, us_data, column):
        """Plot KDE comparison for both regions on the same plot"""
        plt.figure(figsize=(12, 6))
        
        sns.kdeplot(uk_data[column], color='purple', fill=True, alpha=0.6, label='UK')
        sns.kdeplot(us_data[column], color='lightgreen', fill=True, alpha=0.6, label='US')
        plt.title(f"KDE Distribution Comparison: {column} (UK vs US)", fontsize=14)
        plt.xlabel(column)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_combined_boxplot(self, uk_data, us_data, column):
        """Plot boxplot comparison for both regions on the same plot"""
        uk_subset = uk_data[[column]].copy()
        uk_subset['Region'] = 'UK'
        us_subset = us_data[[column]].copy()
        us_subset['Region'] = 'US'
        
        combined_data = pd.concat([uk_subset, us_subset], ignore_index=True)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=combined_data, x='Region', y=column, 
                    palette={'UK': 'purple', 'US': 'lightgreen'},
                    flierprops=dict(marker='x', markersize=6),
                    boxprops=dict(alpha=0.6), 
                    medianprops=dict(color='black'))
        
        plt.title(f"Boxplot Comparison: {column} (UK vs US)", fontsize=14)
        plt.ylabel(column)
        plt.grid(True)
        plt.show()
    
    def analyze_numerical_data(self):
        """Analyze numerical columns"""
        print("Analyzing numerical data...")
        
        # Preprocess data
        self.UK_data = self.preprocess_numeric_column(self.UK_data, "Model Year")
        uk_outliers_year = self.detect_outliers_iqr(self.UK_data, "Model Year")
        print(f"UK model year outliers: {len(self.UK_data[uk_outliers_year])}")

        self.US_data = self.preprocess_numeric_column(self.US_data, "Model Year")
        us_outliers_year = self.detect_outliers_iqr(self.US_data, "Model Year")
        print(f"US model year outliers: {len(self.US_data[us_outliers_year])}")

        # Plot individual boxplots and KDE for each region
        self.plot_boxplot_and_kde(self.UK_data, "Model Year", "UK", "purple")
        self.plot_boxplot_and_kde(self.US_data, "Model Year", "US", "lightgreen")
        self.plot_combined_boxplot(self.UK_data, self.US_data, "Model Year")
        self.plot_combined_kde(self.UK_data, self.US_data, "Model Year")
    
    def analyze_categorical_data(self):
        """Analyze categorical columns"""
        print("Analyzing categorical data...")
        
        fig, axes = plt.subplots(len(self.categorical_columns), 2, 
                               figsize=(14, len(self.categorical_columns)*4), sharey=False)
        if len(self.categorical_columns) == 1:
            axes = axes.reshape(1, 2)

        for idx, column in enumerate(self.categorical_columns):
            if column not in self.UK_data.columns or column not in self.US_data.columns:
                print(f"Warning: '{column}' not in data columns. Skipping.")
                continue
            
            # UK Data bar plot - top 20 values
            uk_counts = self.UK_data[column].value_counts(dropna=False).head(20)
            uk_counts.plot(kind='bar', ax=axes[idx, 0], color='#CBAACB', edgecolor='black')
            axes[idx, 0].set_title(f'UK Data: {column} (Top 20 values)')
            axes[idx, 0].set_xlabel(column)
            axes[idx, 0].set_ylabel('Count')
            axes[idx, 0].tick_params(axis='x', rotation=45)
            axes[idx, 0].grid(axis='y', linestyle='--', alpha=0.7)

            # US Data bar plot - top 10 values
            if column == 'SV Contact Area':
                area_series = self.US_data[column].dropna()
                all_areas = []
                for val in area_series:
                    parts = [part.strip() for part in val.split(',')]
                    all_areas.extend(parts)
                area_counts = Counter(all_areas)
                area_counts_df = pd.Series(area_counts).sort_values(ascending=False).head(10)
                area_counts_df.plot(kind='bar', ax=axes[idx, 1], color='#B5EAD7', edgecolor='black')
            else:
                us_counts = self.US_data[column].value_counts(dropna=False).head(10)
                us_counts.plot(kind='bar', ax=axes[idx, 1], color='#B5EAD7', edgecolor='black')

            axes[idx, 1].set_title(f'US Data: {column} (Top 10 values)')
            axes[idx, 1].set_xlabel(column)
            axes[idx, 1].tick_params(axis='x', rotation=45)
            axes[idx, 1].grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()
    
    def plot_incident_time_radial_side_by_side(self):
        """Plot radial bar charts for incident time distribution side by side"""
        for df in [self.UK_data, self.US_data]:
            if 'Incident Time (24:00)' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['Incident Time (24:00)']):
                    df['Incident Time (24:00)'] = pd.to_datetime(df['Incident Time (24:00)'], errors='coerce').dt.hour
                else:
                    df['Incident Time (24:00)'] = df['Incident Time (24:00)'].dt.hour

        uk_hours = self.UK_data['Incident Time (24:00)'].dropna().astype(int)
        uk_counts = uk_hours.value_counts().reindex(range(24), fill_value=0).sort_index()
        us_hours = self.US_data['Incident Time (24:00)'].dropna().astype(int)
        us_counts = us_hours.value_counts().reindex(range(24), fill_value=0).sort_index()
        hours = range(24)
        theta = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
        width = 2 * np.pi / 24

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': 'polar'})
        colors = sns.color_palette('pastel', 24)

        # UK plot
        axes[0].bar(theta, uk_counts, width=width, bottom=0.0, color=colors, edgecolor='black')
        axes[0].set_theta_zero_location("N")
        axes[0].set_theta_direction(-1)
        axes[0].set_xticks(theta)
        axes[0].set_xticklabels([f'{h}:00' for h in hours])
        axes[0].set_title('UK Incident Time Distribution', fontsize=14, pad=20)

        # US plot
        axes[1].bar(theta, us_counts, width=width, bottom=0.0, color=colors, edgecolor='black')
        axes[1].set_theta_zero_location("N")
        axes[1].set_theta_direction(-1)
        axes[1].set_xticks(theta)
        axes[1].set_xticklabels([f'{h}:00' for h in hours])
        axes[1].set_title('US Incident Time Distribution', fontsize=14, pad=20)

        plt.tight_layout()
        plt.show()
    
    def plot_weekday_month_year_side_by_side(self):
        """Plot weekday, month, and year distributions side by side"""
        for df in [self.UK_data, self.US_data]:
            if 'Incident Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Incident Date']):
                df['Incident Date'] = pd.to_datetime(df['Incident Date'], errors='coerce')

            if 'Weekday' not in df.columns:
                df['Weekday'] = df['Incident Date'].dt.day_name()
            if 'Month' not in df.columns:
                df['Month'] = df['Incident Date'].dt.month
            if 'Year' not in df.columns:
                df['Year'] = df['Incident Date'].dt.year
        
        time_columns = ['Weekday', 'Month', 'Year']
        fig, axes = plt.subplots(len(time_columns), 2, figsize=(14, len(time_columns)*4), sharey=False)
        if len(time_columns) == 1:
            axes = axes.reshape(1, 2)

        for idx, column in enumerate(time_columns):
            # UK plot
            uk_counts = self.UK_data[column].value_counts(dropna=False)
            if column == 'Weekday':
                weekdays_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                uk_counts = uk_counts.reindex(weekdays_order).fillna(0)
            else:
                uk_counts = uk_counts.sort_index()

            uk_counts.plot(kind='bar', ax=axes[idx, 0], color='#CBAACB', edgecolor='black')
            axes[idx, 0].set_title(f'UK Data: {column} (Counts)')
            axes[idx, 0].set_xlabel(column)
            axes[idx, 0].set_ylabel('Count')
            axes[idx, 0].tick_params(axis='x', rotation=45)
            axes[idx, 0].grid(axis='y', linestyle='--', alpha=0.7)

            # US plot
            us_counts = self.US_data[column].value_counts(dropna=False)
            if column == 'Weekday':
                us_counts = us_counts.reindex(weekdays_order).fillna(0)
            else:
                us_counts = us_counts.sort_index()

            us_counts.plot(kind='bar', ax=axes[idx, 1], color='#B5EAD7', edgecolor='black')
            axes[idx, 1].set_title(f'US Data: {column} (Counts)')
            axes[idx, 1].set_xlabel(column)
            axes[idx, 1].tick_params(axis='x', rotation=45)
            axes[idx, 1].grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()
    
    def analyze_datetime_data(self):
        """Analyze datetime columns"""
        print("Analyzing datetime data...")
        self.plot_incident_time_radial_side_by_side()
        self.plot_weekday_month_year_side_by_side()
    
    def plot_uk_choropleth(self, uk_shapefile_path):
        """Create a choropleth map of UK incidents by Local Authority District"""
        print("Creating UK choropleth map...")
        
        # Convert UK_data to GeoDataFrame
        geometry = [Point(xy) for xy in zip(self.UK_data['longitude'], self.UK_data['latitude'])]
        gdf_points = gpd.GeoDataFrame(self.UK_data, geometry=geometry, crs="EPSG:4326")

        # Load UK LAD shapefile
        uk_lads = gpd.read_file(uk_shapefile_path)

        # Reproject and spatial join
        gdf_points = gdf_points.to_crs(uk_lads.crs)
        gdf_joined = gpd.sjoin(gdf_points, uk_lads, how="left", predicate="within")

        # Count incidents per LAD
        lad_counts = gdf_joined.groupby('LAD25NM').size().reset_index(name='incident_count')

        # Merge counts with LAD GeoDataFrame
        choropleth_gdf = uk_lads.merge(lad_counts, how='left', on='LAD25NM')
        choropleth_gdf['incident_count'] = choropleth_gdf['incident_count'].fillna(0)

        # Enhanced plot
        fig, ax = plt.subplots(figsize=(12, 14))
        choropleth_gdf.plot(column='incident_count',
                            cmap='Blues',
                            linewidth=0.6,
                            ax=ax,
                            edgecolor='lightgrey',
                            legend=True)
        
        total = int(choropleth_gdf['incident_count'].sum())
        max_count = int(choropleth_gdf['incident_count'].max())
        plt.title(f'UK Incidents by Local Authority\nTotal: {total:,} | Peak: {max_count:,}', 
                  fontsize=16, fontweight='bold', color='darkblue')
        
        stats = f"Areas with incidents: {len(choropleth_gdf[choropleth_gdf['incident_count'] > 0])}/{len(choropleth_gdf)}"
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='black'))
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Print statistics
        matched = gdf_joined['LAD25NM'].notna().sum()
        unmatched = gdf_joined['LAD25NM'].isna().sum()
        print(f"Matched: {matched:,} | Unmatched: {unmatched:,}")

        top5 = choropleth_gdf.nlargest(5, 'incident_count')[['LAD25NM', 'incident_count']]
        print("\nTop 5 Areas:")
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            print(f"{i}. {row['LAD25NM']}: {int(row['incident_count'])} incidents")
    
    def plot_us_state_choropleth(self):
        """Create a choropleth map of US incidents by state"""
        print("Creating US state choropleth map...")
        
        self.US_data['State'] = self.US_data['State'].astype(str).str.strip().str.upper()
        state_counts = self.US_data['State'].value_counts().reset_index()
        state_counts.columns = ['State', 'Count']
        
        total_incidents = state_counts['Count'].sum()
        max_count = state_counts['Count'].max()
        
        fig = px.choropleth(
            state_counts,
            locations='State',
            locationmode='USA-states',
            color='Count',
            color_continuous_scale='Blues',
            scope='usa',
            title=f'US Incidents by State<br>Total: {total_incidents:,} | Peak: {max_count:,}'
        )

        fig.show()
        
        print("Top 5 States:")
        top_5 = state_counts.head(5)
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"{i}. {row['State']}: {row['Count']} incidents")
    
    def plot_california_cities(self):
        """Create a bar chart of the top 10 cities in California by incident count"""
        print("Creating California cities plot...")
        
        ca_data = self.US_data[self.US_data['State'].str.strip().str.upper() == 'CA']
        city_counts = ca_data['City'].value_counts().head(15).reset_index()
        city_counts.columns = ['City', 'Count']
        total_cities = len(ca_data['City'].value_counts())
        peak_count = city_counts['Count'].max()
        
        fig = px.bar(city_counts, x='City', y='Count', 
                     color='Count', color_continuous_scale='Blues',
                     title=f'Top 15 California Cities | Cities: {total_cities} | Peak: {peak_count}')
        
        fig.show()
    
    def analyze_geospatial_data(self, uk_shapefile_path=None):
        """Analyze geospatial data"""
        print("Analyzing geospatial data...")
        
        if uk_shapefile_path:
            self.plot_uk_choropleth(uk_shapefile_path)
        
        self.plot_us_state_choropleth()
        self.plot_california_cities()
    
    def analyze_severity_data(self):
        """Create severity analysis with stacked bar plots"""
        print("Analyzing severity data...")
        
        severity_col = 'Highest Injury Severity Alleged'
        fig, axes = plt.subplots(len(self.categorical_columns), 2, 
                               figsize=(14, len(self.categorical_columns)*5), sharey=False)
        
        if len(self.categorical_columns) == 1:
            axes = axes.reshape(1, 2)

        for idx, column in enumerate(self.categorical_columns):
            if column not in self.UK_data.columns or column not in self.US_data.columns:
                print(f"Warning: '{column}' not in data columns. Skipping.")
                continue
            if severity_col not in self.UK_data.columns or severity_col not in self.US_data.columns:
                print(f"Warning: Severity column '{severity_col}' missing in data. Skipping.")
                continue

            # UK Data
            uk_grouped = self.UK_data.groupby([column, severity_col]).size().unstack(fill_value=0)
            uk_totals = uk_grouped.sum(axis=1)
            uk_top_categories = uk_totals.sort_values(ascending=False).head(20).index
            uk_grouped_top = uk_grouped.loc[uk_top_categories]
            uk_grouped_top.plot(kind='bar', stacked=True, ax=axes[idx, 0], 
                               colormap='Pastel1', edgecolor='black')
            axes[idx, 0].set_title(f'UK Data: {column} (Top 20 by count, stacked by severity)')
            axes[idx, 0].set_xlabel(column)
            axes[idx, 0].set_ylabel('Count')
            axes[idx, 0].tick_params(axis='x', rotation=45)
            axes[idx, 0].grid(axis='y', linestyle='--', alpha=0.7)

            # US Data - Special case for 'SV Contact Area'
            if column == 'SV Contact Area':
                all_rows = []
                for _, row in self.US_data[[column, severity_col]].dropna().iterrows():
                    areas = [area.strip() for area in row[column].split(',')]
                    for area in areas:
                        all_rows.append({'Area': area, severity_col: row[severity_col]})
                us_expanded = pd.DataFrame(all_rows)
                us_grouped = us_expanded.groupby(['Area', severity_col]).size().unstack(fill_value=0)
                us_totals = us_grouped.sum(axis=1)
                us_top_areas = us_totals.sort_values(ascending=False).head(10).index
                us_grouped_top = us_grouped.loc[us_top_areas]
                us_grouped_top.plot(kind='bar', stacked=True, ax=axes[idx, 1], 
                                   colormap='Pastel2', edgecolor='black')
            else:
                us_grouped = self.US_data.groupby([column, severity_col]).size().unstack(fill_value=0)
                us_totals = us_grouped.sum(axis=1)
                us_top_categories = us_totals.sort_values(ascending=False).head(10).index
                us_grouped_top = us_grouped.loc[us_top_categories]
                us_grouped_top.plot(kind='bar', stacked=True, ax=axes[idx, 1], 
                                   colormap='Pastel2', edgecolor='black')
            
            axes[idx, 1].set_title(f'US Data: {column} (Top 10 by count, stacked by severity)')
            axes[idx, 1].set_xlabel(column)
            axes[idx, 1].tick_params(axis='x', rotation=45)
            axes[idx, 1].grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()
    
    def run_full_analysis(self, uk_shapefile_path=None):
        """Run the complete EDA analysis"""
        print("Starting Exploratory Data Analysis for Autonomous Vehicle Incidents")
        print("=" * 70)
        
        # Section 1: Load data
        self.load_data()
        
        # Section 2: Assess missingness
        print("\n" + "="*50)
        print("SECTION 2: ASSESSING MISSINGNESS")
        print("="*50)
        self.display_missing_values(self.UK_data, "UK Dataset")
        self.display_missing_values(self.US_data, "US Dataset")
        self.plot_missing_correlation(self.US_data, "US Dataset", 
                                    "Highest Injury Severity Alleged", color_map="YlGn")
        
        # Section 3: Numerical analysis
        print("\n" + "="*50)
        print("SECTION 3: EDA FOR NUMERICAL VALUES")
        print("="*50)
        self.analyze_numerical_data()
        
        # Section 4: Categorical analysis
        print("\n" + "="*50)
        print("SECTION 4: EDA FOR CATEGORICAL VALUES")
        print("="*50)
        self.analyze_categorical_data()
        
        # Section 5: DateTime analysis
        print("\n" + "="*50)
        print("SECTION 5: EDA FOR DATETIME VALUES")
        print("="*50)
        self.analyze_datetime_data()
        
        # Section 6: Geospatial analysis
        print("\n" + "="*50)
        print("SECTION 6: GEOSPATIAL ANALYSIS")
        print("="*50)
        self.analyze_geospatial_data(uk_shapefile_path)
        
        # Section 7: Severity analysis
        print("\n" + "="*50)
        print("SECTION 7: SEVERITY ANALYSIS")
        print("="*50)
        self.analyze_severity_data()
        
        print("\n" + "="*70)
        print("EDA ANALYSIS COMPLETE")
        print("="*70)


def main():
    """Main function to run the EDA"""
    # Update these paths to match your data locations
    US_cleaned_path = '/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/US/US-cleaned_data.csv'
    UK_cleaned_path = '/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK/UK-cleaned_data.csv'
    LAD_shapefile = '/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/LAD_MAY_2025_UK_BFC_2360005762104150824'
    
    # Initialize and run analysis
    eda = EDAClass(US_cleaned_path, UK_cleaned_path)
    eda.run_full_analysis(LAD_shapefile)


if __name__ == "__main__":
    main()