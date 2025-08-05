# Filename: Enhanced_EDA_Visualization.py
"""
Description: Enhanced EDA script for UK vs US vehicle incident data comparison
Inputs: Cleaned UK and US datasets via import
Outputs: Professional visualizations and data analysis
"""

# === IMPORTS ===
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prettytable import PrettyTable
import sys
import warnings
warnings.filterwarnings('ignore')

# Set professional styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14
})

# === DATA LOADING ===
print("🚀 Loading datasets and cleaning functions...")

# Add cleaning scripts path
sys.path.append('/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Python Scripts Cleaning')

# Load raw datasets
print("📁 Loading raw data files...")
try:
    collisions = pd.read_csv('/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK/dft-road-casualty-statistics-collision-last-5-years.csv')
    vehicles = pd.read_csv('/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK/dft-road-casualty-statistics-vehicle-last-5-years.csv')
    adas = pd.read_csv('/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/US/SGO-2021-01_Incident_Reports_ADAS.csv')
    ads = pd.read_csv('/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/US/SGO-2021-01_Incident_Reports_ADS.csv')
    print("✅ Raw data loaded successfully")
except FileNotFoundError as e:
    print(f"❌ Error loading files: {e}")
    sys.exit(1)

# Import and apply cleaning functions
print("🧹 Applying data cleaning...")
try:
    from US_Cleaning import main as US_Data_Cleaning
    from UK_Cleaning import main as UK_Data_Cleaning
    
    US_data = US_Data_Cleaning()
    UK_data = UK_Data_Cleaning()
    print("✅ Data cleaning completed successfully")
except ImportError as e:
    print(f"❌ Error importing cleaning functions: {e}")
    sys.exit(1)

# === BASIC DATA INSPECTION ===
print("\n" + "="*60)
print("📊 BASIC DATA INSPECTION")
print("="*60)

print(f"🇬🇧 UK Data Shape: {UK_data.shape}")
print(f"🇺🇸 US Data Shape: {US_data.shape}")

# Column comparison
uk_cols = set(UK_data.columns)
us_cols = set(US_data.columns)
common_cols = uk_cols & us_cols
uk_only = uk_cols - us_cols
us_only = us_cols - uk_cols

print(f"\n📋 Column Analysis:")
print(f"   Common columns: {len(common_cols)}")
print(f"   UK-only columns: {len(uk_only)}")
print(f"   US-only columns: {len(us_only)}")

# Display sample data
print(f"\n🔍 Sample Data Preview:")
print("UK Data Head:")
print(UK_data.head(3).to_string())
print(f"\nUS Data Head:")
print(US_data.head(3).to_string())

# === ENHANCED VISUALIZATION FUNCTIONS ===
def safe_plot_column(data, column, ax, color, title, special_handling=None):
    """
    Safely plot a column with error handling and special cases
    """
    try:
        if column not in data.columns:
            ax.text(0.5, 0.5, f"Column '{column}'\nnot found", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{title}: {column} (Not Available)", color='red')
            return False
        
        # Handle special cases
        if special_handling == 'contact_area':
            # Handle comma-separated values
            area_series = data[column].dropna()
            if len(area_series) == 0:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{title}: {column} (No Data)")
                return False
            
            all_areas = []
            for val in area_series:
                if pd.notna(val) and str(val) != 'nan':
                    parts = [part.strip() for part in str(val).split(',')]
                    all_areas.extend(parts)
            
            if not all_areas:
                ax.text(0.5, 0.5, 'No valid data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{title}: {column} (No Valid Data)")
                return False
            
            area_counts = Counter(all_areas)
            counts_df = pd.Series(area_counts).sort_values(ascending=False).head(10)
        else:
            # Regular column handling
            counts_df = data[column].value_counts(dropna=False).head(10)
        
        if len(counts_df) == 0:
            ax.text(0.5, 0.5, 'No data to plot', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{title}: {column} (Empty)")
            return False
        
        # Create the plot
        bars = counts_df.plot(kind='bar', ax=ax, color=color, alpha=0.8, width=0.7)
        
        # Formatting
        ax.set_title(f"{title}: {column}", fontweight='bold', pad=10)
        ax.set_xlabel(column, fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add value labels on bars
        for bar in ax.patches:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        return True
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error plotting:\n{str(e)[:50]}...', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(f"{title}: {column} (Error)", color='red')
        return False

def create_comparison_visualization():
    """
    Create comprehensive comparison visualizations
    """
    print("\n" + "="*60)
    print("📈 CREATING COMPARISON VISUALIZATIONS")
    print("="*60)
    
    # Columns to analyze
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
    
    # Filter for existing columns
    valid_columns = []
    for col in columns_to_plot:
        if col in UK_data.columns or col in US_data.columns:
            valid_columns.append(col)
            print(f"✅ {col}: Available")
        else:
            print(f"❌ {col}: Not available in either dataset")
    
    if not valid_columns:
        print("⚠️  No valid columns found for visualization")
        return
    
    # Create figure
    n_cols = len(valid_columns)
    fig, axes = plt.subplots(n_cols, 2, figsize=(16, n_cols * 4))
    
    # Handle single row case
    if n_cols == 1:
        axes = axes.reshape(1, -1)
    
    # Color scheme
    uk_color = '#FF6B6B'  # Coral red
    us_color = '#4ECDC4'  # Teal
    
    # Plot each column
    for idx, column in enumerate(valid_columns):
        print(f"📊 Processing: {column}")
        
        # Determine special handling
        special = 'contact_area' if column == 'SV Contact Area' else None
        
        safe_plot_column(UK_data, column, axes[idx, 0], uk_color, "🇬🇧 UK", special)
        safe_plot_column(US_data, column, axes[idx, 1], us_color, "🇺🇸 US", special)
    
    # Overall formatting
    plt.suptitle('🚗 UK vs US Vehicle Incident Data Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save plot
    output_path = '/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/<Python Scripts>/enhanced_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Visualization saved to: {output_path}")
    
    plt.show()
    
    return valid_columns

def generate_summary_statistics():
    """
    Generate summary statistics for both datasets
    """
    print("\n" + "="*60)
    print("📈 SUMMARY STATISTICS")
    print("="*60)
    
    # Basic statistics
    print(f"📊 Dataset Overview:")
    print(f"   🇬🇧 UK Data: {UK_data.shape[0]:,} records, {UK_data.shape[1]} columns")
    print(f"   🇺🇸 US Data: {US_data.shape[0]:,} records, {US_data.shape[1]} columns")

# === MAIN EXECUTION ===
def main():
    """
    Main execution function
    """
    try:
        # Generate summary statistics
        generate_summary_statistics()
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()