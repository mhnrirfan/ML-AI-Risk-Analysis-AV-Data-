# Complete Imputation Pipeline for AV Crash Data
# Streamlined version with imputation, comparison, and heatmap

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tabulate import tabulate
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# =============================================================================
# 1. DATA LOADING AND SETUP
# =============================================================================

def load_and_prepare_data():
    """Load and prepare the datasets"""
    print("📂 Loading datasets...")
    
    # Load data
    US_data = pd.read_csv('/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/US-cleaned_data.csv')
    UK_data = pd.read_csv('/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK-cleaned_data.csv')
    UK_data = UK_data.drop(['latitude', 'longitude'], axis=1)
    
    print(f"US Dataset Shape: {US_data.shape}")
    print(f"UK Dataset Shape: {UK_data.shape}")
    
    # Prepare US data
    # Convert Posted Speed Limit to categorical
    US_data['Posted Speed Limit (MPH) Rounded'] = US_data['Posted Speed Limit (MPH)'].apply(
        lambda x: 5 * round(x / 5) if pd.notnull(x) else np.nan
    )
    US_data['Posted Speed Limit (MPH)'] = US_data['Posted Speed Limit (MPH) Rounded'].astype('Int64').astype(str)
    US_data['Posted Speed Limit (MPH)'] = US_data['Posted Speed Limit (MPH)'].replace('nan', 'Unknown')
    US_data['Posted Speed Limit (MPH)'] = US_data['Posted Speed Limit (MPH)'].astype('category')
    US_data.drop(columns=['Posted Speed Limit (MPH) Rounded'], inplace=True)
    
    # Convert Model Year to categorical and Incident Time to binary
    US_data['Model Year'] = US_data['Model Year'].astype('category')
    US_data['Incident Time (24:00)'] = pd.to_datetime(
        US_data['Incident Time (24:00)'], format='%H:%M:%S', errors='coerce'
    ).notna().astype(int)
    
    return US_data, UK_data

# =============================================================================
# 2. MISSING DATA ASSESSMENT
# =============================================================================

def assess_missing_data(dataset, dataset_name):
    """Display missing and unknown values"""
    print(f"\n📊 Assessing missing data for {dataset_name}...")
    
    features_with_issues = [
        col for col in dataset.columns
        if dataset[col].isnull().sum() > 0 or
           dataset[col].astype(str).str.lower().str.contains('unknown').any()
    ]
    
    table = PrettyTable()
    table.field_names = ["Feature", "Missing %", "Unknown %"]
    
    for col in features_with_issues:
        missing_pct = round(dataset[col].isnull().mean() * 100, 1)
        unknown_pct = round(dataset[col].astype(str).str.lower().str.contains('unknown').mean() * 100, 1)
        table.add_row([col[:25], f"{missing_pct}%", f"{unknown_pct}%"])
    
    print(table)
    return features_with_issues

# =============================================================================
# 3. IMPUTATION METHODS
# =============================================================================

def mode_impute(df, columns):
    """Simple mode imputation for categorical columns"""
    df_imputed = df.copy()
    for col in columns:
        if col in df.columns and df[col].isna().sum() > 0:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df_imputed[col] = df[col].fillna(mode_val[0])
    return df_imputed

def locf_impute(df, columns, sort_by='Incident Date'):
    """Last Occurrence Carried Forward imputation"""
    df_imputed = df.copy()
    if sort_by in df.columns:
        df_imputed = df_imputed.sort_values(sort_by)
    
    for col in columns:
        if col in df.columns:
            df_imputed[col] = df_imputed[col].fillna(method='ffill').fillna(method='bfill')
    
    return df_imputed

def rf_impute_simple(df, columns):
    """Simplified Random Forest imputation"""
    df_imputed = df.copy()
    
    for col in columns:
        if col not in df.columns or df[col].isna().sum() == 0:
            continue
            
        print(f"  🌲 RF imputing {col}...")
        
        # Get complete cases
        complete_cases = df.dropna(subset=[col])
        if len(complete_cases) < 50:
            print(f"    ⚠️ Too few samples, using mode instead")
            df_imputed = mode_impute(df_imputed, [col])
            continue
        
        # Select features (exclude target and ID columns)
        feature_cols = [c for c in df.columns 
                       if c != col 
                       and c not in ['Report ID', 'Report Version', 'Incident Date']
                       and df[c].notna().sum() > len(df) * 0.7]  # At least 70% non-missing
        
        if len(feature_cols) == 0:
            print(f"    ⚠️ No good features, using mode instead")
            df_imputed = mode_impute(df_imputed, [col])
            continue
        
        # Prepare data
        X = pd.get_dummies(complete_cases[feature_cols], drop_first=True)
        is_numerical = pd.api.types.is_numeric_dtype(df[col])
        
        if is_numerical:
            y = complete_cases[col].values
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        else:
            y = complete_cases[col].astype(str)
            le = LabelEncoder()
            y = le.fit_transform(y)
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        
        try:
            model.fit(X, y)
            
            # Predict missing values
            missing_mask = df[col].isna()
            X_missing = pd.get_dummies(df[missing_mask][feature_cols], drop_first=True)
            X_missing = X_missing.reindex(columns=X.columns, fill_value=0)
            
            if not X_missing.empty:
                preds = model.predict(X_missing)
                
                if is_numerical:
                    df_imputed.loc[missing_mask, col] = preds
                else:
                    preds = le.inverse_transform(preds)
                    df_imputed.loc[missing_mask, col] = preds
                    
        except Exception as e:
            print(f"    ⚠️ RF failed, using mode: {str(e)[:50]}...")
            df_imputed = mode_impute(df_imputed, [col])
    
    return df_imputed

# =============================================================================
# 4. IMPUTATION COMPARISON
# =============================================================================

def compare_imputation_methods(df, columns, test_fraction=0.2):
    """Compare all imputation methods"""
    print(f"\n🔬 Testing imputation methods on {len(columns)} columns...")
    
    results = []
    
    for col in columns:
        if col not in df.columns:
            continue
            
        # Skip if too few valid values
        valid_data = df[col].dropna()
        if len(valid_data) < 50:
            print(f"  ⚠️ Skipping {col} - only {len(valid_data)} valid values")
            continue
        
        print(f"  🧪 Testing {col}...")
        
        # Create test mask (sample valid values to mask)
        np.random.seed(42)
        n_test = int(len(valid_data) * test_fraction)
        test_indices = np.random.choice(valid_data.index, size=n_test, replace=False)
        
        # Store original values
        original_values = df.loc[test_indices, col].copy()
        
        # Create masked dataset
        test_df = df.copy()
        test_df.loc[test_indices, col] = np.nan
        
        # Test each method
        try:
            # Mode imputation
            mode_result = mode_impute(test_df, [col])
            mode_pred = mode_result.loc[test_indices, col]
            mode_accuracy = (mode_pred == original_values).mean()
            
            # LOCF imputation
            locf_result = locf_impute(test_df, [col])
            locf_pred = locf_result.loc[test_indices, col]
            locf_accuracy = (locf_pred == original_values).mean()
            
            # RF imputation
            rf_result = rf_impute_simple(test_df, [col])
            rf_pred = rf_result.loc[test_indices, col]
            rf_accuracy = (rf_pred == original_values).mean()
            
            results.append({
                'Column': col,
                'Mode_Accuracy': mode_accuracy,
                'LOCF_Accuracy': locf_accuracy,
                'RF_Accuracy': rf_accuracy,
                'Sample_Size': n_test,
                'Best_Method': max([('Mode', mode_accuracy), ('LOCF', locf_accuracy), ('RF', rf_accuracy)], 
                                 key=lambda x: x[1])[0]
            })
            
        except Exception as e:
            print(f"    ❌ Error testing {col}: {str(e)[:50]}...")
            
    return pd.DataFrame(results)

def display_comparison_results(results_df):
    """Display comparison results in a nice table"""
    if len(results_df) == 0:
        print("No results to display")
        return
    
    # Format for display
    display_df = results_df.copy()
    for col in ['Mode_Accuracy', 'LOCF_Accuracy', 'RF_Accuracy']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")
    
    print("\n📈 IMPUTATION METHOD COMPARISON RESULTS:")
    print("=" * 80)
    print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
    
    # Summary statistics
    print(f"\n📊 SUMMARY:")
    method_wins = results_df['Best_Method'].value_counts()
    for method, wins in method_wins.items():
        avg_acc = results_df[f'{method}_Accuracy'].mean()
        print(f"  {method}: {wins} wins, {avg_acc:.1%} avg accuracy")

# =============================================================================
# 5. FINAL IMPUTATION AND ENCODING
# =============================================================================

def apply_best_imputation(df, results_df, columns):
    """Apply the best method for each column based on test results"""
    print("\n🎯 Applying best imputation method for each column...")
    
    final_df = df.copy()
    
    for _, row in results_df.iterrows():
        col = row['Column']
        best_method = row['Best_Method']
        
        print(f"  Using {best_method} for {col}")
        
        if best_method == 'Mode':
            final_df = mode_impute(final_df, [col])
        elif best_method == 'LOCF':
            final_df = locf_impute(final_df, [col])
        elif best_method == 'RF':
            final_df = rf_impute_simple(final_df, [col])
    
    # Handle any columns not in results (use mode as fallback)
    remaining_cols = [col for col in columns if col not in results_df['Column'].values]
    if remaining_cols:
        print(f"  Using Mode for remaining columns: {remaining_cols}")
        final_df = mode_impute(final_df, remaining_cols)
    
    return final_df

def encode_for_heatmap(df):
    """Simple encoding for heatmap generation"""
    print("\n🔢 Encoding data for heatmap...")
    
    df_encoded = df.copy()
    
    # Encode categorical variables
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if col not in ['Report ID', 'Report Version', 'Incident Date']:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    # Convert datetime if present
    if 'Incident Date' in df_encoded.columns:
        df_encoded['Incident Date'] = pd.to_datetime(df_encoded['Incident Date'], errors='coerce')
        df_encoded['Incident_Year'] = df_encoded['Incident Date'].dt.year
        df_encoded = df_encoded.drop('Incident Date', axis=1)
    
    return df_encoded

# =============================================================================
# 6. HEATMAP GENERATION
# =============================================================================

def create_heatmap(df_encoded, dataset_name, figsize=(14, 10)):
    """Create correlation heatmap"""
    print(f"\n🔥 Creating heatmap for {dataset_name}...")
    
    # Select only numeric columns, exclude ID columns
    numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns
    plot_cols = [col for col in numeric_cols 
                if col not in ['Report ID', 'Report Version']]
    
    if len(plot_cols) == 0:
        print("  ⚠️ No numeric columns for correlation")
        return
    
    # Calculate correlation matrix
    corr_matrix = df_encoded[plot_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Show only lower triangle
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', linewidths=0.5, cbar_kws={'shrink': 0.8})
    
    plt.title(f'{dataset_name} - Feature Correlation Heatmap', pad=20, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================

def run_imputation_pipeline():
    """Run the complete imputation pipeline"""
    print("🚀 Starting Complete Imputation Pipeline")
    print("=" * 60)
    
    # 1. Load data
    US_data, UK_data = load_and_prepare_data()
    
    # 2. Define columns to impute
    cols_to_impute = [
        'Make', 'Model', 'Model Year', 'Automation System Engaged?', 
        'Incident Time (24:00)', 'City', 'Roadway Type', 'Roadway Surface', 
        'Posted Speed Limit (MPH)', 'Lighting', 'Crash With', 
        'Highest Injury Severity Alleged', 'SV Pre-Crash Movement', 'Weather', 
        'SV Contact Area'
    ]
    
    # 3. Assess missing data
    assess_missing_data(US_data, "US Dataset")
    
    # 4. Compare imputation methods
    results_df = compare_imputation_methods(US_data, cols_to_impute)
    display_comparison_results(results_df)
    
    # 5. Apply best imputation
    US_final = apply_best_imputation(US_data, results_df, cols_to_impute)
    
    # 6. Encode and create heatmap
    US_encoded = encode_for_heatmap(US_final)
    create_heatmap(US_encoded, "US Dataset (After Imputation)")
    
    print(f"\n✅ Pipeline Complete!")
    print(f"Final dataset shape: {US_final.shape}")
    print(f"Missing values remaining: {US_final[cols_to_impute].isnull().sum().sum()}")
    
    return US_final, US_encoded, results_df

# =============================================================================
# 8. RUN THE PIPELINE
# =============================================================================

if __name__ == "__main__":
    # Run the complete pipeline
    final_data, encoded_data, comparison_results = run_imputation_pipeline()
    
    # Optional: Save results
    # final_data.to_csv('US_data_imputed.csv', index=False)
    # comparison_results.to_csv('imputation_comparison_results.csv', index=False)