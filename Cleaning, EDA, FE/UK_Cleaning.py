# Filename: UK_Cleaning_Simplified.py
"""
Description: Simplified Python Cleaning Script for UK STATS Data
Inputs: Stats Collision and Vehicle Dataset from https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-accidents-safety-data
Outputs: Cleaned and standardized dataset for EDA 
"""

import pandas as pd
import numpy as np
import itertools
from tabulate import tabulate
import time
from tqdm import tqdm
import reverse_geocode
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def clean_uk_data(base_path='/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK/'):
    """
    Main function to clean UK road accident data
    Returns cleaned DataFrame ready for analysis
    """
    
    print("Starting UK data cleaning...")
    start_time = time.time()
    
    # === STEP 1: LOAD DATA ===
    print("Loading datasets...")
    base_path = Path(base_path)
    
    try:
        collisions = pd.read_csv(base_path / 'dft-road-casualty-statistics-collision-last-5-years.csv')
        vehicles = pd.read_csv(base_path / 'dft-road-casualty-statistics-vehicle-last-5-years.csv')
        print(f"Loaded {len(collisions):,} collision records and {len(vehicles):,} vehicle records")
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. Check the path: {base_path}")
        raise
    
    # === STEP 2: FIND PRIMARY KEY AND MERGE ===
    print("Finding primary key and merging datasets...")
    
    # Find common columns between datasets
    common_cols = set(collisions.columns) & set(vehicles.columns)
    
    # Check for single column primary key
    primary_key = None
    for col in common_cols:
        if collisions[col].nunique() == len(collisions) or vehicles[col].nunique() == len(vehicles):
            primary_key = [col]
            break
    
    # If no single key found, try combinations of 2-3 columns
    if primary_key is None:
        for combo_size in [2, 3]:
            for combo in itertools.combinations(common_cols, combo_size):
                combined = pd.concat([collisions[list(combo)], vehicles[list(combo)]], ignore_index=True)
                if not combined.duplicated(subset=list(combo)).any():
                    primary_key = list(combo)
                    break
            if primary_key:
                break
    
    if primary_key is None:
        raise ValueError("No suitable primary key found for merging")
    
    print(f"Using primary key: {primary_key}")
    
    # Merge datasets
    df = pd.merge(collisions, vehicles, on=primary_key, how='inner', suffixes=('', '_drop'))
    
    # Drop duplicate columns from merge
    drop_cols = [col for col in df.columns if col.endswith('_drop')]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    
    df = df.drop_duplicates(subset=primary_key)
    print(f"Merged dataset: {len(df):,} records")
    
    # === STEP 3: SELECT RELEVANT COLUMNS ===
    print("Selecting relevant columns...")
    
    columns_to_keep = [
        'accident_index', 'accident_reference', 'latitude', 'longitude',
        'location_easting_osgr', 'location_northing_osgr', 'accident_severity',
        'date', 'time', 'road_type', 'speed_limit', 'light_conditions',
        'vehicle_type', 'weather_conditions', 'road_surface_conditions',
        'vehicle_reference', 'vehicle_manoeuvre', 'hit_object_in_carriageway',
        'hit_object_off_carriageway', 'first_point_of_impact',
        'age_of_vehicle', 'generic_make_model'
    ]
    
    # Keep only columns that exist in the dataset
    available_cols = [col for col in columns_to_keep if col in df.columns]
    df = df[available_cols]
    print(f"Selected {len(available_cols)} relevant columns")
    
    # === STEP 4: APPLY VALUE MAPPINGS ===
    print("Applying value mappings...")
    
    # Fill NaN values with -1 for mapping
    df = df.fillna(-1)
    
    # Define mappings for categorical variables
    mappings = {
        'accident_severity': {1: "Fatal", 2: "Serious", 3: "Slight"},
        'road_type': {
            1: "Roundabout", 2: "One way street", 3: "Dual carriageway", 
            6: "Single carriageway", 7: "Slip road", 9: "Unknown", 
            12: "One way street/Slip road", -1: "Missing"
        },
        'light_conditions': {
            1: "Daylight", 4: "Darkness - lights lit", 5: "Darkness - lights unlit", 
            6: "Darkness - no lighting", 7: "Darkness - lighting unknown", -1: "Missing"
        },
        'weather_conditions': {
            1: "Fine no high winds", 2: "Raining no high winds", 3: "Snowing no high winds",
            4: "Fine + high winds", 5: "Raining + high winds", 6: "Snowing + high winds",
            7: "Fog or mist", 8: "Other", 9: "Unknown", -1: "Missing"
        },
        'road_surface_conditions': {
            1: "Dry", 2: "Wet or damp", 3: "Snow", 4: "Frost or ice",
            5: "Flood over 3cm. deep", 6: "Oil or diesel", 7: "Mud", 
            9: "Unknown", -1: "Missing"
        },
        'vehicle_type': {
            1: "Pedal cycle", 2: "Motorcycle 50cc and under", 3: "Motorcycle 125cc and under",
            4: "Motorcycle over 125cc and up to 500cc", 5: "Motorcycle over 500cc",
            8: "Taxi/Private hire car", 9: "Car", 10: "Minibus (8 - 16 passenger seats)",
            11: "Bus or coach (17 or more pass seats)", 16: "Ridden horse",
            17: "Agricultural vehicle", 18: "Tram", 19: "Van / Goods 3.5 tonnes mgw or under",
            20: "Goods over 3.5t. and under 7.5t", 21: "Goods 7.5 tonnes mgw and over",
            22: "Mobility scooter", 23: "Electric motorcycle", 90: "Other vehicle",
            97: "Motorcycle - unknown cc", 98: "Goods vehicle - unknown weight",
            99: "Unknown", -1: "Missing"
        },
        'vehicle_manoeuvre': {
            1: "Reversing", 2: "Parked", 3: "Waiting to go - held up",
            4: "Slowing or stopping", 5: "Moving off", 6: "U-turn",
            7: "Turning left", 8: "Waiting to turn left", 9: "Turning right",
            10: "Waiting to turn right", 11: "Changing lane to left", 12: "Changing lane to right",
            13: "Overtaking moving vehicle - offside", 14: "Overtaking static vehicle - offside",
            15: "Overtaking - nearside", 16: "Going ahead left-hand bend",
            17: "Going ahead right-hand bend", 18: "Going ahead other",
            99: "Unknown", -1: "Missing"
        },
        'hit_object_in_carriageway': {
            0: "None", 1: "Previous accident", 2: "Road works", 4: "Parked vehicle",
            5: "Bridge (roof)", 6: "Bridge (side)", 7: "Bollard or refuge",
            8: "Open door of vehicle", 9: "Central island of roundabout", 10: "Kerb",
            11: "Other object", 12: "Any animal (except ridden horse)", 99: "Unknown", -1: "Missing"
        },
        'hit_object_off_carriageway': {
            0: "None", 1: "Road sign or traffic signal", 2: "Lamp post",
            3: "Telegraph or electricity pole", 4: "Tree", 5: "Bus stop or bus shelter",
            6: "Central crash barrier", 7: "Near/Offside crash barrier", 8: "Submerged in water",
            9: "Entered ditch", 10: "Other permanent object", 11: "Wall or fence",
            99: "Unknown", -1: "Missing"
        },
        'first_point_of_impact': {
            0: "Did not impact", 1: "Front", 2: "Back", 3: "Offside",
            4: "Nearside", 9: "Unknown", -1: "Missing"
        }
    }
    
    # Apply mappings to each column
    for column, mapping in mappings.items():
        if column in df.columns:
            df[column] = df[column].map(mapping).fillna(df[column])
    
    # === STEP 5: FILTER FOR CARS ONLY ===
    print("Filtering for car data only...")
    initial_count = len(df)
    
    if 'vehicle_type' in df.columns:
        df = df[df['vehicle_type'] == 'Car'].copy()
        print(f"Filtered to cars: {initial_count:,} -> {len(df):,} records")
    
    # === STEP 6: ADD LOCATION DETAILS ===
    print("Adding location details using reverse geocoding...")
    
    # Initialize location columns
    df['city'] = 'Unknown'
    df['state'] = 'Unknown' 
    df['country'] = 'UK'
    
    # Check for valid coordinates
    if 'latitude' in df.columns and 'longitude' in df.columns:
        valid_coords = (df['latitude'].notna() & df['longitude'].notna() & 
                       (df['latitude'] != 0) & (df['longitude'] != 0))
        
        if valid_coords.any():
            print(f"Processing {valid_coords.sum():,} records with valid coordinates")
            valid_df = df[valid_coords].copy()
            
            # Process in batches to avoid API limits
            batch_size = 1000
            with tqdm(total=len(valid_df), desc="Reverse geocoding") as pbar:
                for i in range(0, len(valid_df), batch_size):
                    batch = valid_df.iloc[i:i+batch_size]
                    
                    try:
                        coords = [(row['latitude'], row['longitude']) for _, row in batch.iterrows()]
                        locations = reverse_geocode.search(coords)
                        
                        for j, location in enumerate(locations):
                            idx = batch.index[j]
                            df.loc[idx, 'city'] = location.get('city', 'Unknown')
                            df.loc[idx, 'state'] = location.get('county', 'Unknown')
                            df.loc[idx, 'country'] = location.get('country', 'UK')
                    
                    except Exception as e:
                        print(f"Warning: Error in batch {i//batch_size + 1}: {e}")
                        continue
                    
                    pbar.update(len(batch))
    
    # === STEP 7: FIX DATA TYPES ===
    print("Fixing data types...")
    
    # Convert numeric columns
    numeric_columns = ['speed_limit', 'age_of_vehicle']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert date and time
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
        if df['date'].isna().any():
            df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True, errors='coerce')
    
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], format='%H:%M', errors='coerce').dt.time
    
    # === STEP 8: STANDARDIZE AND CLEAN ===
    print("Standardizing columns and cleaning data...")
    
    # Remove rows with missing/unknown values
    initial_rows = len(df)
    df = df[~df.isin([-1, 'Missing', 'Unknown']).any(axis=1)].copy()
    print(f"Removed {initial_rows - len(df):,} rows with missing/unknown values")
    
    # Extract make and model from generic_make_model
    if 'generic_make_model' in df.columns:
        df[['make', 'model']] = df['generic_make_model'].str.extract(r'^(\S+)\s*(.*)$')
        df['model'] = df['model'].fillna('Unknown')
    
    # Calculate model year
    if 'date' in df.columns and 'age_of_vehicle' in df.columns:
        df['accident_year'] = df['date'].dt.year
        df['model_year'] = df['accident_year'] - df['age_of_vehicle']
        # Clean unrealistic model years
        df.loc[(df['model_year'] < 1900) | (df['model_year'] > df['accident_year']), 'model_year'] = np.nan
    
    # === STEP 9: SELECT FINAL COLUMNS AND RENAME ===
    print("Selecting final columns and renaming for standardization...")
    
    final_columns = [
        'accident_index', 'accident_reference', 'accident_severity', 'date', 'time',
        'road_type', 'speed_limit', 'light_conditions', 'weather_conditions',
        'road_surface_conditions', 'vehicle_manoeuvre', 'hit_object_in_carriageway',
        'first_point_of_impact', 'city', 'state', 'country', 'make', 'model', 'model_year'
    ]
    
    # Keep only available columns
    available_final_cols = [col for col in final_columns if col in df.columns]
    df = df[available_final_cols].copy()
    
    # Rename columns to match NHTSA format
    rename_dict = {
        'accident_index': 'Report ID',
        'accident_reference': 'Report Version',
        'make': 'Make',
        'model': 'Model',
        'model_year': 'Model Year',
        'date': 'Incident Date',
        'time': 'Incident Time (24:00)',
        'city': 'City',
        'state': 'State',
        'country': 'Country',
        'road_type': 'Roadway Type',
        'road_surface_conditions': 'Roadway Surface',
        'speed_limit': 'Posted Speed Limit (MPH)',
        'light_conditions': 'Lighting',
        'hit_object_in_carriageway': 'Crash With',
        'accident_severity': 'Highest Injury Severity Alleged',
        'vehicle_manoeuvre': 'SV Pre-Crash Movement',
        'weather_conditions': 'Weather',
        'first_point_of_impact': 'SV Contact Area'
    }
    
    df.rename(columns=rename_dict, inplace=True)
    
    # Add ADS columns for consistency with NHTSA data
    df['ADS Equipped?'] = 'Conventional'
    df['Automation System Engaged?'] = 'Conventional'
    df['Source'] = 'Conventional'
    
    # Apply value standardization to match NHTSA format
    standardization_maps = {
        'Highest Injury Severity Alleged': {'Fatal': 'Fatality', 'Slight': 'Minor'},
        'SV Pre-Crash Movement': {
            'Turning left': 'Making Left Turn',
            'Turning right': 'Making Right Turn',
            'Reversing': 'Backing',
            'Slowing or stopping': 'Stopping',
            'U-turn': 'Making U-Turn', 
            'Changing lane to right': 'Changing Lanes',
            'Changing lane to left': 'Changing Lanes',
            'Going ahead left-hand bend': 'Travelling around Bend',
            'Going ahead right-hand bend': 'Travelling around Bend',
        },
        'Lighting': {
            'Darkness - lights lit': 'Dark - Lighted',
            'Darkness - no lighting': 'Dark - Not Lighted',
            'Darkness - lighting unknown': 'Dark - Unknown Lighting',
        },
        'Crash With': {
            'Other object': 'Other Fixed Object',
            'Any animal (except ridden horse)': 'Animal',
        },
        'Roadway Surface': {
            'Frost or ice': 'Snow / Slush / Ice',
            'Snow': 'Snow / Slush / Ice',
            'Wet or damp': 'Wet',
        },
        'Weather': {
            'Fine no high winds': 'Clear',
            'Fine + high winds': 'Clear',
        }
    }
    
    for column, mapping in standardization_maps.items():
        if column in df.columns:
            df[column] = df[column].replace(mapping)
    
    # === STEP 10: FINAL VALIDATION AND CLEANUP ===
    print("Final data validation...")
    
    initial_rows = len(df)
    
    # Remove rows with too many missing values (>50%)
    missing_threshold = 0.5
    missing_ratio = df.isna().sum(axis=1) / len(df.columns)
    df = df[missing_ratio <= missing_threshold].copy()
    
    # Clean unrealistic values
    if 'Model Year' in df.columns:
        current_year = pd.Timestamp.now().year
        df = df[(df['Model Year'] >= 1900) & (df['Model Year'] <= current_year)].copy()
    
    if 'Posted Speed Limit (MPH)' in df.columns:
        df = df[(df['Posted Speed Limit (MPH)'] >= 5) & (df['Posted Speed Limit (MPH)'] <= 100)].copy()
    
    print(f"Final validation: {initial_rows:,} -> {len(df):,} records")
    
    # === STEP 11: SAVE DATA ===
    print("Saving cleaned data...")
    
    output_path = base_path / 'loaded_data.csv'
    df.to_csv(output_path, index=False)
    
    # Save metadata
    metadata = {
        'total_records': len(df),
        'columns': list(df.columns),
        'processing_date': pd.Timestamp.now().isoformat(),
        'data_types': df.dtypes.to_dict()
    }
    
    metadata_path = base_path / 'loaded_data_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # === FINAL SUMMARY ===
    end_time = time.time()
    print(f"\n✅ UK data cleaning completed successfully!")
    print(f"⏱️  Processing time: {end_time - start_time:.2f} seconds")
    print(f"📊 Final dataset shape: {df.shape}")
    print(f"💾 Data saved to: {output_path}")
    print(f"📋 Metadata saved to: {metadata_path}")
    
    return df

# === MAIN EXECUTION ===
def main():
    """Main function to run the data cleaning"""
    try:
        # Run the cleaning process
        df = clean_uk_data()
        
        # Display sample of cleaned data
        print(f"\nSample of cleaned data:")
        print(tabulate(df.head(), headers='keys', tablefmt='grid', showindex=False))
        return df
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    df = main()