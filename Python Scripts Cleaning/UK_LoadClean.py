# Import necessary libraries
import pandas as pd
import itertools
from tabulate import tabulate
import time
from tqdm import tqdm
import reverse_geocode

# Load datasets
collisions = pd.read_csv('/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK/dft-road-casualty-statistics-collision-last-5-years.csv')
vehicles = pd.read_csv('/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK/dft-road-casualty-statistics-vehicle-last-5-years.csv')

def find_primary_key(df1, df2):
    """
    Find primary key(s) between two dataframes by checking unique columns or unique combinations.
    """
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Check single column uniqueness
    for col in combined_df.columns:
        if combined_df[col].is_unique:
            return (col,)
    # Check combinations of columns
    for i in range(2, len(combined_df.columns) + 1):
        for combo in itertools.combinations(combined_df.columns, i):
            if combined_df.duplicated(subset=list(combo)).sum() == 0:
                return combo[:1]
    return None

def drop_unnecessary_columns(collisions, vehicles):
    # Merge datasets on common key (usually 'accident_index')
    df = pd.merge(collisions, vehicles, on='accident_index', how='inner', suffixes=('', '_drop'))

    # Print preview of merged data
    print(tabulate(df.head(10), headers='keys', tablefmt='grid', showindex=True))

    # Drop duplicates by accident_index to avoid redundant rows
    df = df.drop_duplicates(subset='accident_index')

    columns_to_keep = ['accident_index', 'accident_reference', 'latitude', 'longitude', 'location_easting_osgr', 'location_northing_osgr',
                       'accident_severity', 'date', 'time', 'road_type', 'speed_limit',
                       'light_conditions', 'vehicle_type',
                       'weather_conditions', 'road_surface_conditions', 'vehicle_reference','vehicle_manoeuvre', 'hit_object_in_carriageway',
                       'hit_object_off_carriageway', 'first_point_of_impact', 'age_of_vehicle', 'generic_make_model']

    return df[columns_to_keep]

def mapping_values_glossary(df):
    df = df.fillna(-1)  # Fill NaNs for mapping

    accident_severity_mapping = {1: "Fatal", 2: "Serious", 3: "Slight"}
    road_type_mapping = {1: "Roundabout", 2: "One way street", 3: "Dual carriageway", 6: "Single carriageway", 7: "Slip road", 9: "Unknown", 12: "One way street/Slip road", -1: "Missing"}
    light_conditions_mapping = {1: "Daylight", 4: "Darkness - lights lit", 5: "Darkness - lights unlit", 6: "Darkness - no lighting", 7: "Darkness - lighting unknown", -1: "Missing"}
    weather_conditions_mapping = {1: "Fine no high winds", 2: "Raining no high winds", 3: "Snowing no high winds", 4: "Fine + high winds", 5: "Raining + high winds", 6: "Snowing + high winds", 7: "Fog or mist", 8: "Other", 9: "Unknown", -1: "Missing"}
    road_surface_conditions_mapping = {1: "Dry", 2: "Wet or damp", 3: "Snow", 4: "Frost or ice", 5: "Flood over 3cm. deep", 6: "Oil or diesel", 7: "Mud", 9: "Unknown", -1: "Missing"}
    vehicle_type_mapping = {1: "Pedal cycle", 2: "Motorcycle 50cc and under", 3: "Motorcycle 125cc and under", 4: "Motorcycle over 125cc and up to 500cc", 5: "Motorcycle over 500cc",
                            8: "Taxi/Private hire car", 9: "Car", 10: "Minibus (8 - 16 passenger seats)", 11: "Bus or coach (17 or more pass seats)", 16: "Ridden horse",
                            17: "Agricultural vehicle", 18: "Tram", 19: "Van / Goods 3.5 tonnes mgw or under", 20: "Goods over 3.5t. and under 7.5t", 21: "Goods 7.5 tonnes mgw and over",
                            22: "Mobility scooter", 23: "Electric motorcycle", 90: "Other vehicle", 97: "Motorcycle - unknown cc", 98: "Goods vehicle - unknown weight", 99: "Unknown", -1: "Missing"}
    vehicle_manoeuvre_mapping = {1: "Reversing", 2: "Parked", 3: "Waiting to go - held up", 4: "Slowing or stopping", 5: "Moving off", 6: "U-turn",
                                7: "Turning left", 8: "Waiting to turn left", 9: "Turning right", 10: "Waiting to turn right", 11: "Changing lane to left", 12: "Changing lane to right",
                                13: "Overtaking moving vehicle - offside", 14: "Overtaking static vehicle - offside", 15: "Overtaking - nearside", 16: "Going ahead left-hand bend",
                                17: "Going ahead right-hand bend", 18: "Going ahead other", 99: "Unknown", -1: "Missing"}
    hit_object_in_carriageway_mapping = {0: "None", 1: "Previous accident", 2: "Road works", 4: "Parked vehicle", 5: "Bridge (roof)", 6: "Bridge (side)", 7: "Bollard or refuge",
                                         8: "Open door of vehicle", 9: "Central island of roundabout", 10: "Kerb", 11: "Other object", 12: "Any animal (except ridden horse)", 99: "Unknown", -1: "Missing"}
    hit_object_off_carriageway_mapping = {0: "None", 1: "Road sign or traffic signal", 2: "Lamp post", 3: "Telegraph or electricity pole", 4: "Tree", 5: "Bus stop or bus shelter", 6: "Central crash barrier",
                                          7: "Near/Offside crash barrier", 8: "Submerged in water", 9: "Entered ditch", 10: "Other permanent object", 11: "Wall or fence", 99: "Unknown", -1: "Missing"}
    first_point_of_impact_mapping = {0: "Did not impact", 1: "Front", 2: "Back", 3: "Offside", 4: "Nearside", 9: "Unknown", -1: "Missing"}

    df["accident_severity"] = df["accident_severity"].map(accident_severity_mapping)
    df["road_type"] = df["road_type"].map(road_type_mapping)
    df["light_conditions"] = df["light_conditions"].map(light_conditions_mapping)
    df["weather_conditions"] = df["weather_conditions"].map(weather_conditions_mapping)
    df["road_surface_conditions"] = df["road_surface_conditions"].map(road_surface_conditions_mapping)
    df["vehicle_type"] = df["vehicle_type"].map(vehicle_type_mapping)
    df["vehicle_manoeuvre"] = df["vehicle_manoeuvre"].map(vehicle_manoeuvre_mapping)
    df["hit_object_in_carriageway"] = df["hit_object_in_carriageway"].map(hit_object_in_carriageway_mapping)
    df["hit_object_off_carriageway"] = df["hit_object_off_carriageway"].map(hit_object_off_carriageway_mapping)
    df["first_point_of_impact"] = df["first_point_of_impact"].map(first_point_of_impact_mapping)

    return df

def get_location_details(df):
    tqdm.pandas(desc="Processing rows")
    start_time = time.time()

    def fetch_location(row):
        try:
            location = reverse_geocode.search([(row['latitude'], row['longitude'])])[0]
            return pd.Series([location['city'], location['county'], location['country']])
        except Exception:
            return pd.Series(['Unknown', 'Unknown', 'Unknown'])

    df[['city', 'state', 'country']] = df.progress_apply(fetch_location, axis=1)

    end_time = time.time()
    print(f"Time taken for reverse geocoding: {end_time - start_time:.2f} seconds")
    return df

def fix_formats(df):
    df['speed_limit'] = df['speed_limit'].astype(float)
    df['age_of_vehicle'] = df['age_of_vehicle'].astype(float)

    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
    df['time'] = pd.to_datetime(df['time'], format='%H:%M', errors='coerce').dt.time
    return df

def standardise_columns(df):
    df = df[~df.isin([-1]).any(axis=1)]

    # Extract make and model from generic_make_model
    df[['make', 'model']] = df['generic_make_model'].str.extract(r'^(\S+)\s*(.*)$')

    df['accident_year'] = df['date'].dt.year
    df['model_year'] = df['accident_year'] - df['age_of_vehicle']

    columns_to_keep = ['accident_index', 'accident_reference', 'accident_severity', 'date', 'time',
                       'road_type', 'speed_limit', 'light_conditions', 'weather_conditions', 'road_surface_conditions',
                       'vehicle_manoeuvre', 'hit_object_in_carriageway', 'first_point_of_impact', 'city', 'state', 'country',
                       'latitude', 'longitude', 'make', 'model', 'model_year']

    df = df[columns_to_keep]

    # Rename columns for standardisation
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

    df = df.drop(columns=['latitude', 'longitude'])

    # Add ADS columns defaulting to Conventional
    df['ADS Equipped?'] = 'Conventional'
    df['Automation System Engaged?'] = 'Conventional'

    # Standardize some categorical values
    df['Highest Injury Severity Alleged'] = df['Highest Injury Severity Alleged'].replace({'Fatal': 'Fatality', 'Slight': 'Minor'})

   
    df['SV Pre-Crash Movement'] = df['SV Pre-Crash Movement'].replace({
        'Turning left': 'Making Left Turn',
        'Turning right': 'Making Right Turn',
        'Reversing': 'Backing',
        'Slowing or stopping': 'Stopping',
        'U-turn': 'Making U-Turn',
        'Chnaging lane to right': 'Changing Lanes',
        'Chnaging lane to left': 'Changing Lanes',
        'Going ahead left-hand bend': 'Travelling around Bend',
        'Going ahead right-hand bend': 'Travelling around Bend',
        
    })


    df['Lighting'] = df['Lighting'].replace({
        'Darkness - lights lit': 'Dark - Lighted',
        'Darkness - no lighting': 'Dark - Not Lighted',
        'Darkness - lighting unknown': 'Dark - Unknown Lighting',
    })

    df['Crash With'] = df['Crash With'].replace({
        'Other object': 'Other Fixed Object',
        'Any animal (except ridden horse)': 'Animal',
    })

    df['Roadway Surface'] = df['Roadway Surface'].replace({
        'Frost or ice': 'Snow / Slush / Ice',
        'Snow': 'Snow / Slush / Ice',
        'Wet or damp': 'Wet',
    })

    df['Weather'] = df['Weather'].replace({
        'Fine no high winds': 'Clear',
        'Fine + high winds': 'Clear',
    })


    return df

def UK_Data_Cleaning(collisions, vehicles):
    # Identify primary key
    keys = find_primary_key(collisions, vehicles)
    print(f"Primary key(s) for merging: {keys}")

    # Merge collisions and vehicles on primary key
    df = pd.merge(collisions, vehicles, on=keys, how='inner', suffixes=('', '_drop'))

    # Drop duplicates by primary key to avoid redundancy
    df = df.drop_duplicates(subset=keys)

    # Drop unnecessary columns
    df = drop_unnecessary_columns(collisions, vehicles)

    # Filter for vehicle_type 'Car' only (value 9 as per mapping)
    df = mapping_values_glossary(df)
    df = df[df['vehicle_type'] == 'Car']

    # Remove unknown and missing values
    df.replace(['Unknown', 'Missing'], pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Add 'Source' column
    df['Source'] = 'Conventional'

    # Add location details by reverse geocoding lat/lon
    df = get_location_details(df)

    # Fix formats for speed_limit, age_of_vehicle, date and time
    df = fix_formats(df)

    # Standardise column names and other transformations
    df = standardise_columns(df)

    # Save cleaned data CSV
    df.to_csv('/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK/loaded_data.csv', index=False)

    return df

# Run the cleaning function on UK data
df = UK_Data_Cleaning(collisions, vehicles)
print(f"Shape of the cleaned data: {df.shape}")
print("UK data cleaning completed successfully.")
