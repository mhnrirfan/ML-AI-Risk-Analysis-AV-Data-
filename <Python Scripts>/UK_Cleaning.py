# Filename: UK_Cleaning_Simplified.py
"""
Description: Simplified Python Cleaning Script for UK Road Collision Data
Inputs: UK DfT Road Casualty Statistics - Collision and Vehicle datasets
Outputs: Cleaned and standardized dataset for EDA 
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import itertools
from tabulate import tabulate
import matplotlib.pyplot as plt
import reverse_geocode
import os


class UKDataClass:
    """
    Purpose: Class to handle loading, cleaning, and processing UK Road Collision Data
    - load_datasets: Load datasets from specified directory
    - Running the methods for data cleaning pipeline
    """
    
    def __init__(self, data_dir):
        """
        Purpose: Constructor to the UKDataClass
        - contains the attributes of the dataframes where these methods can be applied to
        """
        self.data_dir = data_dir
        self.df_collisions = None
        self.df_vehicles = None
        self.merged_df = None

    def load_datasets(self):
        """
        Purpose: Basic Loading of the datasets found in the data directory folders
        """
        try:
            self.df_collisions = pd.read_csv(os.path.join(self.data_dir, 'dft-road-casualty-statistics-collision-last-5-years.csv'))
            self.df_vehicles = pd.read_csv(os.path.join(self.data_dir, 'dft-road-casualty-statistics-vehicle-last-5-years.csv'))
        except FileNotFoundError as e:
            print(f"Error loading datasets: {e}")
            raise
    
    def find_primary_keys(self, df):
        """
        Purpose: Finding the primary keys in each dataframe 
        Methods:
            - Check each row for this column is unique (count of column vals = number of rows)
            - If not, then use itertools to experiment with every combination to find what combination is unique
        Input: Dataframe 
        Output: Keys within a string for easy printing
        """
        # Check for single primary key
        for col in df.columns:
            if df[col].is_unique:
                return f"Primary key: ({col})"
        
        # Loop to check every combination
        for i in range(2, len(df.columns) + 1):
            for combo in itertools.combinations(df.columns, i):
                if df[list(combo)].drop_duplicates().shape[0] == df.shape[0]:
                    return f"Primary key: {combo}"
                
        return "No primary key found"
    
    def analyze_datasets(self):
        """
        Purpose: Analyze basic properties of the datasets from shape, duplicates, and missing values, primary keys
        """
        print("\nDataset Analysis Basic Information")
        
        # Find primary keys
        print("Primary Keys:")
        print("Collisions: ", self.find_primary_keys(self.df_collisions))
        print("Vehicles: ", self.find_primary_keys(self.df_vehicles))
        
        # Basic information about the datasets
        collision_keys = ['accident_index']
        vehicle_keys = ['accident_index', 'vehicle_reference']
        
        data = {
            "Dataset": ["COLLISIONS", "VEHICLES"],
            "Shape": [self.df_collisions.shape, self.df_vehicles.shape],
            "Unique Keys": [
                self.df_collisions[collision_keys].drop_duplicates().shape[0],
                self.df_vehicles[vehicle_keys].drop_duplicates().shape[0]
            ],
            "Duplicates": [
                self.df_collisions.duplicated(subset=collision_keys).sum(),
                self.df_vehicles.duplicated(subset=vehicle_keys).sum()
            ]
        }
        
        df_info = pd.DataFrame(data)
        print("\nDataset Summary:")
        print(tabulate(df_info, headers='keys', tablefmt='pretty'))
        
        # Missing values analysis
        self.analyze_missing_values()
    
    def missing_percentage(self, df):
        """
        Purpose: Calculate the percentage of missing values in a DataFrame.
        Input: DataFrame
        Output: Percentage of missing values in the DataFrame
        """
        total_cells = df.size
        total_missing = df.isnull().sum().sum()
        return (total_missing / total_cells) * 100
    
    def analyze_missing_values(self):
        """
        Purpose: Find how much missing data is in each dataset
        """
        print("\nMissing Value Per Dataset")
        print(f"Collisons missing: {self.missing_percentage(self.df_collisions):.2f}%")
        print(f"Vehicles missing: {self.missing_percentage(self.df_vehicles):.2f}%")
    
    def merge_datasets(self):
        """
        Purpose: Merging the collision and vehicle datasets into a single DataFrame
        """
        print("\nMerging Datasets")
        
        print(f"Shape of df_collisions: {self.df_collisions.shape}")
        print(f"Shape of df_vehicles: {self.df_vehicles.shape}")
        
        # Merge on accident_index (common key)
        self.merged_df = pd.merge(self.df_collisions, self.df_vehicles, on='accident_index', how='inner', suffixes=('', '_drop'))
        print(f"Shape of merged dataframe: {self.merged_df.shape}")
        
        # Remove duplicates
        self.merged_df = self.merged_df.drop_duplicates(subset='accident_index')
    
    def select_relevant_columns(self):
        """
        Purpose: Select only the columns needed for analysis
        """
        print("\nSelecting Relevant Columns")
        
        columns_to_keep = [
            'accident_index', 'accident_reference', 'latitude', 'longitude', 
            'location_easting_osgr', 'location_northing_osgr', 'accident_severity', 
            'date', 'time', 'road_type', 'speed_limit', 'light_conditions', 
            'vehicle_type', 'age_of_vehicle', 'weather_conditions', 
            'road_surface_conditions', 'vehicle_reference', 'vehicle_manoeuvre', 
            'hit_object_in_carriageway', 'hit_object_off_carriageway', 
            'first_point_of_impact', 'generic_make_model'
        ]
        
        self.merged_df = self.merged_df[columns_to_keep]
        print(f"Shape after column selection: {self.merged_df.shape}")
    
    def create_mappings(self):
        """
        Purpose: Create mapping dictionaries for categorical variables
        """
        return {
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
                1: "Reversing", 2: "Parked", 3: "Waiting to go - held up", 4: "Slowing or stopping",
                5: "Moving off", 6: "U-turn", 7: "Turning left", 8: "Waiting to turn left",
                9: "Turning right", 10: "Waiting to turn right", 11: "Changing lane to left",
                12: "Changing lane to right", 13: "Overtaking moving vehicle - offside",
                14: "Overtaking static vehicle - offside", 15: "Overtaking - nearside",
                16: "Going ahead left-hand bend", 17: "Going ahead right-hand bend",
                18: "Going ahead other", 99: "Unknown", -1: "Missing"
            },
            'hit_object_in_carriageway': {
                0: "None", 1: "Previous accident", 2: "Road works", 4: "Parked vehicle",
                5: "Bridge (roof)", 6: "Bridge (side)", 7: "Bollard or refuge",
                8: "Open door of vehicle", 9: "Central island of roundabout", 10: "Kerb",
                11: "Other object", 12: "Any animal (except ridden horse)",
                99: "Unknown", -1: "Missing"
            },
            'hit_object_off_carriageway': {
                0: "None", 1: "Road sign or traffic signal", 2: "Lamp post",
                3: "Telegraph or electricity pole", 4: "Tree", 5: "Bus stop or bus shelter",
                6: "Central crash barrier", 7: "Near/Offside crash barrier",
                8: "Submerged in water", 9: "Entered ditch", 10: "Other permanent object",
                11: "Wall or fence", 99: "Unknown", -1: "Missing"
            },
            'first_point_of_impact': {
                0: "Did not impact", 1: "Front", 2: "Back", 3: "Offside",
                4: "Nearside", 9: "Unknown", -1: "Missing"
            }
        }
    
    def apply_mappings(self):
        """
        Purpose: Apply categorical mappings to dataframe
        """
        print("\nApplying Categorical Mappings")
        
        # Fill NaN values with -1 for easier mapping
        self.merged_df = self.merged_df.fillna(-1)
        
        mappings = self.create_mappings()
        for column, mapping in mappings.items():
            if column in self.merged_df.columns:
                self.merged_df[column] = self.merged_df[column].map(mapping)
        

    
    def filter_cars_only(self):
        """
        Purpose: Filter dataset to include only cars
        """
        print("\nFiltering for Cars Only")
        
        initial_shape = self.merged_df.shape
        self.merged_df = self.merged_df[self.merged_df['vehicle_type'] == 'Car']
        print(f"Shape before filtering: {initial_shape}")
        print(f"Shape after filtering for cars: {self.merged_df.shape}")
    
    def count_and_sum_missing_values(self):
        """
        Purpose: Count and sum missing, empty, NaN, and 'unknown' values for each column in the DataFrame.
        Input: DataFrame
        Output: Prints a summary table with the counts and percentages of missing values.
        """
        print("\nMissing Values Summary")
        
        overall_summary = []
        for col in self.merged_df.columns:
            null_count = self.merged_df[col].isnull().sum()
            empty_count = (self.merged_df[col] == '').sum()
            nan_count = self.merged_df[col].apply(lambda x: pd.isna(x)).sum()
            unknown_count = self.merged_df[col].apply(lambda x: str(x).strip().lower() == 'unknown').sum()
            missing_count = self.merged_df[col].apply(lambda x: str(x).strip().lower() == 'missing').sum()
            total = null_count + empty_count + nan_count + unknown_count + missing_count
            percentage = (total / len(self.merged_df)) * 100
            overall_summary.append((col, null_count, empty_count, nan_count, unknown_count, missing_count, total, percentage))
        
        summary_df = pd.DataFrame(overall_summary, columns=['Column', 'Null', 'Empty', 'NaN', 'Unknown', 'Missing', 'Total', 'Percentage'])
        summary_df = summary_df.sort_values(by='Percentage', ascending=False)
        
        print("Top 10 columns with highest missing values:")
        print(tabulate(summary_df.head(10), headers='keys', tablefmt='grid'))
        return summary_df
    
    def clean_missing_data(self):
        """
        Purpose: Replace 'Unknown' and 'Missing' with NaN and drop rows with any NaN values
        """
        print("\nCleaning Missing Data")
        
        initial_shape = self.merged_df.shape
        
        # Replace 'Unknown' and 'Missing' with NaN
        self.merged_df.replace(['Unknown', 'Missing', '-1'], pd.NA, inplace=True)
        
        # Drop rows with any NaN values
        self.merged_df.dropna(inplace=True)

    
    def add_location_data(self):
        """
        Purpose: Add city, state, country using reverse geocoding
        """
        print("\nAdding Location Data (takes up to 5mins)")
        
        def get_location_offline(row):
            try:
                coords = (row['latitude'], row['longitude'])
                location = reverse_geocode.search([coords])[0]
                return pd.Series([location['city'], location['state'], location['country']])
            except:
                return pd.Series(['Unknown', 'Unknown', 'Unknown'])
        
        self.merged_df[['city', 'state', 'country']] = self.merged_df.apply(get_location_offline, axis=1)
    
    def process_data_types(self):
        """
        Purpose: Convert columns to appropriate data types
        """
        print("\nProcessing Data Types")
        
        # Convert all columns to string first
        self.merged_df = self.merged_df.astype(str)
        
        # Convert specific columns to appropriate types
        self.merged_df['speed_limit'] = pd.to_numeric(self.merged_df['speed_limit'], errors='coerce')
        self.merged_df['age_of_vehicle'] = pd.to_numeric(self.merged_df['age_of_vehicle'], errors='coerce')
        
        # Convert date and time
        self.merged_df['date'] = pd.to_datetime(self.merged_df['date'], format='%d/%m/%Y', errors='coerce')
        self.merged_df['time'] = pd.to_datetime(self.merged_df['time'], format='%H:%M', errors='coerce').dt.time
        
    
    def process_vehicle_details(self):
        """
        Purpose: Process vehicle make, model, and calculate model year
        """
        print("\nProcessing Vehicle Details")
        
        # Remove rows with placeholder -1 values
        self.merged_df = self.merged_df[~self.merged_df.isin([-1]).any(axis=1)]
        
        # Split 'generic_make_model' into 'make' and 'model'
        self.merged_df[['make', 'model']] = self.merged_df['generic_make_model'].str.extract(r'^(\S+)\s*(.*)$')
        
        # Ensure 'age_of_vehicle' is numeric
        self.merged_df['age_of_vehicle'] = pd.to_numeric(self.merged_df['age_of_vehicle'], errors='coerce')
        
        # Convert 'date' to datetime
        self.merged_df['incident_date'] = pd.to_datetime(self.merged_df['date'], errors='coerce')
        
        # Calculate model date by subtracting age from incident date
        self.merged_df['model_date'] = self.merged_df['incident_date'] - self.merged_df['age_of_vehicle'].apply(
            lambda x: pd.Timedelta(days=365.25 * x) if pd.notnull(x) else pd.NaT
        )
        
        # Extract model year from model_date
        self.merged_df['model_year'] = self.merged_df['model_date'].dt.year
        self.merged_df['model_year'] = self.merged_df['model_year'].where(self.merged_df['model_year'].notna(), 'Unknown')
        
    
    def create_vehicle_age_plot(self):
        """
        Purpose: Create boxplot of vehicle age at accident
        """
        print("\nCreating Vehicle Age Boxplot")
        
        age_data = self.merged_df['age_of_vehicle'].dropna()
        plt.figure(figsize=(8, 6))
        plt.boxplot(age_data, vert=True, patch_artist=True)
        plt.title('Boxplot of Vehicle Age at Accident')
        plt.ylabel('Vehicle Age (years)')
        plt.grid(True)
        plt.show()
    
    def select_final_columns(self):
        """
        Purpose: Select final columns for output
        """
        print("\nSelecting Final Columns")
        
        columns_to_keep = [
            'accident_index', 'accident_reference', 'accident_severity', 'date', 'time',
            'road_type', 'speed_limit', 'light_conditions', 'weather_conditions',
            'road_surface_conditions', 'vehicle_manoeuvre', 'hit_object_in_carriageway',
            'first_point_of_impact', 'city', 'state', 'country', 'latitude', 'longitude',
            'make', 'model', 'model_year'
        ]
        
        # Keep only columns that exist in the dataframe
        available_columns = [col for col in columns_to_keep if col in self.merged_df.columns]
        self.merged_df = self.merged_df[available_columns]
        
        print(f"Final column selection complete. Shape: {self.merged_df.shape}")
    
    def standardize_column_names(self):
        """
        Purpose: Rename columns to match standardized format
        """
        print("\nStandardizing Column Names")
        
        column_rename_mapping = {
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
        
        # Rename columns that exist in the dataframe
        existing_mappings = {old: new for old, new in column_rename_mapping.items() if old in self.merged_df.columns}
        self.merged_df.rename(columns=existing_mappings, inplace=True)
        
        # Add ADS columns
        self.merged_df['ADS Equipped?'] = 'Conventional'
        self.merged_df['Automation System Engaged?'] = 'Conventional'
        
    
    def standardize_values(self):
        """
        Purpose: Standardize values to match expected format
        """
        print("\nStandardizing Values")
        
        # Severity mapping
        if 'Highest Injury Severity Alleged' in self.merged_df.columns:
            severity_mapping = {'Fatal': 'Fatality', 'Slight': 'Minor'}
            self.merged_df['Highest Injury Severity Alleged'] = self.merged_df['Highest Injury Severity Alleged'].replace(severity_mapping)
        
        # Movement mapping
        if 'SV Pre-Crash Movement' in self.merged_df.columns:
            movement_mapping = {
                'Turning left': 'Making Left Turn',
                'Turning right': 'Making Right Turn',
                'Reversing': 'Backing',
                'Slowing or stopping': 'Stopping',
                'U-turn': 'Making U-Turn',
                'Changing lane to right': 'Changing Lanes',
                'Changing lane to left': 'Changing Lanes',
                'Going ahead left-hand bend': 'Travelling around Bend',
                'Going ahead right-hand bend': 'Travelling around Bend'
            }
            self.merged_df['SV Pre-Crash Movement'] = self.merged_df['SV Pre-Crash Movement'].replace(movement_mapping)
        
        # Lighting mapping
        if 'Lighting' in self.merged_df.columns:
            lighting_mapping = {
                'Darkness - lights lit': 'Dark - Lighted',
                'Darkness - no lighting': 'Dark - Not Lighted',
                'Darkness - lighting unknown': 'Dark - Unknown Lighting'
            }
            self.merged_df['Lighting'] = self.merged_df['Lighting'].replace(lighting_mapping)
        
        # Crash object mapping
        if 'Crash With' in self.merged_df.columns:
            crash_mapping = {
                'Other object': 'Other Fixed Object',
                'Any animal (except ridden horse)': 'Animal',
                'None': 'No Object'
            }
            self.merged_df['Crash With'] = self.merged_df['Crash With'].replace(crash_mapping)
        
        # Surface condition mapping
        if 'Roadway Surface' in self.merged_df.columns:
            surface_mapping = {
                'Frost or ice': 'Snow / Slush / Ice',
                'Snow': 'Snow / Slush / Ice',
                'Wet or damp': 'Wet'
            }
            self.merged_df['Roadway Surface'] = self.merged_df['Roadway Surface'].replace(surface_mapping)
        
        # Weather mapping
        if 'Weather' in self.merged_df.columns:
            weather_mapping = {
                'Fine no high winds': 'Clear',
                'Fine + high winds': 'Clear'
            }
            self.merged_df['Weather'] = self.merged_df['Weather'].replace(weather_mapping)
        
    
    def reorder_columns(self):
        """
        Purpose: Reorder columns to match desired output format
        """
        print("\nReordering Columns")
        
        desired_order = [
            'Report ID', 'Report Version', 'Make', 'Model', 'Model Year', 'ADS Equipped?',
            'Automation System Engaged?', 'Incident Date', 'Incident Time (24:00)', 'City',
            'State', 'Roadway Type', 'Roadway Surface', 'Posted Speed Limit (MPH)', 'Lighting',
            'Crash With', 'Highest Injury Severity Alleged', 'SV Pre-Crash Movement',
            'Weather', 'SV Contact Area', 'Country', 'latitude', 'longitude'
        ]
        
        # Keep only columns that exist in the dataframe
        available_columns = [col for col in desired_order if col in self.merged_df.columns]
        self.merged_df = self.merged_df[available_columns]
        
    
    def save_cleaned_data(self, output_path):
        """
        Purpose: Save the cleaned DataFrame to a CSV file
        Input: The output file path where the cleaned data will be saved
        Output: CSV file containing the cleaned dataset
        """
        self.merged_df.to_csv(output_path, index=False)
        print(f"Final dataset shape: {self.merged_df.shape}")
        print(f"Data saved to: {output_path}")
    
    def run_full_pipeline(self, output_path):
        """
        Description: Running all the functions in sequence to process the dataset
        Input: The dataset folder path 
        Output: Output cleaned dataset file into the specified path
        """
        try:
            self.load_datasets()  # Step 1: Load data
            self.analyze_datasets()  # Step 2: Analyze datasets
            self.merge_datasets()  # Step 3: Merge datasets
            self.select_relevant_columns()  # Step 4: Select relevant columns
            self.apply_mappings()  # Step 5: Apply categorical mappings
            self.filter_cars_only()  # Step 6: Filter for cars only
            self.count_and_sum_missing_values()  # Step 7: Analyze missing values
            self.clean_missing_data()  # Step 8: Clean missing data
            self.add_location_data()  # Step 9: Add location data
            self.process_data_types()  # Step 10: Process data types
            self.process_vehicle_details()  # Step 11: Process vehicle details
            # self.create_vehicle_age_plot()  # Step 12: Create plot (optional)
            self.select_final_columns()  # Step 13: Select final columns
            self.standardize_column_names()  # Step 14: Standardize column names
            self.standardize_values()  # Step 15: Standardize values
            self.reorder_columns()  # Step 16: Reorder columns
            self.save_cleaned_data(output_path)  # Step 17: Save cleaned data
            
            # Output the column names of the cleaned dataset
            print("\nFinal Column Names:")
            print(self.merged_df.columns.tolist())
            
        except Exception as e:  # Catch any errors
            print(f"Cleaning Failed: {str(e)}")
            raise


def main():
    """
    Description: Main function to run the UK data cleaning pipeline 
    - processor sets up the data class 
    - functions to run all the methods in the class
    """
    Dataset_Folder_Path = "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK"  # Update this path - Where the Datasets are stored
    Output_Saving_Folder_Path = "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK/UK-cleaned_data.csv"  # Update this path
    
    # Making and processing the dataset
    processor = UKDataClass(Dataset_Folder_Path)
    # Running all the processing steps
    processor.run_full_pipeline(Output_Saving_Folder_Path)


if __name__ == "__main__":
    main()