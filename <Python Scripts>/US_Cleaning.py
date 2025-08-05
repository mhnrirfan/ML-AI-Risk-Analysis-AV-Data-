# Filename: US_Cleaning_Simplified.py
"""
Description: Simplified Python Cleaning Script for US NHSTA AV Incident Data
Inputs: Stats Collision and Vehicle Dataset from https://www.nhtsa.gov/nhtsa-datasets-and-apis
Outputs: Cleaned and standardized dataset for EDA 
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import itertools
from tabulate import tabulate
from dateutil import parser
import os


class USDataClass:
    """
    Purpose: Class to handle loading, cleaning, and processing US NHSTA AV Incident Data
    - load_datasets: Load datasets from specified directory
    - Running the methods for data cleaning pipeline
    """
    
    def __init__(self, data_dir):
        """
        Purpose: Constructor to the USDataClass
        - contains the attributes of the dataframes where these methods can be applied to
        Reference: https://realpython.com/python-classes/ for applying how to make dictionaries
        """
        # the 5 different  dataframes that will be used
        self.data_dir = data_dir
        self.df_adas = None
        self.df_ads = None
        self.df_other = None
        self.merged_df = None

    def load_datasets(self):
        """
        Purpose: Basic Loading of the datasets found in the data directory folders
        reference: 
        - https://www.youtube.com/watch?v=-EO2lHJdyzE for using data folder paths instead of hardcoding links
        - https://www.w3schools.com/python/python_try_except.asp for try except error custom error handling
        """
        try:
            self.df_adas = pd.read_csv(os.path.join(self.data_dir, 'SGO-2021-01_Incident_Reports_ADAS.csv'))
            self.df_ads = pd.read_csv(os.path.join(self.data_dir, 'SGO-2021-01_Incident_Reports_ADS.csv'))
            self.df_other = pd.read_csv(os.path.join(self.data_dir, 'SGO-2021-01_Incident_Reports_OTHER.csv'))
            print(" All datasets loaded successfully")
        except FileNotFoundError as e:
            print(f"Error loading datasets: {e}")
            raise
    
    def find_primary_keys_common(self, df, common_columns):
        """
        Purpose: Finding the primary keys in each dataframe using only common columns
        Methods:
            - Check each row for this column is unique (count of column vals = number of rows)
            - If not, then use itertools to experiment with every combination to find what combination is unique
        Input: Dataframe, Set of common columns
        Output: Keys within a string for easy printing
        Reference:
        - https://medium.com/@tkprof.h/find-uniq-conbinations-of-fields-from-csv-file-32a460e775fb how to find the primary key for a dataset and between datasets
        - https://docs.python.org/3/library/itertools.html understanding how itertool combination actually work

        """
        # Check for single primary key
        for col in common_columns:
            if df[col].is_unique:
                print(f"Primary key found in {col}")
                return f"Primary key: ({col})"
        
        # Loop to check every combination of the common columns
        for i in range(2, len(common_columns) + 1):
            for combo in itertools.combinations(common_columns, i):
                if df[list(combo)].drop_duplicates().shape[0] == df.shape[0]:
                    return f"Primary key: {combo}"
                
        return "No primary key found"
    
    def analyze_datasets(self):
        """
        Purpose: Analyze basic properties of the datasets from shape, duplcates, and missing values, primary keys
        """
        print("\nDataset Analysis Basic Information")
        
        # Find common columns across all three datasets
        common_columns = set(self.df_adas.columns).intersection(
            set(self.df_ads.columns)).intersection(set(self.df_other.columns))
        
        # Find primary keys
        print("PRIMARY KEYS:")
        print("ADAS: ", self.find_primary_keys_common(self.df_adas, common_columns))
        print("ADS: ", self.find_primary_keys_common(self.df_ads, common_columns))
        print("OTHER: ", self.find_primary_keys_common(self.df_other, common_columns))
        
        # Basic information about the datasets
        keys = ['Report ID', 'Report Version']
        data = {
            "Dataset": ["ADAS", "ADS", "OTHER"],
            "Shape": [self.df_adas.shape, self.df_ads.shape, self.df_other.shape],
            f"{keys}": [
                self.df_adas[keys].drop_duplicates().shape[0],
                self.df_ads[keys].drop_duplicates().shape[0],
                self.df_other[keys].drop_duplicates().shape[0]
            ],
            "Duplicates": [
                self.df_adas.duplicated(subset=keys).sum(),
                self.df_ads.duplicated(subset=keys).sum(),
                self.df_other.duplicated(subset=keys).sum()
            ]
        }
        
        df_info = pd.DataFrame(data)
        print("\nDataset Summary:")
        print(tabulate(df_info, headers='keys', tablefmt='pretty'))
        
        # Missing values analysis
        self.analyze_missing_values()
    
    def missing_percentage(self, df):
        """
        Purpose: Outputs a table of the missing values for each dataframe
        """
        total_cells = df.size
        total_missing = df.isnull().sum().sum()
        return (total_missing / total_cells) * 100
    
    def analyze_missing_values(self):
        """
        Purpose: find how much missing data is in each dataset
        Reference: https://www.w3schools.com/python/python_string_formatting.asp using place holder rounded 2 decimal places in formatted strings
        """
        print("\nMissing Value Per Dataset")
        print(f"ADAS missing: {self.missing_percentage(self.df_adas):.2f}%")
        print(f"ADS missing: {self.missing_percentage(self.df_ads):.2f}%")
        print(f"OTHER missing: {self.missing_percentage(self.df_other):.2f}%")
    
    def merge_datasets(self):
        """
        Purpose: Merging the ADAS and ADS datasets into a single DataFrame
        """
        print("\nMerging Datasets")
        
        print(f"Shape of df_adas: {self.df_adas.shape}")
        print(f"Shape of df_ads: {self.df_ads.shape}")
        
        # Add source column to know where data comes from
        self.df_adas['Source'] = 'ADAS'
        self.df_ads['Source'] = 'ADS'
        
        # Combine dataframes for adas and ads
        self.merged_df = pd.concat([self.df_adas, self.df_ads], ignore_index=True)
        print(f"Shape of merged dataframe: {self.merged_df.shape}")
    
        # Keep only the latest version of each report
        self.handle_multiple_versions()
    
    def handle_multiple_versions(self):
        """
        Purpose: Keep only the latest version of each report
        Reference: https://www.geeksforgeeks.org/pandas/python-pandas-dataframe-groupby/ how to actually combine the latest versions of each report by goruping by Report ID
        """
        print("\nKeep only latest versions of each report")
        
        # Count versions per report ID as this is primary key
        version_counts = self.merged_df.groupby('Report ID')['Report Version'].nunique()
        multi_version = version_counts[version_counts > 1]
        if len(multi_version) > 0: # if there are multiple versions
            multi_version = multi_version.sort_values(ascending=False)
        
        # Keep only latest versions
        self.merged_df = self.merged_df.loc[
            self.merged_df.groupby('Report ID')['Report Version'].idxmax()
        ].reset_index(drop=True)

        # should reduce to 4000 ish
        print(f"Shape after keeping latest versions: {self.merged_df.shape}") 
    
    def combine_and_drop(self, new_col_name, cols_to_combine):
        """  
        Purpose: Combine column values into 1, if they contain Y then place the column name 
        Input:
            merged_df: DataFrame to modify
            new_col_name: Name of the new column to create
            cols_to_combine: List of columns to combine
        Output: DataFrame with the new column and specified columns dropped
        """  
        def combine_values(row):
            """
            Purpose: Combine values from specified columns into a single string if they contain 'Y'.
            Input: Row of DataFrame
            Output: Combined string of column names where the value is 'Y'
            """
            combined_values = []
            for col in cols_to_combine:
                if str(row[col]).strip().upper() == 'Y': # if y then add the column name
                    combined_values.append(col.split(' - ')[-1])
            return ', '.join(combined_values)

        # Combine values from specified columns into a new column
        self.merged_df[new_col_name] = self.merged_df[cols_to_combine].apply(combine_values, axis=1)
        self.merged_df.drop(columns=cols_to_combine, inplace=True)
    
    def combine_related_columns(self):
        """
        Purpose: Columns with similar meanings are combined into one column
        Reference: https://www.geeksforgeeks.org/python-pandas-dataframe-combine-columns-into-one/ combine columns with similar meanings into one column
        """
        print("\nCombining Related Columns")
        
        # Define column combinations
        combinations = [
            ('CP Contact Area', [
                'CP Contact Area - Rear Left', 'CP Contact Area - Left', 'CP Contact Area - Front Left',
                'CP Contact Area - Rear', 'CP Contact Area - Top', 'CP Contact Area - Front',
                'CP Contact Area - Rear Right', 'CP Contact Area - Right', 'CP Contact Area - Front Right',
                'CP Contact Area - Bottom'
            ]),
            ('ADAS/ADS System Version', [
                'ADAS/ADS System Version', 'ADAS/ADS System Version - Unk', 'ADAS/ADS System Version CBI'
            ]),
            ('ADAS/ADS Hardware Version', [
                'ADAS/ADS Hardware Version', 'ADAS/ADS Hardware Version - Unk', 'ADAS/ADS Hardware Version CBI'
            ]),
            ('ADAS/ADS Software Version', [
                'ADAS/ADS Software Version', 'ADAS/ADS Software Version - Unk', 'ADAS/ADS Software Version CBI'
            ]),
            ('Other Reporting Entities', [
                'Other Reporting Entities?', 'Other Reporting Entities? - Unk', 'Other Reporting Entities? - NA'
            ]),
            ('Federal Regulatory Exemption', [
                'Federal Regulatory Exemption?', 'Other Federal Reg. Exemption',
                'Federal Reg. Exemption - Unk', 'Federal Reg. Exemption - No'
            ]),
            ('State or Local Permit', [
                'State or Local Permit?', 'State or Local Permit'
            ]),
            ('Source', [
                'Source - Complaint/Claim', 'Source - Telematics', 'Source - Law Enforcement',
                'Source - Field Report', 'Source - Testing', 'Source - Media',
                'Source - Other', 'Source - Other Text'
            ]),
            ('Weather', [
                'Weather - Clear', 'Weather - Snow', 'Weather - Cloudy', 'Weather - Fog/Smoke',
                'Weather - Rain', 'Weather - Severe Wind', 'Weather - Other', 'Weather - Other Text'
            ]),
            ('SV Contact Area', [
                'SV Contact Area - Rear Left', 'SV Contact Area - Left', 'SV Contact Area - Front Left',
                'SV Contact Area - Rear', 'SV Contact Area - Top', 'SV Contact Area - Front',
                'SV Contact Area - Rear Right', 'SV Contact Area - Right', 'SV Contact Area - Front Right',
                'SV Contact Area - Bottom'
            ]),
            ('Data Availability', [
                'Data Availability - EDR', 'Data Availability - Police Rpt', 'Data Availability - Telematics',
                'Data Availability - Complaints', 'Data Availability - Video', 'Data Availability - Other',
                'Data Availability - No Data'
            ])
        ]
        
        # Apply combinations
        for new_col, cols in combinations: # go through all columns in list
            # Check if columns exist before combining
            existing_cols = []
            for col in cols: 
                if col in self.merged_df.columns:
                    existing_cols.append(col)
            if existing_cols:
                self.combine_and_drop(new_col, existing_cols)
        
        print(f"Shape after combining columns: {self.merged_df.shape}")
    
    def drop_unnecessary_columns(self):
        """
        Purpose: Drop columns we don't need for analysis
        Reference:
        https://pandas.pydata.org/docs/user_guide/indexing.html for lookup and logical not so perform element-wise logical not oppostie for those who dont contain unknown
        - https://stackoverflow.com/questions/15998188/how-can-i-obtain-the-element-wise-logical-not-of-a-pandas-series
        """        
        print("\nDropping Unnecessary Columns")
        
        # Drop unknown columns (logical not)
        self.merged_df = self.merged_df.loc[:, ~self.merged_df.columns.str.contains('Unknown')]
        
        # Define columns to drop these are not useful for analysis
        cols_to_drop = [
            'Report Type', 'Report Month', 'Report Year', 'Report Submission Date',
            'Driver / Operator Type', 'Notice Received Date', 'Reporting Entity', 
            'Operating Entity', 'Serial Number', 'Data Availability',
            'Latitude', 'Longitude', 'Address', 'Zip Code', 
            'Investigating Agency', 'Rep Ent Or Mfr Investigating?', 'Investigating Officer Name',
            'Investigating Officer Phone', 'Investigating Officer Email',
            'Other Reporting Entities', 'Federal Regulatory Exemption',
            'Within ODD? - CBI', 'Within ODD?',
            'Same Incident ID', 'Same Vehicle ID',
            'Narrative', 'Narrative - CBI?',
            'VIN', 'Law Enforcement Investigating?', 'Source'
        ]
        
        # Drop columns that exist esure they are in the DataFrame
        cols_to_drop = [col for col in cols_to_drop if col in self.merged_df.columns]
    
        self.merged_df.drop(columns=cols_to_drop, inplace=True)
        
        print(f"Shape after dropping unnecessary columns: {self.merged_df.shape}")
    
    def count_and_sum_missing_values(self):
        """
        Purpose: Count and sum missing, empty, NaN, and 'unknown' values for each column in the DataFrame.
        Input: DataFrame
        Output: Prints a summary table with the counts and percentages of missing values.
        Reference:
        https://stackoverflow.com/questions/62879263/calculations-using-pandas-apply-lambda - for debugging the lamdba function
        https://www.w3schools.com/python/python_lambda.asp for using lambda functions to count NaN and 'unknown' values
        """
        print("\nMissing Values Summary")
        # Initialize a list to store the summary for each column
        overall_summary = []
        for col in self.merged_df.columns:
            # lambda function to count NaN and 'unknown' values
            nan_count = self.merged_df[col].apply(lambda x: pd.isna(x)).sum() # count NaN values
            unknown_count = self.merged_df[col].apply(lambda x: str(x).strip().lower() == 'unknown').sum() # count 'unknown' values
            total = nan_count + unknown_count
            percentage = (total / len(self.merged_df)) * 100 # calculate percentage of missing values
            overall_summary.append((col, nan_count, unknown_count, total, percentage))
        
        # Create a DataFrame for the summary and sort by the highest percentage
        summary_df = pd.DataFrame(overall_summary, columns=['Column', 'NaN', 'Unknown', 'Total', 'Percentage'])
        summary_df = summary_df.sort_values(by='Percentage', ascending=False)
        
        # these will automatically be removed if percentage higher than 50 in next function
        print("Top 10 columns with highest missing values:")
        print(tabulate(summary_df.head(10), headers='keys', tablefmt='grid'))
        return summary_df
    
    def clean_high_missing_columns(self):
        """
        Purpose: Remove columns with more than 50% missing values and clean unknown values
        Reference: Figuring out how to make all unknowns NaN
        - https://stackoverflow.com/questions/37060385/pandas-apply-lambda-function-null-values
        - https://stackoverflow.com/questions/73127383/replace-str-values-in-series-into-np-nan/73127521
        """
        print("\nRemoving columns where >50% of values are missing")
        # Get summary of missing values
        summary_df = self.count_and_sum_missing_values()

        # Drop columns with more than 50% missing values
        high_missing_cols = summary_df[summary_df['Percentage'] > 50]['Column'].tolist()
        if high_missing_cols:
            self.merged_df.drop(columns=high_missing_cols, inplace=True)

        # Replace any cell containing "unknown" (case-insensitive) with NaN
        self.merged_df = self.merged_df.applymap( # mapping to replace anything that contans unknown to nan
            lambda x: np.nan if isinstance(x, str) and 'unknown' in x.lower() else x
        )

        # Show new shape of the cleaned DataFrame
        print(f"Shape after cleaning: {self.merged_df.shape}")

    
    def clean_incident_date(self, val):
        """
        Purpose: Clean and parse incident date values converting to NaT to catch empty strings or invalid dates
        Reference: https://stackoverflow.com/questions/2052390/how-to-convert-string-to-datetime-in-python
        """
        try: # try to parse the date
            val = str(val).strip()
            if val.lower() == '':
                return pd.NaT # return NaT if empty
            return parser.parse(val, dayfirst=False, yearfirst=False)
        except:
            return pd.NaT
    
    def process_datetime_columns(self):
        """
        Purpose: Process and clean datetime-related columns
        Reference:
        -https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html to help make the datetime columns
        """
        print("\nProcessing Datetime Columns")

        # Clean incident date
        if 'Incident Date' in self.merged_df.columns:
            self.merged_df['Incident Date'] = self.merged_df['Incident Date'].apply(self.clean_incident_date)
            self.merged_df['Incident Date'] = pd.to_datetime(
                self.merged_df['Incident Date'], format='%b-%Y', errors='coerce'
            )
            # Drop rows with NaT incident dates as these are basically missing data
            initial_shape = self.merged_df.shape[0]
            self.merged_df = self.merged_df.dropna(subset=['Incident Date'])
            dropped = initial_shape - self.merged_df.shape[0] # calculate how many rows were dropped
        
        # Clean incident time hours-minutes format
        if 'Incident Time (24:00)' in self.merged_df.columns:
            self.merged_df['Incident Time (24:00)'] = pd.to_datetime(
                self.merged_df['Incident Time (24:00)'], format='%H:%M', errors='coerce'
            ).dt.time
    
    def process_numerical_columns(self):
        """
        Purpose: Conveting Numerical columns to numerical types
        """
        print("\nConverting Numerical Columns")
        # the numerical columns that are in the dataset
        numerical_columns = ['Report Version', 'Mileage', 'Posted Speed Limit (MPH)', 
                           'SV Precrash Speed (MPH)', 'Model Year']
        existing_numerical = [col for col in numerical_columns if col in self.merged_df.columns]
        if existing_numerical:
            self.merged_df[existing_numerical] = self.merged_df[existing_numerical].apply(
                pd.to_numeric, errors='coerce'
            )
    
    def add_country_column(self):
        """
        Purpose: Add country column for US data 
        """
        self.merged_df['Country'] = 'US'
        print("Added 'Country' column with value 'US'")
    
    def drop_us_specific_columns(self):
        """
        Purpose: Drop columns that are specific to US data format
        """
        print("\nDropping Columns not in STATS19 format")
        columns_to_drop_us_only = [
            'Roadway Description', 'Mileage', 'Property Damage?', 
            'SV Precrash Speed (MPH)', 'CP Contact Area', 'SV Any Air Bags Deployed?', 'SV Was Vehicle Towed?','CP Pre-Crash Movement'
        ]
        existing_cols = []
        for col in columns_to_drop_us_only:
            if col in self.merged_df.columns:
                existing_cols.append(col)
        if existing_cols:
            self.merged_df.drop(columns=existing_cols, inplace=True)
        print(f"Final shape: {self.merged_df.shape}") # reduce by 4
    
    def remap_contact_area(self, cell):
        """
        Purpose: Rename and standardize contact area values to match STATS19 format this is the mapping function
        Reference:
        - https://learn.microsoft.com/en-us/office/vba/language/reference/user-interface-help/split-function split all the values after the and replace them with the new values
        """
        if pd.isna(cell):
            return cell
        # Define the mapping for contact areas
        value_mapping = {
            'Rear Left': 'Back',
            'Rear': 'Back',
            'Rear Right': 'Back',
            'Left': 'Nearside',
            'Top': 'Top',
            'Front': 'Front',
            'Front Right': 'Front',
            'Front Left': 'Front',
            'Right': 'Offside',
            'Bottom': 'Bottom'
        }
        # Split the cell by commas and map values
        values = [v.strip() for v in cell.split(',')]
        mapped = [value_mapping.get(v, v) for v in values]
        return ', '.join(sorted(set(mapped)))  # Remove duplicates and sort
    
    def standardize_contact_areas(self):
        """
        Purpose: Rename and standardize contact area values to match STATS19 format
        """
        if 'SV Contact Area' in self.merged_df.columns:
            self.merged_df['SV Contact Area'] = self.merged_df['SV Contact Area'].apply(self.remap_contact_area)
            print("Standardized SV Contact Area values")
    
    def save_cleaned_data(self, output_path):
        """
        Purpose: Save the cleaned DataFrame to a CSV file
        Inputs: The output file path where the cleaned data will be saved
        Outputs: CSV file containing the cleaned dataset
        """
        self.merged_df.to_csv(output_path, index=False)
        print(f"Final dataset shape: {self.merged_df.shape}")
    
    def run_full_pipeline(self, output_path):
        """
        Description: Running all the functions in sequence to process the dataset
        Inputs: The dataset set folder path 
        Outputs: Output cleaned dataset file into the specified path
        """
        try:
            self.load_datasets()  # Step 1: Load data
            self.analyze_datasets() # Step 2: Analyze datasets
            self.merge_datasets() # Step 3: Merge datasets
            self.combine_related_columns() # Step 4: Combine related columns
            self.drop_unnecessary_columns() # Step 5: Drop unnecessary columns 
            self.clean_high_missing_columns() # Step 6: Clean high missing value columns
            self.process_datetime_columns() # Step 7: Process datetime columns
            self.process_numerical_columns()  # Step 8: Process numerical columns
            self.add_country_column()  # Step 9: Add country column
            self.drop_us_specific_columns()  # Step 10: Drop US-specific columns
            self.standardize_contact_areas() # Step 11: Standardize contact areas
            self.save_cleaned_data(output_path) # Step 12: Save cleaned data
            # Output the column names of the cleaned dataset
            print("\nFinal Column Names:")
            print(self.merged_df.columns.tolist())
        except Exception as e: # catch any errors 
            print(f"Cleaning Failed: {str(e)}")
            raise


def main():
    """
    Description: Main function to run the US data cleaning pipeline 
    - processor what sets up the data class 
    - functions to fun the all the methods in the class
    """
    Dataset_Folder_Path = "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/US"  # Where the Datasets are stores
    Output_Saving_Folder_Path = "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/US-cleaned_data.csv"  # Update this path
    # making and processing the dataset
    processor = USDataClass(Dataset_Folder_Path)
    # running all the cell code
    processor.run_full_pipeline(Output_Saving_Folder_Path) 

if __name__ == "__main__":
    main()