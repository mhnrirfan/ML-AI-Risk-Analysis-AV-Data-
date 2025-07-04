# Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import seaborn as sns
import numpy as np
import plotly.express as px
import itertools
from tabulate import tabulate
from dateutil import parser

# Reading in the Datasets
adas = pd.read_csv('/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/US/SGO-2021-01_Incident_Reports_ADAS.csv')
ads = pd.read_csv('/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/US/SGO-2021-01_Incident_Reports_ADS.csv')

def find_primary_key(df1, df2):
    """
    Purpose: Finding the primary keys in each dataframe 
    Methods:
        - Check each column in the dataframe to see if it is unique (count of column values = number of rows)
        - If not, use itertools to experiment with every combination to find a unique combination
    Input: Two DataFrames
    Output: tuple of primary key column names, or None if no key found
    """
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Check for single primary key
    for col in combined_df.columns:
        if combined_df[col].is_unique:
            return (col,)
    # Check combinations of columns
    for i in range(2, len(combined_df.columns) + 1):
        for combo in itertools.combinations(combined_df.columns, i):
            if combined_df.duplicated(subset=list(combo)).sum() == 0:
                return combo
    return None

def drop_unnecessary_columns(df):
    """
    Drop unnecessary columns from the dataframe.
    """
    # drop all the unknown columns as they can be empty and imputed later (22 columns)
    df = df.loc[:, ~df.columns.str.contains('Unknown')]
    print(tabulate(df.head(2), headers='keys', tablefmt='grid'))
    print(f"Number of columns after dropping 'Unknown': {df.shape[1]}")
    
    combine_and_drop(df, 'CP Contact Area', [
        'CP Contact Area - Rear Left', 'CP Contact Area - Left', 'CP Contact Area - Front Left',
        'CP Contact Area - Rear', 'CP Contact Area - Top', 'CP Contact Area - Front',
        'CP Contact Area - Rear Right', 'CP Contact Area - Right', 'CP Contact Area - Front Right',
        'CP Contact Area - Bottom' 
    ])

    combine_and_drop(df, 'ADAS/ADS System Version', [
        'ADAS/ADS System Version', 'ADAS/ADS System Version - Unk', 'ADAS/ADS System Version CBI'
    ])

    combine_and_drop(df, 'ADAS/ADS Hardware Version', [
        'ADAS/ADS Hardware Version', 'ADAS/ADS Hardware Version - Unk', 'ADAS/ADS Hardware Version CBI'
    ])

    combine_and_drop(df, 'ADAS/ADS Software Version', [
        'ADAS/ADS Software Version', 'ADAS/ADS Software Version - Unk', 'ADAS/ADS Software Version CBI'
    ])

    combine_and_drop(df, 'Other Reporting Entities', [
        'Other Reporting Entities?', 'Other Reporting Entities? - Unk', 'Other Reporting Entities? - NA'
    ])

    combine_and_drop(df, 'Federal Regulatory Exemption', [
        'Federal Regulatory Exemption?', 'Other Federal Reg. Exemption',
        'Federal Reg. Exemption - Unk', 'Federal Reg. Exemption - No'
    ])

    combine_and_drop(df, 'State or Local Permit', [
        'State or Local Permit?', 'State or Local Permit'
    ])

    combine_and_drop(df, 'Source', [
        'Source - Complaint/Claim', 'Source - Telematics', 'Source - Law Enforcement',
        'Source - Field Report', 'Source - Testing', 'Source - Media',
        'Source - Other', 'Source - Other Text'
    ])

    combine_and_drop(df, 'Weather', [
        'Weather - Clear', 'Weather - Snow', 'Weather - Cloudy', 'Weather - Fog/Smoke',
        'Weather - Rain', 'Weather - Severe Wind', 'Weather - Other', 'Weather - Other Text'
    ])

    combine_and_drop(df, 'SV Contact Area', [
        'SV Contact Area - Rear Left', 'SV Contact Area - Left', 'SV Contact Area - Front Left',
        'SV Contact Area - Rear', 'SV Contact Area - Top', 'SV Contact Area - Front',
        'SV Contact Area - Rear Right', 'SV Contact Area - Right', 'SV Contact Area - Front Right',
        'SV Contact Area - Bottom'
    ])

    combine_and_drop(df, 'Data Availability', [
        'Data Availability - EDR', 'Data Availability - Police Rpt', 'Data Availability - Telematics',
        'Data Availability - Complaints', 'Data Availability - Video', 'Data Availability - Other',
        'Data Availability - No Data'
    ])

    # Apply remapping to the 'SV Contact Area' column
    df['SV Contact Area'] = df['SV Contact Area'].apply(remap_contact_area)

    cols_to_drop = [
        'Report Type', 'Report Month', 'Report Year', 'Report Submission Date','Driver / Operator Type',
        'Notice Received Date','Reporting Entity', 'Operating Entity', 
        'Serial Number','Data Availability',
        'Latitude', 'Longitude', 'Address', 'Zip Code', 
        'Investigating Agency', 'Rep Ent Or Mfr Investigating?', 'Investigating Officer Name',
        'Investigating Officer Phone', 'Investigating Officer Email',
        'Other Reporting Entities', 'Federal Regulatory Exemption',
        'Within ODD? - CBI','Within ODD?',
        'Same Incident ID', 'Same Vehicle ID',
        'Narrative', 'Narrative - CBI?',
        'VIN', 
        'Law Enforcement Investigating?',
        'Source',
        'CP Pre-Crash Movement', 'CP Any Air Bags Deployed?', 'CP Was Vehicle Towed?', 'SV Any Air Bags Deployed?',
        'SV Was Vehicle Towed?', 'SV Were All Passengers Belted?', 'Roadway Description', 
        'Mileage', 
        'Property Damage?', 
        'SV Precrash Speed (MPH)', 
        'CP Contact Area'
    ]
    
    # Drop the unwanted columns if they exist
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    
    return df

def remap_contact_area(cell):
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
    if pd.isna(cell):
        return cell
    values = [v.strip() for v in str(cell).split(',')]
    mapped = [value_mapping.get(v, v) for v in values]
    return ', '.join(sorted(set(mapped)))

def combine_and_drop(merged_df, new_col_name, cols_to_combine):
    '''
    Purpose: Combine column values into 1, if they contain Y then place the column name 
    Input:
        merged_df: DataFrame to modify
        new_col_name: Name of the new column to create
        cols_to_combine: List of columns to combine
    Output: DataFrame with the new column and specified columns dropped
    '''
    def combine_values(row):
        combined_values = []
        for col in cols_to_combine:  # go through all columns in list
            if str(row[col]).strip().upper() == 'Y':  # if value is 'Y' then
                combined_values.append(col.split(' - ')[-1])  # only place what is after the column name
        return ', '.join(combined_values)

    merged_df[new_col_name] = merged_df[cols_to_combine].apply(combine_values, axis=1)
    merged_df.drop(columns=cols_to_combine, inplace=True)

def clean_incident_date(val):
    try:
        val = str(val).strip()

        if val.lower() == 'missing' or val == '':
            return pd.NaT

        # Try parsing common formats
        return parser.parse(val, dayfirst=False, yearfirst=False)
    except:
        return pd.NaT

def fix_formats(df):
    """
    Fix formats of specific columns in the dataframe.
    """
    df['Incident Date'] = df['Incident Date'].apply(clean_incident_date)
    df['Incident Date'] = pd.to_datetime(
        df['Incident Date'], format='%b-%Y', errors='coerce'
    )

    # First: Clean string representation and convert float-like years
    df['Model Year'] = df['Model Year'].apply(
        lambda x: str(int(float(x))) if pd.notnull(x) and str(x).replace('.', '', 1).isdigit() else x
    )

    # Then convert to datetime (expects 4-digit year like '2021')
    df['Model Year'] = pd.to_datetime(
        df['Model Year'], format='%Y', errors='coerce'
    )

    # Convert time strings to time objects
    df['Incident Time (24:00)'] = pd.to_datetime(
        df['Incident Time (24:00)'], format='%H:%M', errors='coerce'
    ).dt.time

def US_Data_Cleaning(adas, ads):
    # Add a source column
    adas['Source'] = 'ADAS'
    ads['Source'] = 'ADS'

    # Find the primary key for merging
    keys = find_primary_key(adas, ads)
    print(f"Primary key for merging: {keys}")

    # Merge the two datasets
    merged_df = pd.concat([adas, ads], ignore_index=True)

    # Drop duplicates based on the keys
    merged_df = merged_df.drop_duplicates(subset=keys, keep='first')

    # Keep only the rows with max Report Version for each Report ID
    merged_df = merged_df.loc[merged_df.groupby('Report ID')['Report Version'].idxmax()].reset_index(drop=True)

    # Drop unnecessary columns
    merged_df = drop_unnecessary_columns(merged_df)
    
    # Add the 'Country' column with all values as 'US'
    merged_df['Country'] = 'US' 

    # Fill NaNs and empty strings with 'Missing'
    merged_df.fillna('Missing', inplace=True)
    merged_df.replace('', 'Missing', inplace=True)

    # Fix formats INPLACE, no assignment
    fix_formats(merged_df)

    # Save cleaned data (update the path as needed)
    merged_df.to_csv('/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/US/cleaned_data.csv', index=False)

    return merged_df

# Run the cleaning function on UK data
df = US_Data_Cleaning(adas,ads)
print(f"Shape of the cleaned data: {df.shape}")
print("US data cleaning completed successfully.")
