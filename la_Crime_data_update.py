from datetime import datetime
import re
import pandas as pd
from sodapy import Socrata
import numpy as np
import requests
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import math
import statsmodels.api as sm
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, \
    classification_report, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor   # For regression tasks
from imblearn.over_sampling import SMOTE
from scipy.stats import chi2_contingency
import time
import datetime as dt
from PyQt5.QtWidgets import QFileDialog, QApplication
import sys
from itertools import product

########## first select the lacrimedatafifteencat file because it is referenced in the code ########
# Open the file dialog to select a file
file_path, _ = QFileDialog.getOpenFileName()

# Use the selected file path to read the file into a DataFrame
if file_path:  # Making sure a file path was selected, select the 'lacrimedatafifteencat.xlsx' file from wherever you saved it on your desktop
    crime_cat_map = pd.read_excel(file_path)


###############################################################################
################### pull data from live API source in LAPD ####################
start_time = time.time()
# Unauthenticated client only works with public data sets. Note 'None'
# in place of application token, and no username or password:
client = Socrata("data.lacity.org", None)
# Example authenticated client (needed for non-public datasets):
# client = Socrata(data.lacity.org,
#                  MyAppToken,
#                  username="user@example.com",
#                  password="AFakePassword")

# First 2000 results, returned as JSON from API / converted to Python list of
# dictionaries by sodapy.
results = client.get("2nrs-mtv8", limit = 2000000)
# Convert to pandas DataFrame
results_df = pd.DataFrame.from_records(results)
end_time = time.time()
time_taken = end_time - start_time
print(f"Live Data Pull From LAPD took: {time_taken} seconds")
###############################################################################
################### clean and stadardize the dataset for saving ###############
###############################################################################

start_time2 = time.time()
 

crime_cat_map = crime_cat_map.set_index('Crime Code')['Category'].to_dict()

results_df['crm_cd_1'] = results_df['crm_cd_1'].fillna(0)
results_df['crm_cd_1'] = results_df['crm_cd_1'].astype(int)

results_df['crime_category'] = results_df['crm_cd_1'].map(crime_cat_map).fillna('Unknown/Unreported')



results_df['location'] = results_df['location'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
results_df['vict_sex'] = results_df['vict_sex'].str.replace('M','Male')
results_df['vict_sex'] = results_df['vict_sex'].str.replace('F','Female')
results_df['vict_sex'] = results_df['vict_sex'].str.replace('X','Unknown')
results_df['vict_sex']


def replace_descent(code):
    if code == 'A':
        return 'Other Asian'
    elif code == 'B':
        return 'Black'
    elif code == 'C':
        return 'Chinese'
    elif code == 'D':
        return 'Cambodian'
    elif code == 'F':
        return 'Filipino'
    elif code == 'G':
        return 'Guamanian'
    elif code == 'H':
        return 'Hispanic/Latin/Mexican'
    elif code == 'I':
        return 'American Indian/Alaskan Native'
    elif code == 'J':
        return 'Japanese'
    elif code == 'K':
        return 'Korean'
    elif code == 'L':
        return 'Laotian'
    elif code == 'O':
        return 'Other'
    elif code == 'P':
        return 'Pacific Islander'
    elif code == 'S':
        return 'Samoan'
    elif code == 'U':
        return 'Hawaiian'
    elif code == 'V':
        return 'Vietnamese'
    elif code == 'W':
        return 'White'
    elif code == 'X':
        return 'Unknown'
    elif code == 'Z':
        return 'Asian Indian'
    else:
        return 'Unknown / Not Recorded'  # For any code not specifically listed

# Assuming results_df is your DataFrame
# Apply the function to each value in the 'vict_descent' column
results_df['vict_descent'] = results_df['vict_descent'].apply(replace_descent)









results_df.rename(columns={'date_rptd': 'date_reported', 'date_occ':'date_occurred',\
                           'time_occ':'time_occurred'},inplace=True)

results_df['date_reported'] = pd.to_datetime(results_df['date_reported'].str.replace('0:00','').str.strip())
## results_df['date_reported'] = pd.to_datetime(results_df['date_reported'])

results_df['date_occurred'] = pd.to_datetime(results_df['date_occurred'].str.replace('0:00','').str.strip())
## results_df['date_occurred'] = pd.to_datetime(results_df['date_occurred'])

results_df = results_df.replace('\*', '', regex=True)

def categorize_weapon(code):
    if pd.isna(code):
        return 'Unknown'
    else:
        prefix = str(code)[0]
        return {
            '1': 'Firearm',
            '2': 'Edged and Pointed Weapons',
            '3': 'Blunt Instruments',
            '4': 'Hand-to-Hand Combat/Physical Force',
            '5': 'Threats and Non-Physical Force'
        }.get(prefix, 'Other')
# Apply the categorization function to the 'Weapon Used Cd' column
results_df['level_one_weapon_description'] = results_df['weapon_used_cd'].apply(categorize_weapon)


results_df['Premis_Cd_Temp'] = results_df.apply(\
  lambda row: 0 \
  if pd.isna(row['premis_desc']) or row['premis_desc'].strip() == ''\
  else row['premis_cd'], axis=1)

    
def categorize_premises(code):
    if pd.isna(code):
        return 'Unknown'
    else:
        prefix = str(code)[0]
        return {
            '1': 'Public Spaces and Transit',
            '2': 'Commercial and Business Locations',
            '3': 'Specialized Facilities',
            '4': 'Retail and Consumer Services',
            '5': 'Residential Properties',
            '6': 'Financial Institutions',
            '7': 'Institutions and Cultural Locations',
            '8': 'Mass Transit Locations',
            '9': 'MTA Specific Locations',
            '0': 'Unknown / Not Recorded'
        }.get(prefix, 'Other')
# Apply the categorization function to the 'Premise Cd' column
results_df['level_one_premise_description'] = results_df['Premis_Cd_Temp'].apply(categorize_premises)

results_df.drop(columns='Premis_Cd_Temp',inplace = True)




def convert_int_to_12hour_time(time_int):
    # Convert to string and pad with zeros to ensure it's 4 characters long
    time_str = str(time_int).zfill(4)

    # Parse the string into a datetime object
    time_obj = datetime.strptime(time_str, '%H%M')

    # Format the time object into a 12-hour format string with AM/PM
    time_12hr_format = time_obj.strftime('%I:%M %p')

    # Return the time in 12-hour format
    return time_12hr_format

# Apply the function to the entire 'TIME_OCC' column
results_df['time_occurred_12_hour'] = results_df['time_occurred'].apply(convert_int_to_12hour_time)


def convert_int_to_time(time_int):
    # Convert to string and pad with zeros to ensure it's 4 characters long
    time_str = str(time_int).zfill(4)

    # Parse the string into a datetime object
    time_obj = datetime.strptime(time_str, '%H%M')

    # Return the time object
    return time_obj.time()

# Apply the function to the entire 'TIME_OCC' column
results_df['time_occurred_24_hour'] = results_df['time_occurred'].apply(convert_int_to_time)

end_time2 = time.time()

time_taken2 = end_time - start_time

results_df['date_time_occurred'] = pd.to_datetime(results_df['date_occurred'].astype(str) + ' ' + results_df['time_occurred_12_hour'].astype(str))

results_df = results_df[results_df['vict_age'].astype(int)>=0]
results_df = results_df[results_df['vict_age'].astype(int)<=100]

print(f"It took {time_taken2/60} minutes to run the code")
print(f"It took {(time_taken + time_taken2)/60} minutes to run the ful script")

results_df.info()

###############################################################################
################## save standardized file to excel // csv #####################
###############################################################################
folder_path = '' #type your personal folder path here if you want to save the file on your computer
file_name = 'Crime_Data_from_2020_to_present.xlsx'  # You can change this to your preferred file name
full_path = f"{folder_path}\\{file_name}"

results_df.to_excel(full_path)

folder_path = '' #type your personal folder path here if you want to save the file on your computer
file_name = 'Crime_Data_from_2020_to_present.csv'  # You can change this to your preferred file name
full_path = f"{folder_path}\\{file_name}"

results_df.to_csv(full_path)
