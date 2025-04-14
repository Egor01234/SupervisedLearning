# ============ Load Core Libraries for Data ============
import pandas as pd
import numpy as np
import os

# Create a 'Plots' directory if it doesn't exist
if not os.path.exists('Plots'):
    os.makedirs('Plots')

# Load data
path = os.path.dirname(os.path.abspath(__file__))
filename = 'MOTORCYCLIST_KSI_-9032082310316605521.csv'
fullpath = os.path.join(path, filename)
dataFrame = pd.read_csv(fullpath, sep=',')

# Explore data
print("Types of data in Data Frame:")
print(dataFrame.dtypes)
print()
print("Missing values in Data Frame:")
print(dataFrame.isnull().sum())
print()
print("Numeric fields:")
print(dataFrame.describe())
print()
print(dataFrame.columns)
print()

# ============ Prepare Target and Features ============
target = dataFrame['ACCLASS']
target = target.replace({'Fatal': 1, 'Non-Fatal Injury': 0})
dataFrame_features = dataFrame.drop('ACCLASS', axis=1)

# Drop unnecessary columns
cols = ['DIVISION', 'NEIGHBOURHOOD_140', 'NEIGHBOURHOOD_158', 'AG_DRIV', 'SPEEDING', 'MOTORCYCLE',
        'PEDCOND', 'PEDTYPE', 'INITDIR', 'IMPACTYPE', 'VISIBILITY', 'LATITUDE', 'DISTRICT', 'OBJECTID',
        'INDEX', 'STREET2', 'TRAFFCTL', 'LIGHT', 'INVAGE', 'FATAL_NO', 'CYCLISTYPE', 'CYCCOND', 'CYCLIST',
        'TRSN_CITY_VEH', 'DISABILITY', 'HOOD_158', 'HOOD_140', 'PASSENGER', 'TRUCK', 'VEHTYPE', 'INJURY',
        'INVTYPE', 'y', 'ACCNUM', 'DATE', 'x', 'ROAD_CLASS', 'EMERG_VEH', 'PEDESTRIAN', 'DRIVCOND',
        'DRIVACT', 'PEDACT', 'OFFSET', 'LONGITUDE', 'CYCACT', 'REDLIGHT', 'ALCOHOL']
dataFrame_features = dataFrame_features.drop(cols, axis=1)

# Identify numeric and categorical features
numeric_df = dataFrame_features.select_dtypes(include=int)
categorical_df = dataFrame_features.select_dtypes(exclude=int)