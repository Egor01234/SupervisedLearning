
#Pre-processing - Retrieve & prepare the data: Load and explore the dataset referenced in section 4 in this document using techniques learnt during this course.
# Visualize the data and describe it thoroughly, identify correlations..etc.
# Clean, transform categorical data and model the dataset using the techniques learnt throughout the course in preparation for building a predictive model.
# Model building & fine tuningBuild a supervised predictive model based using a suitable classification algorithm(s) in python , utilizing scikit-learn, pandas, numpy…etc. To provide predictions as specified in project specification, section 3 of this report.
# Validate / score and evaluate the models and choose the best model after carrying out hyper-parameter tuning.
# Model deployment: Build an API for the model using Python Flask framework.
# Deploy the model on local host.
# Build a simple front end to access the API and pass new feature values to the prediction model for inference. 




import pandas as pd
import os
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_val_score
from numpy import asarray 
from sklearn.preprocessing import OneHotEncoder
path = "/Users/egor/Documents/GitHub/SupervisedLearning/"
filename = 'MOTORCYCLIST_KSI_-9032082310316605521.csv'

fullpath = os.path.join(path,filename)
dataFrame = pd.read_csv(fullpath,sep=',')

print("Types of data in Data Frame:")
print(dataFrame.dtypes)
print()
print("Missing values in Data Frame:")
print(dataFrame.isnull().sum())
print()
print("Numeric fields:")
print(dataFrame.describe())
print()

cols = ['OBJECTID', 'INDEX', 'ACCNUM', 'STREET2', 'OFFSET', 'LATITUDE', 'LONGITUDE', 'INVTYPE', 'INJURY', 'FATAL_NO', 'INITDIR', 'PEDACT', 'PEDCOND', 'CYCLISTYPE', 'CYCACT', 'CYCCOND', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH', 'PEDTYPE', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY', 'HOOD_140', 'NEIGHBOURHOOD_140', 'DIVISION', 'x', 'y'] 
target = 'ACCLASS'
dataFrame = dataFrame.drop(cols, axis=1)

# sns.countplot(x='ACCLASS', data=dataFrame)
# plt.title('Fatal vs. Non-Fatal Accidents')
# plt.show()

# sns.stripplot(x= 'VISIBILITY',y = 'ACCLASS', data = dataFrame, jitter = True)
# plt.show()

# plt.figure(figsize=(12, 8))
# sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# plt.title("Correlation Heatmap of Numerical Features")
# plt.show()


numeric_df = dataFrame.select_dtypes(include = int)
categorical_df = dataFrame.select_dtypes(exclude = int)

miss_cols = ['ACCLOC', 'VEHTYPE', 'MANOEUVER', 'DRIVACT', 'DRIVCOND', 'TRAFFCTL', 'VISIBILITY', 'IMPACTYPE', 'ROAD_CLASS', 'DISTRICT']

for col in miss_cols:
    dataFrame[col] = dataFrame[col].fillna("Unknown")

# category_cols = ['DATE','STREET1','ROAD_CLASS','LIGHT','RDSFCOND','ACCLASS','NEIGHBOURHOOD_158','ACCLOC', 'VEHTYPE', 'MANOEUVER', 'DRIVACT', 'DRIVCOND', 'TRAFFCTL', 'VISIBILITY', 'IMPACTYPE', 'DISTRICT']

category_cols = categorical_df.columns

transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(sparse_output=False), category_cols)], remainder='passthrough')

dataFrame_trans = transformer.fit_transform(dataFrame[category_cols])

encoded_columns = transformer.get_feature_names_out()

dataFrame_trans = pd.DataFrame(dataFrame_trans, columns=encoded_columns)

dataFrame = dataFrame.drop(category_cols, axis = 1)

dataFrame = dataFrame.join(dataFrame_trans)

print("Types of data in Data Frame:")
print(dataFrame.dtypes)
print()


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)