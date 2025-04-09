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
import pickle
from PIL.features import features
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from numpy import asarray 
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE

path = os.path.dirname(os.path.abspath(__file__))
fullpath = r"MOTORCYCLIST_KSI_-9032082310316605521.csv"
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
print(dataFrame.columns) 
print()

target = dataFrame['ACCLASS']
target = target.replace({'Fatal': 1, 'Non-Fatal Injury': 0})
dataFrame_features = dataFrame.drop('ACCLASS', axis = 1)

from imblearn.pipeline import Pipeline as ImbPipeline  # instead of sklearn Pipeline
from imblearn.over_sampling import SMOTE

numeric_df = dataFrame_features.select_dtypes(include = int)
categorical_df = dataFrame_features.select_dtypes(exclude = int)

# for col in numeric_df.columns:
#     plt.figure(figsize=(10, 4))
#     sns.boxplot(x=dataFrame_features[col], orient='h')
#     plt.title(f'Boxplot of {col} (Outlier Detection)')
#     plt.xlabel(col)
#     plt.grid(True, axis='x', linestyle='--', alpha=0.5)
#     plt.show()

# for col in numeric_df.columns:
#     plt.figure(figsize=(8, 4))
#     sns.violinplot(x=dataFrame_features[col], orient='h')
#     plt.title(f'Violinplot of {col} (Smooth density distribution)')
#     plt.ylabel(col)
#     plt.grid(True, axis='x', linestyle='--', alpha=0.5)
#     plt.show()

# for col in categorical_df.columns:
#     plt.figure(figsize=(12, 4))
#     sns.countplot(x=col, data=dataFrame_features, order=dataFrame_features[col].value_counts().index)
#     plt.title(f'Count of categories in {col}')
#     plt.xticks(rotation=45)
#     plt.show()


# cols = ['OBJECTID', 'INDEX', 'ACCNUM', 'STREET2', 'OFFSET', 'LATITUDE', 'LONGITUDE', 'INVTYPE', 'INJURY', 'FATAL_NO', 'INITDIR', 'PEDACT', 'PEDCOND', 'CYCLISTYPE', 'CYCACT', 'CYCCOND', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH', 'PEDTYPE', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY', 'HOOD_140', 'NEIGHBOURHOOD_140', 'DIVISION', 'x', 'y'] 
# dataFrame = dataFrame.drop(cols, axis=1)

cols = ['DIVISION', 'NEIGHBOURHOOD_140', 'NEIGHBOURHOOD_158', 'AG_DRIV', 'SPEEDING', 'MOTORCYCLE', 'PEDCOND', 'PEDTYPE', 'INITDIR', 'IMPACTYPE', 'VISIBILITY', 'LATITUDE', 'DISTRICT', 'OBJECTID', 'INDEX', 'STREET2', 'TRAFFCTL', 'LIGHT', 'INVAGE', 'FATAL_NO', 'CYCLISTYPE', 'CYCCOND', 'CYCLIST', 'TRSN_CITY_VEH', 'DISABILITY', 'HOOD_158', 'HOOD_140', 'PASSENGER', 'TRUCK', 'VEHTYPE', 'INJURY', 'INVTYPE', 'y', 'ACCNUM', 'DATE', 'x', 'ROAD_CLASS', 'EMERG_VEH', 'PEDESTRIAN', 'DRIVCOND', 'DRIVACT', 'PEDACT', 'OFFSET', 'LONGITUDE', 'CYCACT', 'REDLIGHT', 'ALCOHOL'] 
dataFrame_features = dataFrame_features.drop(cols, axis=1)

numeric_df = dataFrame_features.select_dtypes(include = int)
categorical_df = dataFrame_features.select_dtypes(exclude = int)


sns.countplot(x='ACCLASS', data=dataFrame)
plt.title('Fatal vs. Non-Fatal Accidents')
plt.show()

sns.stripplot(x= 'VISIBILITY',y = 'ACCLASS', data = dataFrame, jitter = True)
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()


# Preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing numerical values
    ('scaler', StandardScaler())  # Normalize numerical features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing categorical values
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Encode categorical features
])

preprocessor = ColumnTransformer(transformers=[

    ('num', numerical_transformer, numeric_df.columns),
    ('cat', categorical_transformer, categorical_df.columns)
])

# Build complete pipeline
pipeline = make_pipeline(
    preprocessor,
    RandomForestClassifier(random_state=42, class_weight='balanced')
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    dataFrame_features, target, test_size=0.2, random_state=42, stratify=target
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

models = {
    'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'DecisionTree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'NaiveBayes': GaussianNB()
}

results = {}

for model_name, model_instance in models.items():
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model_instance)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Store scores in a dictionary
    results[model_name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    }

# Display the results
for model, scores in results.items():
    print(f"\nModel: {model}")
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")

modelResults_df = pd.DataFrame(results).T  # Transpose for better format
print(f"\nModel Dictionary: {modelResults_df}\n")

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),            
    ('smote', SMOTE(random_state=42)),     
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))  
])

# Train the model
pipeline.fit(X_train, y_train)


feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

importances = pipeline.named_steps['classifier'].feature_importances_

feat_imp = pd.Series(importances, index=feature_names)

feat_imp = feat_imp.sort_values(ascending=False).head(10)

feat_imp.plot(kind='barh', figsize=(10, 8), title="Top 10 Most Important Features")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()


# Evaluate the model
y_pred = pipeline.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Cross-validation
cv_scores = cross_val_score(pipeline, dataFrame_features, target, cv=5, scoring='f1_weighted')
print(f"Cross-validation F1 scores: {cv_scores}")
print(f"Average F1 score: {np.mean(cv_scores):.2f}")

with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

#Geohashing
dataFrame['lat_bin'] = pd.cut(dataFrame['LATITUDE'], bins=50)
dataFrame['lon_bin'] = pd.cut(dataFrame['LONGITUDE'], bins=50)
hot_zones = dataFrame.groupby(['lat_bin', 'lon_bin'], observed=True).size().reset_index(name='counts')
top_zones = hot_zones.sort_values(by='counts', ascending=False).head(10)
print(top_zones)
print()
zone_counts = dataFrame['NEIGHBOURHOOD_140'].value_counts().reset_index()
zone_counts.columns = ['Neighbourhood', 'Accident Count']

# Show top 10 accident-prone areas
print(zone_counts.head(10))

import matplotlib.pyplot as plt

# Top lat/lon zones
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.barh(top_zones.index.astype(str), top_zones['counts'], color='skyblue')
plt.title('Top Grid Hot Zones')
plt.xlabel('Accident Count')
plt.ylabel('Lat/Lon Bin Index')

# Top neighborhoods
plt.subplot(1, 2, 2)
plt.barh(zone_counts['Neighbourhood'].head(10), zone_counts['Accident Count'].head(10), color='orange')
plt.title('Top Neighbourhoods')
plt.xlabel('Accident Count')
plt.tight_layout()
plt.show()

