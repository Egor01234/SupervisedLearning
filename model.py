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
print(dataFrame.columns)  # This will show all the column names in your dataframe.
print()

target = dataFrame['ACCLASS']
target = target.replace({'Fatal': 1, 'Non-Fatal Injury': 0})
dataFrame_features = dataFrame.drop('ACCLASS', axis = 1)

from imblearn.pipeline import Pipeline as ImbPipeline  # instead of sklearn Pipeline
from imblearn.over_sampling import SMOTE

# Build complete pipeline with SMOTE


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


cols = ['OBJECTID', 'INDEX', 'ACCNUM', 'STREET2', 'OFFSET', 'LATITUDE', 'LONGITUDE', 'INVTYPE', 'INJURY', 'FATAL_NO', 'INITDIR', 'PEDACT', 'PEDCOND', 'CYCLISTYPE', 'CYCACT', 'CYCCOND', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH', 'PEDTYPE', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY', 'HOOD_140', 'NEIGHBOURHOOD_140', 'DIVISION', 'x', 'y'] 
dataFrame = dataFrame.drop(cols, axis=1)

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

# Build complete pipeline
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),  # Preprocessing step
#     ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))  # Classifier
# ])

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),            # Preprocessing step
    ('smote', SMOTE(random_state=42)),        # Oversampling on training set
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))  # Classifier
])

# Train the model
pipeline.fit(X_train, y_train)




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