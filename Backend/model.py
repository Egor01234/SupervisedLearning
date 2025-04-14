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

# ============ Visualization ============
import matplotlib.pyplot as plt
import seaborn as sns

# Plot Fatal vs. Non-Fatal Accidents
plt.figure(figsize=(8, 5))
sns.countplot(x='ACCLASS', data=dataFrame)
plt.title('Fatal vs. Non-Fatal Accidents')
plt.savefig('Plots/fatal_vs_non_fatal_accidents.png')  
plt.show()

# Print class distribution
class_counts = target.value_counts()
print("Class distribution:")
print(class_counts)
print(f"Proportion of Fatal cases: {class_counts[1]/len(target):.2%}")

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.savefig('Plots/correlation_heatmap.png')  
plt.show()

# ============ Train-Test Split ============
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    dataFrame_features, target, test_size=0.2, random_state=42, stratify=target
)

# ============ Preprocessing Pipelines ============
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numeric_df.columns),
    ('cat', categorical_transformer, categorical_df.columns)
])

# ============ Modeling & Evaluation ============
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

models = {
    'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'DecisionTree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'NaiveBayes': GaussianNB(),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

results = {}
