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

# Train and evaluate models
for model_name, model_instance in models.items():
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model_instance)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    results[model_name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    }

    print(f"\nModel: {model_name}")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'Plots/confusion_matrix_{model_name}.png') 
    plt.show()
    
    # ============ GeoDataFrame Plot ============
import geopandas as gpd
import contextily as ctx

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(dataFrame,
                       geometry=gpd.points_from_xy(dataFrame.LONGITUDE, dataFrame.LATITUDE),
                       crs="EPSG:4326")

# Convert to Web Mercator for contextily
gdf = gdf.to_crs(epsg=3857)

# Plot Accident Locations
fig, ax = plt.subplots(figsize=(12, 8))
gdf.plot(ax=ax, alpha=0.5, marker='o', color='red', markersize=10)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
plt.title('Accident Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('Plots/accident_locations.png') 
plt.show()

# ============ DBSCAN Clustering ============
from sklearn.cluster import DBSCAN
import numpy as np

# Extract coordinates
coords = dataFrame[['LATITUDE', 'LONGITUDE']].to_numpy()

# Apply DBSCAN
db = DBSCAN(eps=0.01, min_samples=10, metric='haversine').fit(np.radians(coords))

# Add cluster labels to the DataFrame
dataFrame['cluster'] = db.labels_

# Visualize Clusters
plt.figure(figsize=(12, 8))
plt.scatter(dataFrame['LONGITUDE'], dataFrame['LATITUDE'], c=dataFrame['cluster'], cmap='viridis', s=10)
plt.title('Accident Clusters')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('Plots/accident_clusters.png')  
plt.show()

# ============ Accidents by Hour ============
dataFrame['DATE'] = pd.to_datetime(dataFrame['DATE'])
dataFrame['Hour'] = dataFrame['DATE'].dt.hour
hourly_accidents = dataFrame.groupby('Hour').size()
hourly_accidents.plot(kind='bar', figsize=(12, 6), title='Accidents by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Accidents')
plt.savefig('Plots/accidents_by_hour.png')
plt.show()

# ============ Top 5 Model ============
import joblib

model_save_directory = 'model_pkl'

if not os.path.exists(model_save_directory):
    os.makedirs(model_save_directory)
    print(f"Created directory: {model_save_directory}")

top_models = ['DecisionTree', 'RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM']

for model_name in top_models:
    print(f"\nTraining and saving {model_name} model...")

    final_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', models[model_name])
    ])

    final_pipeline.fit(X_train, y_train)

    file_name = f'{model_name}_model.pkl'
    save_path = os.path.join(model_save_directory, file_name)

    joblib.dump(final_pipeline, save_path)
    print(f"{model_name} model saved as {save_path}")

print("\nFinished saving top models.")