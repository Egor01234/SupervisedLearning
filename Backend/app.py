import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import traceback

MODEL_DIR = 'model_pkl'

MODEL_FILES = {
    'DecisionTree': 'DecisionTree_model.pkl',
    'RandomForest': 'RandomForest_model.pkl',
    'GradientBoosting': 'GradientBoosting_model.pkl',
    'LogisticRegression': 'LogisticRegression_model.pkl',
    'SVM': 'SVM_model.pkl'
}

EXPECTED_FEATURES = ['ACCLOC', 'MANOEUVER', 'AUTOMOBILE', 'STREET1', 'TIME', 'RDSFCOND']

app = Flask(__name__)

cors = CORS(app, resources={
    r"/predict/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"]
    },
     r"/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"]
    }
})


models = {}
print("Loading models...")

for model_key, file_name in MODEL_FILES.items():
    file_path = os.path.join(MODEL_DIR, file_name)
    print(f"Attempting to load: {file_path}")
    try:
        loaded_model = joblib.load(file_path)
        models[model_key] = loaded_model
        print(f"[SUCCESS] Loaded '{model_key}' from {file_path}")
    except FileNotFoundError:
        print(f"[ERROR] Model file not found: {file_path}. Skipping this model.")
        models[model_key] = None
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_key} from {file_path}: {e}")
        traceback.print_exc()
        models[model_key] = None

print("Model loading complete.")
print("-" * 30)
