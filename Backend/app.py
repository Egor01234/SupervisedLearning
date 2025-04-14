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

def make_prediction(model_name, data):
    """Handles input validation and prediction for a given model."""
    model_pipeline = models.get(model_name)

    if model_pipeline is None:
        print(f"[ERROR] Prediction requested for model '{model_name}', but it was not loaded.")
        return jsonify({"error": f"Model '{model_name}' is not available."}), 500

    if not all(field in data for field in EXPECTED_FEATURES):
        missing = [field for field in EXPECTED_FEATURES if field not in data]
        print(f"[ERROR] Missing fields in request for {model_name}: {missing}")
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        input_data = {}
        input_data['ACCLOC'] = data['ACCLOC']
        input_data['MANOEUVER'] = data['MANOEUVER']
        input_data['STREET1'] = data['STREET1']
        input_data['RDSFCOND'] = data['RDSFCOND']

        try:
            automobile_val = int(data['AUTOMOBILE'])
            if automobile_val < 0:
                 print(f"[ERROR] Invalid AUTOMOBILE value for {model_name}: {data['AUTOMOBILE']}")
                 return jsonify({"error": "AUTOMOBILE count cannot be negative."}), 400
            input_data['AUTOMOBILE'] = automobile_val
        except (ValueError, TypeError):
            print(f"[ERROR] Invalid AUTOMOBILE type for {model_name}: {data['AUTOMOBILE']}")
            return jsonify({"error": "Invalid input for AUTOMOBILE. Must be a whole number."}), 400

        try:
            time_val = int(data['TIME'])
            input_data['TIME'] = time_val
        except (ValueError, TypeError):
             print(f"[ERROR] Invalid TIME type for {model_name}: {data['TIME']}")
             return jsonify({"error": "Invalid input for TIME. Must be a number (e.g., 830, 1700)."}), 400

        input_df = pd.DataFrame([input_data], columns=EXPECTED_FEATURES)
        print(f"\n--- Prediction Request ({model_name}) ---")
        print("Input DataFrame for model:")
        print(input_df.to_string())

        prediction = model_pipeline.predict(input_df)
        probabilities = model_pipeline.predict_proba(input_df)

        prediction_code = int(prediction[0])
        prob_non_fatal = float(probabilities[0][0])
        prob_fatal = float(probabilities[0][1])

        result_label = "Fatal" if prediction_code == 1 else "Non-Fatal Injury"

        print(f"Prediction: {result_label} (Code: {prediction_code})")
        print(f"Probabilities: Non-Fatal={prob_non_fatal:.4f}, Fatal={prob_fatal:.4f}")
        print("-" * 30)

        return jsonify({
            "model_used": model_name,
            "prediction": result_label,
            "prediction_code": prediction_code,
            "probability_non_fatal": prob_non_fatal,
            "probability_fatal": prob_fatal
        }), 200

    except ValueError as ve:
         print(f"[ERROR] Value error during prediction preparation for {model_name}: {ve}")
         traceback.print_exc()
         return jsonify({"error": f"Prediction failed due to data issue: {ve}"}), 400
    except Exception as e:
        print(f"[ERROR] Prediction failed unexpectedly for model {model_name}: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An internal error occurred during prediction with {model_name}."}), 500


@app.route('/')
def home():
    loaded_models = [name for name, model in models.items() if model is not None]
    return jsonify({
        "message": "Motorcyclist Accident Severity Prediction API",
        "status": "Service is running",
        "available_models": loaded_models,
        "endpoints": [f"/predict/{name.lower()}" for name in loaded_models]
        })


@app.route('/predict/decisiontree', methods=['POST'])
def predict_decision_tree():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    return make_prediction('DecisionTree', data)

@app.route('/predict/randomforest', methods=['POST'])
def predict_random_forest():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    return make_prediction('RandomForest', data)

@app.route('/predict/gradientboosting', methods=['POST'])
def predict_gradient_boosting():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    return make_prediction('GradientBoosting', data)

@app.route('/predict/logisticregression', methods=['POST'])
def predict_logistic_regression():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    return make_prediction('LogisticRegression', data)

@app.route('/predict/svm', methods=['POST'])
def predict_svm():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    return make_prediction('SVM', data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)