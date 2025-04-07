from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from flask_cors import CORS 

app = Flask(__name__)
CORS(app) 

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: model.pkl not found. Make sure the model file is in the same directory.")
    model = None 
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def index():
  
    return "Flask API is running. Access the prediction endpoint at /predict."



@app.route('/predict', methods=['POST'])
def predict():
    if not model:
         return jsonify({'error': 'Model not loaded'}), 500
    try:
       
        data_in = request.json # Use this if sending JSON from React

        if not data_in:
             return jsonify({'error': 'No input data received'}), 400

        data = {
            'ACCLOC': data_in.get('ACCLOC'),
            'MANOEUVER': data_in.get('MANOEUVER'),
            'AUTOMOBILE': int(data_in.get('AUTOMOBILE', 0)), 
            'STREET1': data_in.get('STREET1'),
            'TIME': data_in.get('TIME'), 
            'RDSFCOND': data_in.get('RDSFCOND')
        }

        input_df = pd.DataFrame([data])

        prediction_proba = model.predict_proba(input_df) 
        prediction = model.predict(input_df) 

        label = 'Fatal' if prediction[0] == 1 else 'Non-Fatal Injury'
        probability_fatal = prediction_proba[0][1] 
        probability_non_fatal = prediction_proba[0][0] 

        # Return JSON response
        return jsonify({
            'prediction': label,
            'prediction_code': int(prediction[0]), # Send the raw code too
            'probability_fatal': float(probability_fatal),
            'probability_non_fatal': float(probability_non_fatal)
        })

    except KeyError as e:
         return jsonify({'error': f'Missing field: {e}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid value type: {e}'}), 400
    except Exception as e:
        # Log the detailed error on the server for debugging
        app.logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500 # Return generic error to client


if __name__ == '__main__':
    # Make sure debug=False in production!
    # Use host='0.0.0.0' to make it accessible on your network
    app.run(host='0.0.0.0', port=5000, debug=True)