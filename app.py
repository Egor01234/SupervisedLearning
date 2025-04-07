from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive form data
        data = {
            'ACCLOC': request.form['ACCLOC'],
            'MANOEUVER': request.form['MANOEUVER'],
            'AUTOMOBILE': int(request.form['AUTOMOBILE']),
            'STREET1': request.form['STREET1'],
            'TIME': request.form['TIME'],
            'RDSFCOND': request.form['RDSFCOND']
        }

        # Create DataFrame and predict
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)
        label = 'Fatal' if prediction[0] == 1 else 'Non-Fatal Injury'
        return render_template('index.html', prediction=label)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
