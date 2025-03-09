import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__, template_folder="../templates")

# Load model and preprocessing objects
model = pickle.load(open('ChurnModel.pkl', 'rb'))
encoder_dict = pickle.load(open('encoders.pkl', 'rb'))
scaler = pickle.load(open('scaler.pickle', 'rb'))

# Define expected feature order (MUST match training data order)
EXPECTED_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# Features that need scaling (must match scaler's training order)
SCALED_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

@app.route("/")
def home():
    return render_template("index.html")
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data received'}), 400

        # Validate all required fields
        missing_fields = [field for field in EXPECTED_FEATURES if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {missing_fields}'}), 400

        # Initialize data containers
        formatted_data = []
        scaled_values = []
        scaled_indices = []

        # Process each feature in PRECISE order
        for idx, feature in enumerate(EXPECTED_FEATURES):
            value = data[feature]
            
            try:
                if feature in SCALED_FEATURES:
                    # Validate numerical values
                    scaled_val = float(value)
                    scaled_values.append(scaled_val)
                    scaled_indices.append(idx)
                    formatted_data.append(0.0)  # Placeholder
                elif feature in encoder_dict:
                    # Validate categorical values
                    encoded_val = encoder_dict[feature].transform([str(value)])[0]
                    formatted_data.append(encoded_val)
                else:
                    # Validate other numericals (e.g., SeniorCitizen)
                    formatted_data.append(float(value))
            except ValueError as e:
                return jsonify({'error': f'Invalid value for {feature}: {str(e)}'}), 400

        # Apply scaling
        scaled_results = scaler.transform([scaled_values])[0]
        for i, idx in enumerate(scaled_indices):
            formatted_data[idx] = scaled_results[i]

        # Final validation
        if len(formatted_data) != len(EXPECTED_FEATURES):
            return jsonify({'error': 'Data length mismatch'}), 500

        prediction = model.predict([formatted_data])[0]
        return jsonify({'prediction': 'churn' if prediction == 1 else 'not churn'})

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(debug=True)
