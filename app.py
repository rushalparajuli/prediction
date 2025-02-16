import numpy as np
from flask import Flask, request, jsonify, render_template, session
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for sessions

# Load model and encoders
try:
    model = pickle.load(open('ChurnModel.pkl', 'rb'))
    encoder_dict = pickle.load(open('encoders.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model or encoders: {e}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result")
def result():
    return render_template("result.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Remove customerID as it's not used in prediction
        data.pop('customerID', None)
        
        formatted_data = []
        
        # Define expected feature order based on model training
        expected_features = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ]
        
        # Convert SeniorCitizen from "Yes"/"No" to 1/0
        if 'SeniorCitizen' in data:
            data['SeniorCitizen'] = 1 if data['SeniorCitizen'] == 'Yes' else 0
        
        # Process features in correct order
        for feature in expected_features:
            value = data.get(feature)
            
            if feature in encoder_dict:
                # Handle categorical feature encoding
                try:
                    encoded_value = encoder_dict[feature].transform([value])[0]
                except ValueError as e:
                    return jsonify({'error': f"Invalid value '{value}' for feature '{feature}'"}), 400
                formatted_data.append(encoded_value)
            else:
                # Handle numerical features
                try:
                    formatted_data.append(float(value))
                except ValueError:
                    return jsonify({'error': f"Invalid numerical value for '{feature}'"}), 400
        
        # Make prediction
        prediction = model.predict(np.array([formatted_data]))[0]
        session['prediction'] = "The customer will churn" if prediction == 1 else "The customer will not churn"
        
        return jsonify({'prediction': session['prediction']})

    except Exception as e:
        print(f"Error during prediction: {str(e)}")  # Log error for debugging
        return jsonify({'error': str(e)}), 400

# ... (rest of the app remains the same)

@app.route('/get_prediction')
def get_prediction():
    prediction = session.get('prediction', None)
    if prediction:
        return jsonify({'prediction': prediction})
    return jsonify({'error': 'No prediction found'}), 404

if __name__ == "__main__":
    app.run(debug=True)

