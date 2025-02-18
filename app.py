import numpy as np
import pandas as pd
from flask import Flask,request,jsonify, render_template
import pickle
#create a flask app
app = Flask(__name__)

#load the pickle model
model = pickle.load(open('ChurnModel.pkl', 'rb'))
encoder_dict = pickle.load(open('encoders.pkl', 'rb'))
scaler = pickle.load(open('scaler.pickle', 'rb'))  # Load the scaler

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Receive input JSON
        print("Received Data:", data)  # Debugging

        formatted_data = []

        for column, value in data.items():
            if column in encoder_dict:  # If the column needs encoding
                encoded_value = encoder_dict[column].transform([value])[0]  # Apply LabelEncoder
                formatted_data.append(encoded_value)
            elif column in ['tenure', 'MonthlyCharges', 'TotalCharges']:  # Scale numeric values
                formatted_data.append(float(value))  # Convert to float
            else:
                formatted_data.append(float(value))  # Convert other numerical values

        # Scale numerical features
        numeric_values = np.array([formatted_data[:3]])  # Extract tenure, MonthlyCharges, TotalCharges
        scaled_values = scaler.transform(numeric_values)[0]  # Apply scaling

        # Replace original values with scaled ones
        formatted_data[:3] = scaled_values.tolist()

        formatted_data = np.array([formatted_data])  # Convert to 2D array for the model
        print("Formatted Data:", formatted_data)  # Debugging

        # Make prediction
        prediction = model.predict(formatted_data)[0]  # Get single prediction

        # Convert numerical prediction to meaningful text
        result_text = "The customer will churn" if prediction == 1 else "The customer will not churn"
        print("Prediction Result:", result_text)  # Debugging

        return jsonify({'prediction': result_text})

    except Exception as e:
        print("Error:", str(e))  # Debugging
        return jsonify({'error': str(e)}), 400
if __name__ == "__main__":
    app.run(debug=True)
