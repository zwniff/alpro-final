from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS

import pandas as pd
import pickle
import os

# Load the model from the pkl file
with open('Lab/model_lgb.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Serve the HTML page
@app.route('/')
def serve_index():
    return render_template('index.html')

@app.route('/feature')
def serve_feature():
    return render_template('feature.html')

@app.route('/blog')
def serve_blog():
    return render_template('blog.html')

@app.route('/about')
def serve_about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        # Check for required fields
        required_fields = ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Gender encoding
        gender = data['Gender'].strip().lower()
        if gender == 'male':
            gender_encoded = 0
        elif gender == 'female':
            gender_encoded = 1
        else:
            return jsonify({'error': 'Invalid Gender value. Must be either "Male" or "Female".'}), 400

        # Prepare input data
        input_data = pd.DataFrame([[
            gender_encoded,
            float(data['AGE']),
            float(data['Urea']),
            float(data['Cr']),
            float(data['HbA1c']),
            float(data['Chol']),
            float(data['TG']),
            float(data['HDL']),
            float(data['LDL']),
            float(data['VLDL']),
            float(data['BMI'])
        ]], columns=required_fields)

        # Make prediction
        prediction = model.predict(input_data)

        # Map prediction to labels
        prediction_labels = {
            0: "No-Diabetes",
            1: "Pre-Diabetes",
            2: "Diabetes"
        }
        result_label = prediction_labels.get(int(prediction[0]), "Unknown")

        # Return the predicted result
        return jsonify({'prediction': result_label})

    except ValueError as e:
        return jsonify({'error': 'Invalid input value: ' + str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred during prediction: ' + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)