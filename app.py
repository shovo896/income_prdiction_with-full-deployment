
from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load('model.pkl')

# Load feature names from preprocessing (optional)
from preprocessing import X_train
feature_names = X_train.columns.tolist()

@app.route('/')
def home():
    return "<h2>🚀 Decision Tree Income Predictor is Running!</h2><p>Use /predict endpoint to send data.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Example JSON input:
    {
        "age": 39,
        "workclass": 4,
        "fnlwgt": 77516,
        "education": 9,
        "education_num": 13,
        "marital_status": 2,
        "occupation": 1,
        "relationship": 1,
        "race": 4,
        "gender": 1,
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": 39
    }
    """
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data], columns=feature_names)
        prediction = model.predict(input_data)[0]

        result = "Income >50K" if prediction == 1 else "Income <=50K"
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
