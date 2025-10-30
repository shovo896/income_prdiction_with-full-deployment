from flask import Flask, render_template, request
import joblib
import pandas as pd
from preprocessing import X_train

app = Flask(__name__)

model = joblib.load('model.pkl')
feature_names = X_train.columns.tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Form data 
        form_data = request.form.to_dict()

        # Encode 
        df = pd.DataFrame([form_data])

        # Convert numeric columns
        numeric_cols = ['age', 'hours-per-week']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])

        # Dummy encode categorical columns (same columns as training set)
        df = pd.get_dummies(df)
        df = df.reindex(columns=feature_names, fill_value=0)

        # Prediction
        pred = model.predict(df)[0]
        result = "Income >50K" if pred == 1 else "Income <=50K"

        return render_template('result.html', prediction=result)
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)


