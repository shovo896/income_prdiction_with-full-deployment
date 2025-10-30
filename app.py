

from flask import Flask, request, render_template
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
        # form data 
        data = request.form.to_dict()
        data = {k: float(v) for k, v in data.items()}  # সব value কে float-এ convert করা
        
        
        input_data = pd.DataFrame([data], columns=feature_names)
        
        # Prediction করা
        prediction = model.predict(input_data)[0]
        result = "Income >50K" if prediction == 1 else "Income <=50K"
        
       
        return render_template('result.html', prediction=result)
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)

