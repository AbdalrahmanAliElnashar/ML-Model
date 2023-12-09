from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the saved Random Forest model
loaded_model = joblib.load('Random_Forest_Model.pkl')

# Load the saved StandardScaler
sc = joblib.load('standard_scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form submission
        features = [
            float(request.form['HR']),
            float(request.form['RESP']),
            float(request.form['SpO2']),
            float(request.form['TEMP'])
        ]
        
        input_data = np.array(features).reshape(1, -1)

        # Normalize the input data using the loaded StandardScaler
        input_data = sc.transform(input_data)

        # Make predictions using the loaded model
        prediction = int(loaded_model.predict(input_data)[0])

        # Map the prediction to a meaningful label
        prediction_label = "Normal" if prediction == 1 else "Abnormal"

        # Render the index page with the prediction label
        return render_template('index.html', prediction=prediction_label)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predictApi', methods=['POST'])
def predict_api():
    try:
        # Extract features from the JSON data
        data = request.get_json()

        # Extract features from the JSON data
        features = [
            float(data['HR']),
            float(data['RESP']),
            float(data['SpO2']),
            float(data['TEMP'])
        ]

        input_data = np.array(features).reshape(1, -1)

        # Normalize the input data using the loaded StandardScaler
        input_data = sc.transform(input_data)

        # Make predictions using the loaded model
        prediction = int(loaded_model.predict(input_data)[0])

        # Map the prediction to a meaningful label
        prediction_label = "Normal" if prediction == 1 else "Abnormal"

        # Return the prediction in JSON format
        return jsonify({'prediction': prediction_label})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000)
