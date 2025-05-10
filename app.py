from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load tất cả mô hình trong thư mục hiện tại
model_files = [
    ('SVM (Scratch)', 'svm_scratch_model.pkl'),
    ('KNN (Scratch)', 'knn_scratch_model.pkl'),
    ('Decision Tree (Scratch)', 'dt_scratch_model.pkl'),
    ('Random Forest (Scratch)', 'rf_scratch_model.pkl'),
    ('Logistic Regression (Scratch)', 'logistic_scratch_model.pkl')
]

models = {}
for name, file in model_files:
    with open(file, 'rb') as f:
        checkpoint = pickle.load(f)
        models[name] = {
            'model': checkpoint['model'],
            'scaler': checkpoint['scaler'],
            'label_encoder': checkpoint['label_encoder']
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            form_data = request.form
            # print(f"Form data: {form_data}")  # Debug form input
            feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            data = pd.DataFrame(
                [[
                    float(form_data['N']),
                    float(form_data['P']),
                    float(form_data['K']),
                    float(form_data['temperature']),
                    float(form_data['humidity']),
                    float(form_data['ph']),
                    float(form_data['rainfall'])
                ]],
                columns=feature_names
            )
            # print(f"Input data shape: {data.shape}")  # Debug input shape
            results = {}
            for model_name, obj in models.items():
                # print(f"Predicting with model: {model_name}")
                scaled = obj['scaler'].transform(data)
                # print(f"Scaled data shape: {scaled.shape}")
                pred = obj['model'].predict(scaled)
                # print(f"Raw prediction output: {pred}, type: {type(pred)}")
                # Ensure pred is a 1D NumPy array
                pred = np.array(pred, dtype=object).flatten()
                # print(f"Processed prediction: {pred}, shape: {pred.shape}")
                # Verify label_encoder compatibility
                crop = obj['label_encoder'].inverse_transform(pred.astype(int))[0]
                results[model_name] = crop
                # print(f"Predicted crop for {model_name}: {crop}")

            return render_template('predict.html', results=results, values=form_data)
        except Exception as e:
            print(f"Error details: {str(e)}")
            return f"Error: {str(e)}"
    return render_template('predict.html')

@app.route('/model')
def model_info():
    return render_template('model.html')

if __name__ == '__main__':
    app.run(debug=True)