from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model + scaler + label encoder
with open('svm_scratch_model.pkl', 'rb') as f:
    checkpoint = pickle.load(f)

model = checkpoint['model']
scaler = checkpoint['scaler']
label_encoder = checkpoint['label_encoder']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = [
                float(request.form['N']),
                float(request.form['P']),
                float(request.form['K']),
                float(request.form['temperature']),
                float(request.form['humidity']),
                float(request.form['ph']),
                float(request.form['rainfall'])
            ]
            data_scaled = scaler.transform([data])
            prediction = model.predict(data_scaled)
            crop_name = label_encoder.inverse_transform(prediction)[0]
            return render_template('predict.html', crop=crop_name, values=request.form)
        except Exception as e:
            return f"Error: {e}"
    return render_template('predict.html')

@app.route('/model')
def model_info():
    return render_template('model.html')

# KHỞI CHẠY SERVER
if __name__ == '__main__':
    app.run(debug=True)