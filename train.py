import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from model import MultiClassSVM

# Load dataset
df = pd.read_csv('Crop_recommendation.csv')
label_encoder = LabelEncoder()
df['crop'] = label_encoder.fit_transform(df['label'])

scaler = StandardScaler()
X = scaler.fit_transform(df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']])
y = df['crop']

# Train
multi_svm = MultiClassSVM()
multi_svm.fit(X, y)

# Save model, scaler, label encoder
with open('svm_scratch_model.pkl', 'wb') as f:
    pickle.dump({
        'model': multi_svm,
        'scaler': scaler,
        'label_encoder': label_encoder
    }, f)

print("Save model into 'svm_scratch_model.pkl'")
