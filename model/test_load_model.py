import joblib

try:
    model_data = joblib.load('saved_model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print("Failed loading model:", e)
