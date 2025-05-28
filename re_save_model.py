import joblib

# Load old model (with numpy 1.24.4)
model = joblib.load("saved_model.pkl")  

# Re-save with compatible format
joblib.dump(model, "saved_model_v2.pkl")  
print("Model re-saved successfully! New file: saved_model_v2.pkl")