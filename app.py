from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import json
import os
import requests
import gdown
import numpy

# NumPy compatibility fix for _core module and structseq issues
# import sys
# try:
#     import numpy._core
# except ImportError:
#     import numpy.core as _core
#     numpy._core = _core
#     sys.modules['numpy._core'] = _core

# # Fix for structseq compatibility issues
# import warnings
# warnings.filterwarnings('ignore', category=FutureWarning)
# warnings.filterwarnings('ignore', category=UserWarning)

# # Additional compatibility patches
# try:
#     import numpy.core._multiarray_umath
# except ImportError:
#     pass

# # Set numpy array type compatibility
# import numpy as np
# if hasattr(np, 'set_printoptions'):
#     np.set_printoptions(legacy='1.13')

app = FastAPI()

# CORS config for frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for lazy loading
model_data = None
model = None
columns = None
players = None

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def download_model_if_needed():
    model_path = "saved_model.pkl"
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        if size > 10_000_000:  # 10 MB sanity check
            print(f"Model already exists with size {size} bytes.")
            try:
                return joblib.load(model_path)
            except Exception as e:
                print(f"Error loading existing model with joblib: {e}")
                print("Trying alternative loading method...")
                try:
                    import pickle
                    with open(model_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e2:
                    print(f"Alternative loading also failed: {e2}")
                    print("Deleting corrupted model and re-downloading...")
                    os.remove(model_path)
        else:
            print(f"Existing model file too small ({size} bytes), deleting...")
            os.remove(model_path)

    print("Downloading model using gdown...")
    url = "https://drive.google.com/uc?id=1KZWOEyoklJ7XZySAFd-oQG1FTrRQWOyb"
    gdown.download(url, model_path, quiet=False)

    size = os.path.getsize(model_path)
    print(f"Downloaded model size: {size} bytes")

    if size < 10_000_000:
        raise Exception("Downloaded file is too small, probably incorrect!")

    # Try joblib first, then pickle as fallback
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading with joblib: {e}")
        print("Trying with pickle...")
        try:
            import pickle
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e2:
            print(f"Error loading with pickle: {e2}")
            raise e

def load_model():
    global model_data, model, columns
    if model is None:
        try:
            model_data = download_model_if_needed()
            model = model_data['model']
            columns = model_data['columns']
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

def load_players():
    global players
    if players is None:
        try:
            with open("filtered_players.json", "r") as f:
                players = json.load(f)
            print("Players loaded successfully!")
        except Exception as e:
            print(f"Error loading players: {e}")
            players = []

class PredictRequest(BaseModel):
    player1: str
    player2: str
    surface: str  # "Hard", "Clay", "Grass"

@app.on_event("startup")
async def startup_event():
    print("Running startup event: loading model and players...")
    print(f"NumPy version on startup: {numpy.__version__}")
    try:
        load_model()
        load_players()
        print("Startup: Model and players loaded successfully!")
    except Exception as e:
        print(f"Startup failed: {e}")

@app.get("/")
async def root():
    return {"message": "AcePredictor backend is running"}

@app.get("/health")
async def health():
    try:
        load_model()
    except Exception as e:
        print(f"Health check: model loading failed: {e}")
    try:
        load_players()
    except Exception as e:
        print(f"Health check: players loading failed: {e}")
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "players_loaded": players is not None
    }

@app.get("/players")
async def get_players():
    load_players()
    return {"players": players}

@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        print("Received request:", req)
        print(f"Predicting: {req.player1} vs {req.player2} on {req.surface}")
        load_model()

        input_df = pd.DataFrame([[0] * len(columns)], columns=columns)
        print("Initialized input_df with zeros.")

        key_player1 = "player1_" + req.player1
        key_player2 = "player2_" + req.player2
        key_surface = "surface_" + req.surface

        print(f"Checking keys: {key_player1}, {key_player2}, {key_surface}")

        if key_player1 not in columns:
            return {"error": f"Player1 '{req.player1}' not found in model features."}
        if key_player2 not in columns:
            return {"error": f"Player2 '{req.player2}' not found in model features."}
        if key_surface not in columns:
            return {"error": f"Surface '{req.surface}' not found in model features."}

        input_df.at[0, key_player1] = 1
        input_df.at[0, key_player2] = 1
        input_df.at[0, key_surface] = 1

        print("DataFrame ready for prediction:", input_df)

        proba = model.predict_proba(input_df)[0]
        print("Prediction probabilities:", proba)

        prob_player1_wins = proba[1]
        prob_player2_wins = proba[0]

        if prob_player1_wins >= prob_player2_wins:
            winner = req.player1
            confidence = prob_player1_wins * 100
        else:
            winner = req.player2
            confidence = prob_player2_wins * 100

        return {
            "winner": winner,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        print("Prediction error:", str(e))
        return {"error": f"Prediction failed: {str(e)}"}

@app.get("/debug/columns")
async def debug_columns():
    try:
        load_model()
        if columns is None:
            return {"error": "Model loaded but columns is still None"}
        return {"columns": columns, "count": len(columns)}
    except Exception as e:
        print("Error in /debug/columns:", str(e))
        return {"error": f"Failed to load columns: {str(e)}"}

@app.get("/debug/model-file")
async def debug_model_file():
    exists = os.path.exists("saved_model.pkl")
    size = os.path.getsize("saved_model.pkl") if exists else 0
    return {"exists": exists, "size_bytes": size}