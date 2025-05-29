# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import joblib
# import pandas as pd
# import json
# import os
# import numpy

# app = FastAPI()

# # CORS config for frontend calls
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global variables for lazy loading
# model_data = None
# model = None
# columns = None
# players = None

# def load_model():
#     global model_data, model, columns
#     if model is None:
#         model_path = "saved_model_v2.pkl"
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Model file {model_path} not found. Please upload it.")
#         model_data = joblib.load(model_path)
#         model = model_data['model']
#         columns = model_data['columns']
#         print("Model loaded successfully!")

# def load_players():
#     global players
#     if players is None:
#         try:
#             with open("filtered_players.json", "r") as f:
#                 players = json.load(f)
#             print("Players loaded successfully!")
#         except Exception as e:
#             print(f"Error loading players: {e}")
#             players = []

# class PredictRequest(BaseModel):
#     player1: str
#     player2: str
#     surface: str  # "Hard", "Clay", "Grass"

# @app.on_event("startup")
# async def startup_event():
#     print("Running startup event: loading model and players...")
#     print(f"NumPy version on startup: {numpy.__version__}")
#     try:
#         load_model()
#         load_players()
#         print("Startup: Model and players loaded successfully!")
#     except Exception as e:
#         print(f"Startup failed: {e}")

# @app.get("/")
# async def root():
#     return {"message": "AcePredictor backend is running"}

# @app.get("/health")
# async def health():
#     try:
#         load_model()
#     except Exception as e:
#         print(f"Health check: model loading failed: {e}")
#     try:
#         load_players()
#     except Exception as e:
#         print(f"Health check: players loading failed: {e}")
    
#     return {
#         "status": "healthy",
#         "model_loaded": model is not None,
#         "players_loaded": players is not None
#     }

# @app.get("/players")
# async def get_players():
#     load_players()
#     return {"players": players}

# @app.post("/predict")
# async def predict(req: PredictRequest):
#     try:
#         print("Received request:", req)
#         print(f"Predicting: {req.player1} vs {req.player2} on {req.surface}")
#         load_model()

#         input_df = pd.DataFrame([[0] * len(columns)], columns=columns)
#         print("Initialized input_df with zeros.")

#         key_player1 = "player1_" + req.player1
#         key_player2 = "player2_" + req.player2
#         key_surface = "surface_" + req.surface

#         print(f"Checking keys: {key_player1}, {key_player2}, {key_surface}")

#         if key_player1 not in columns:
#             return {"error": f"Player1 '{req.player1}' not found in model features."}
#         if key_player2 not in columns:
#             return {"error": f"Player2 '{req.player2}' not found in model features."}
#         if key_surface not in columns:
#             return {"error": f"Surface '{req.surface}' not found in model features."}

#         input_df.at[0, key_player1] = 1
#         input_df.at[0, key_player2] = 1
#         input_df.at[0, key_surface] = 1

#         print("DataFrame ready for prediction:", input_df)

#         proba = model.predict_proba(input_df)[0]
#         print("Prediction probabilities:", proba)

#         prob_player1_wins = proba[1]
#         prob_player2_wins = proba[0]

#         if prob_player1_wins >= prob_player2_wins:
#             winner = req.player1
#             confidence = prob_player1_wins * 100
#         else:
#             winner = req.player2
#             confidence = prob_player2_wins * 100

#         return {
#             "winner": winner,
#             "confidence": round(confidence, 2)
#         }

#     except Exception as e:
#         print("Prediction error:", str(e))
#         return {"error": f"Prediction failed: {str(e)}"}

# @app.get("/debug/columns")
# async def debug_columns():
#     try:
#         load_model()
#         if columns is None:
#             return {"error": "Model loaded but columns is still None"}
#         return {"columns": columns, "count": len(columns)}
#     except Exception as e:
#         print("Error in /debug/columns:", str(e))
#         return {"error": f"Failed to load columns: {str(e)}"}

# @app.get("/debug/model-file")
# async def debug_model_file():
#     exists = os.path.exists("saved_model_v2.pkl")
#     size = os.path.getsize("saved_model_v2.pkl") if exists else 0
#     return {"exists": exists, "size_bytes": size}
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import json
import os
import numpy

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
player_to_int = None
surface_to_int = None

def load_model():
    global model_data, model, columns, player_to_int, surface_to_int
    if model is None:
        model_path = "saved_model_v2.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Please upload it.")
        model_data = joblib.load(model_path)
        model = model_data['model']
        columns = model_data['columns']
        
        # Load encoders
        player_to_int = joblib.load('encoders/player_to_int.pkl')
        surface_to_int = joblib.load('encoders/surface_to_int.pkl')
        
        print("Model and encoders loaded successfully!")

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
        "players_loaded": players is not None,
        "encoders_loaded": player_to_int is not None and surface_to_int is not None
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

        # Prepare empty input row (same as predict.py)
        input_df = pd.DataFrame(columns=columns)
        input_df.loc[0] = 0

        # Encode players
        if req.player1 in player_to_int.classes_:
            input_df.at[0, 'player1_enc'] = player_to_int.transform([req.player1])[0]
        else:
            return {"error": f"Player1 '{req.player1}' not found in training data."}

        if req.player2 in player_to_int.classes_:
            input_df.at[0, 'player2_enc'] = player_to_int.transform([req.player2])[0]
        else:
            return {"error": f"Player2 '{req.player2}' not found in training data."}

        # Encode surface
        if req.surface in surface_to_int.classes_:
            input_df.at[0, 'surface_enc'] = surface_to_int.transform([req.surface])[0]
        else:
            return {"error": f"Surface '{req.surface}' not found in training data."}

        print("DataFrame ready for prediction:", input_df)

        # Make prediction
        proba = model.predict_proba(input_df)[0]
        print("Prediction probabilities:", proba)

        prob_player1 = proba[1]
        prob_player2 = proba[0]

        if prob_player1 >= prob_player2:
            winner = req.player1
            confidence = prob_player1 * 100
        else:
            winner = req.player2
            confidence = prob_player2 * 100

        return {
            "winner": winner,
            "confidence": round(confidence, 2),
            "player1_probability": round(prob_player1 * 100, 2),
            "player2_probability": round(prob_player2 * 100, 2)
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
    exists = os.path.exists("saved_model_v2.pkl")
    size = os.path.getsize("saved_model_v2.pkl") if exists else 0
    return {"exists": exists, "size_bytes": size}

@app.get("/debug/encoders")
async def debug_encoders():
    try:
        load_model()
        player_classes = list(player_to_int.classes_) if player_to_int else []
        surface_classes = list(surface_to_int.classes_) if surface_to_int else []
        
        return {
            "player_encoder_loaded": player_to_int is not None,
            "surface_encoder_loaded": surface_to_int is not None,
            "available_players": player_classes[:10],  # First 10 players
            "total_players": len(player_classes),
            "available_surfaces": surface_classes
        }
    except Exception as e:
        return {"error": f"Failed to load encoders: {str(e)}"}