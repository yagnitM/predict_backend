from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import json
import os
import requests

app = FastAPI()

# CORS config for frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def download_model_if_needed():
    model_path = "saved_model.pkl"
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        model_url = "https://drive.google.com/uc?export=download&id=1KZWOEyoklJ7XZySAFd-oQG1FTrRQWOyb"
        response = requests.get(model_url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully!")
    return joblib.load(model_path)

# Load model and columns once on startup
model_data = download_model_if_needed()
model = model_data['model']
columns = model_data['columns']

# Load filtered players list from JSON
with open("filtered_players.json", "r") as f:
    players = json.load(f)

class PredictRequest(BaseModel):
    player1: str
    player2: str
    surface: str  # "Hard", "Clay", "Grass"

@app.get("/")
async def root():
    return {"message": "AcePredictor backend is running"}

@app.get("/players")
async def get_players():
    return {"players": players}

@app.post("/predict")
async def predict(req: PredictRequest):
    input_df = pd.DataFrame(columns=columns)
    input_df.loc[0] = 0

    key_player1 = "player1_" + req.player1
    key_player2 = "player2_" + req.player2
    key_surface = "surface_" + req.surface

    if key_player1 not in columns:
        return {"error": f"Player1 '{req.player1}' not found in model features."}
    if key_player2 not in columns:
        return {"error": f"Player2 '{req.player2}' not found in model features."}
    if key_surface not in columns:
        return {"error": f"Surface '{req.surface}' not found in model features."}

    input_df.at[0, key_player1] = 1
    input_df.at[0, key_player2] = 1
    input_df.at[0, key_surface] = 1

    proba = model.predict_proba(input_df)[0]
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