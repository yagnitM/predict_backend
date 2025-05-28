import pandas as pd
import os
import pickle
from sklearn.preprocessing import LabelEncoder

# Ensure 'encoders/' folder exists
os.makedirs("encoders", exist_ok=True)

# Load the data
data_path = "data/processed_matches.csv"
df = pd.read_csv(data_path)

# Extract player names and surfaces
players = pd.concat([df["winner_name"], df["loser_name"]]).unique()
surfaces = df["surface"].dropna().unique()

# Create label encoders
player_encoder = LabelEncoder()
surface_encoder = LabelEncoder()

# Fit encoders
player_encoder.fit(players)
surface_encoder.fit(surfaces)

# Save encoders
with open("encoders/player_to_int.pkl", "wb") as f:
    pickle.dump(player_encoder, f)

with open("encoders/surface_to_int.pkl", "wb") as f:
    pickle.dump(surface_encoder, f)

print("✅ Encoders created and saved in 'encoders/' folder.")
