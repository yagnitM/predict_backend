import pandas as pd
import json

# Load dataset
df = pd.read_csv('data/processed_matches.csv')  # Replace with your actual file path

# Get all unique player IDs from winner and loser columns
player_ids_winner = set(df['winner_id'].unique())
player_ids_loser = set(df['loser_id'].unique())

all_player_ids = player_ids_winner.union(player_ids_loser)

# Map player ID to player name (using winner_name and loser_name columns)
id_to_name = {}

# Build a mapping from id to name using winner rows
for _, row in df.iterrows():
    if row['winner_id'] not in id_to_name:
        id_to_name[row['winner_id']] = row['winner_name']
    if row['loser_id'] not in id_to_name:
        id_to_name[row['loser_id']] = row['loser_name']

# Filter players active since year 2000
active_player_ids = set()

for pid in all_player_ids:
    # max year where player won
    max_year_winner = df[df['winner_id'] == pid]['year'].max()
    # max year where player lost
    max_year_loser = df[df['loser_id'] == pid]['year'].max()

    max_year = max(filter(lambda x: pd.notna(x), [max_year_winner, max_year_loser]))

    if max_year >= 2000:
        active_player_ids.add(pid)

# Create dictionary with player name as key and player id (int) as value
active_players = {id_to_name[pid]: int(pid) for pid in active_player_ids if pid in id_to_name}

# Save to JSON file
with open('filtered_players.json', 'w') as f:
    json.dump(active_players, f, indent=2)

print(f"Filtered {len(active_players)} active players since 2000.")
