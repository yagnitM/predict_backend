import pandas as pd
import json

df = pd.read_csv('data/processed_matches.csv')  # Replace with actual filename

players1 = set(df['winner_name'].unique())
players2 = set(df['loser_name'].unique())

all_players = players1.union(players2)

# Filter players who played in or after 2000, either as winner or loser
filtered_players = []
for player in all_players:
    # max year where player won
    max_year_winner = df[df['winner_name'] == player]['year'].max()
    # max year where player lost
    max_year_loser = df[df['loser_name'] == player]['year'].max()
    
    max_year = max(filter(lambda x: x == x, [max_year_winner, max_year_loser]))  # filter NaN
    
    if max_year >= 2000:
        filtered_players.append(player)

# Save to JSON
with open('filtered_players.json', 'w') as f:
    json.dump(filtered_players, f)

print(f"Filtered {len(filtered_players)} players active since 2000.")
