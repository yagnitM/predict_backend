import os
import pandas as pd

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_matches.csv')

def load_and_merge_csvs():
    all_matches = []
    for filename in os.listdir(RAW_DATA_DIR):
        if filename.startswith("atp_matches_") and filename.endswith(".csv"):
            file_path = os.path.join(RAW_DATA_DIR, filename)
            df = pd.read_csv(file_path)
            all_matches.append(df)
    
    print(f"Loaded {len(all_matches)} files.")
    combined_df = pd.concat(all_matches, ignore_index=True)
    print(f"Total matches: {len(combined_df)}")
    return combined_df

def clean_data(df):
    # Drop rows where critical info is missing
    important_cols = ['winner_name', 'loser_name', 'surface', 'score']
    df = df.dropna(subset=important_cols)

    # Filter out walkovers or incomplete matches
    df = df[~df['score'].str.contains('W/O', na=False)]

    # Optional: remove matches missing stats like aces, double faults, etc.
    df = df.dropna(subset=['w_ace', 'l_ace', 'w_df', 'l_df', 'best_of'])

    # Reset index
    df.reset_index(drop=True, inplace=True)
    return df

def preprocess():
    df = load_and_merge_csvs()

    # Convert 'tourney_date' to year if present
    if 'tourney_date' in df.columns:
        df['year'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d').dt.year
    elif 'year' not in df.columns:
        raise ValueError("No 'year' or 'tourney_date' column found for filtering by year.")

    # Filter for matches from 2000 onward
    df = df[df['year'] >= 2000]

    print(f"Matches after filtering by year >= 2000: {len(df)}")

    df_cleaned = clean_data(df)
    save_processed_data(df_cleaned)

def save_processed_data(df):
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved cleaned data to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    preprocess()
