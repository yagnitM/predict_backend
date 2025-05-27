import joblib
import pandas as pd

def predict_winner(player1, player2, surface):
    # Load saved model and columns
    data = joblib.load('saved_model.pkl')
    model = data['model']
    columns = data['columns']

    # Prepare input dataframe with zeroes
    input_df = pd.DataFrame(columns=columns)
    input_df.loc[0] = 0

    # Build feature keys
    key_player1 = 'player1_' + player1
    key_player2 = 'player2_' + player2
    key_surface = 'surface_' + surface

    # Set 1 for these features if they exist in the training columns
    if key_player1 in columns:
        input_df.at[0, key_player1] = 1
    else:
        print(f"⚠️ Warning: Player1 '{player1}' not found in training data.")

    if key_player2 in columns:
        input_df.at[0, key_player2] = 1
    else:
        print(f"⚠️ Warning: Player2 '{player2}' not found in training data.")

    if key_surface in columns:
        input_df.at[0, key_surface] = 1
    else:
        print(f"⚠️ Warning: Surface '{surface}' not found in training data.")

    # Predict probabilities
    proba = model.predict_proba(input_df)[0]
    prob_player1_wins = proba[1]
    prob_player2_wins = proba[0]

    if prob_player1_wins >= prob_player2_wins:
        winner = player1
        confidence = prob_player1_wins * 100
    else:
        winner = player2
        confidence = prob_player2_wins * 100

    print(f"\n🎾 Predicted winner: {winner}")
    print(f"📊 Winning chance: {confidence:.2f}%")

if __name__ == "__main__":
    player1 = input("Enter player 1: ").strip()
    player2 = input("Enter player 2: ").strip()
    surface = input("Enter surface (Hard, Clay, Grass): ").strip()

    predict_winner(player1, player2, surface)
