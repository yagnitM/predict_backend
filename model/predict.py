import joblib
import pandas as pd

# Load encoders
player_to_int = joblib.load('encoders/player_to_int.pkl')
surface_to_int = joblib.load('encoders/surface_to_int.pkl')

def predict_winner(player1, player2, surface):
    # Load model and column structure
    data = joblib.load('saved_model_v2.pkl')
    model = data['model']
    columns = data['columns']

    # Prepare empty input row
    input_df = pd.DataFrame(columns=columns)
    input_df.loc[0] = 0

    # Encode players
    if player1 in player_to_int.classes_:
        input_df.at[0, 'player1_enc'] = player_to_int.transform([player1])[0]
    else:
        print(f"⚠️ Warning: Player1 '{player1}' not found in training data.")

    if player2 in player_to_int.classes_:
        input_df.at[0, 'player2_enc'] = player_to_int.transform([player2])[0]
    else:
        print(f"⚠️ Warning: Player2 '{player2}' not found in training data.")

    # Encode surface
    if surface in surface_to_int.classes_:
        input_df.at[0, 'surface_enc'] = surface_to_int.transform([surface])[0]
    else:
        print(f"⚠️ Warning: Surface '{surface}' not found in training data.")

    # Make prediction
    proba = model.predict_proba(input_df)[0]
    prob_player1 = proba[1]
    prob_player2 = proba[0]

    winner = player1 if prob_player1 >= prob_player2 else player2
    confidence = max(prob_player1, prob_player2) * 100

    # Output result
    print(f"\n🎾 Predicted winner: {winner}")
    print(f"📊 Winning chance: {confidence:.2f}%")

if __name__ == "__main__":
    player1 = input("Enter player 1: ").strip()
    player2 = input("Enter player 2: ").strip()
    surface = input("Enter surface (Hard, Clay, Grass): ").strip()

    predict_winner(player1, player2, surface)
