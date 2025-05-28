import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from tqdm import tqdm

# Load data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_matches.csv')
df = pd.read_csv(data_path)

# Filter relevant columns and create balanced dataset with progress bar
matches = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating balanced dataset"):
    matches.append({
        'player1': row['winner_name'],
        'player2': row['loser_name'],
        'surface': row['surface'],
        'result': 1
    })
    matches.append({
        'player1': row['loser_name'],
        'player2': row['winner_name'],
        'surface': row['surface'],
        'result': 0
    })

balanced_df = pd.DataFrame(matches)

# Shuffle dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# One-hot encode features
X = pd.get_dummies(balanced_df[['player1', 'player2', 'surface']])
y = balanced_df['result']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model with verbose to see progress in console
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully!")
print(f"📈 Accuracy on test set: {accuracy * 100:.2f}%")

# Delete existing saved model if it exists
model_path = 'saved_model_v2.pkl'
if os.path.exists(model_path):
    os.remove(model_path)

# Save model and feature columns with compression
joblib.dump({'model': model, 'columns': X.columns.tolist()}, model_path, compress=3)
print(f"💾 Model saved with compression to {model_path}")
