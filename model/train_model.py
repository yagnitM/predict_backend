import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
from tqdm import tqdm

# Load data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_matches.csv')
df = pd.read_csv(data_path)

# Create balanced dataset with progress bar
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

# Label encode categorical features
le_player1 = LabelEncoder()
le_player2 = LabelEncoder()
le_surface = LabelEncoder()

balanced_df['player1_enc'] = le_player1.fit_transform(balanced_df['player1'])
balanced_df['player2_enc'] = le_player2.fit_transform(balanced_df['player2'])
balanced_df['surface_enc'] = le_surface.fit_transform(balanced_df['surface'])

# Prepare features and target
X = balanced_df[['player1_enc', 'player2_enc', 'surface_enc']]
y = balanced_df['result']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train smaller RandomForest model
model = RandomForestClassifier(n_estimators=40, max_depth=15, random_state=42, n_jobs=-1)
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

# Save model and label encoders + columns with compression
joblib.dump({
    'model': model,
    'le_player1': le_player1,
    'le_player2': le_player2,
    'le_surface': le_surface,
    'columns': X.columns.tolist()
}, model_path, compress=3)
print(f"💾 Model saved with compression to {model_path}")
