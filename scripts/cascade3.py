import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib
import os

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_to_train.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "model_C_capture.pkl")

# FEATURES FOR CAPTURE
FEATURES = [
    'encirclement_score',    # Output from Model B (CRITICAL)
    'dist_to_front_m',       # Output from Model A (CRITICAL)
    'pct_occupied',          # Do they already have a foot inside?
    'fortification_level',   # Defenses (more casualties needed to take it)
    'population',            # Large cities are urban nightmares (Stalingrad)
    'geo_area_km2',          # Area to cover
    'ru_total_lag7',         # Do Russians have combat forces available?
    'uk_total',              # Do Ukrainians have forces to defend?
    'momentum_30d'           # If they come with momentum, more likely to finish
]

TARGET = 'target_is_captured_next'

def train_model_c():
    print("--- Loading data for Model C (Final Capture) ---")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    
    # 1. STATE FILTER: We only learn from cities that HAVE NOT YET fallen.
    # If is_captured is already 1, there's nothing to predict.
    free_cities_mask = df['is_captured'] == 0
    
    # 2. KILL ZONE FILTER:
    # We only consider capture risk if they are very close to the front (< 5km)
    # Or if troops are already inside (pct_occupied > 0)
    kill_zone_mask = (df['dist_to_front_m'] < 5000) | (df['pct_occupied'] > 0.1)
    
    # 3. Temporal Split (Standard)
    train_time = (df['date'] < '2024-01-01')
    val_time = (df['date'] >= '2024-01-01') & (df['date'] < '2024-07-01')
    test_time = (df['date'] >= '2024-07-01')

    # Master Mask
    valid_rows = free_cities_mask & kill_zone_mask
    
    print(f"Total rows: {len(df)}")
    print(f"'Final Assault' situations (Zone <5km + Free City): {valid_rows.sum()}")

    # Sets
    X_train = df.loc[train_time & valid_rows, FEATURES]
    y_train = df.loc[train_time & valid_rows, TARGET]
    w_train = df.loc[train_time & valid_rows, 'sample_weight']

    X_val = df.loc[val_time & valid_rows, FEATURES]
    y_val = df.loc[val_time & valid_rows, TARGET]

    X_test = df.loc[test_time & valid_rows, FEATURES]
    y_test = df.loc[test_time & valid_rows, TARGET]

    # --- TRAINING (CLASSIFIER) ---
    # We use XGBClassifier because the output is YES/NO (1/0)
    # scale_pos_weight: Vital because there are few captures compared to resistance days.
    # Helps the model not be too conservative.
    ratio = float(len(y_train[y_train == 0])) / len(y_train[y_train == 1])
    
    print(f"\nTraining Classifier (Imbalance Ratio: {ratio:.1f})...")
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=ratio, # Compensates for captures being rare events
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )

    # --- EVALUATION ---
    preds = model.predict(X_test)
    
    print("\n--- Model C Results ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"F1-Score (Capture): {f1_score(y_test, preds):.4f}")
    
    print("\nDetailed Report:")
    print(classification_report(y_test, preds, target_names=['Resists', 'Captured']))

    # --- FEATURE IMPORTANCE ---
    print("\n--- What defines a city's fall? ---")
    importance = pd.DataFrame({
        'feature': FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance.head(10))

    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"\nModel C saved at: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model_c()