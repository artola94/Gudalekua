import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import joblib
import numpy as np
import os

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_to_train.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "model_B_encirclement.pkl")

# FEATURES FOR MODEL B
# The hypothesis is: Encirclement depends on distance, defense, and artillery/armor pressure
FEATURES = [
    'dist_to_front_m',       # MAIN variable
    'fortification_level',   # High defenses reduce chance of quick encirclement
    'terrain_score',         # Difficult terrain complicates surrounding
    'is_transport_hub',      # Logistics hubs hold longer
    'geo_area_km2',          # Large cities are harder to surround
    'ru_armor_lag1',         # Recent Russian tank pressure
    'uk_artillery',          # Ukrainian response capability
    'delta_dist_daily'       # If the front moves fast today, encirclement advances
]

TARGET = 'target_encirclement_next'

def train_model_b():
    print("--- Loading data for Model B (Encirclement) ---")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    
    # 1. Temporal Split (Identical to Model A for coherence)
    train_mask = (df['date'] < '2024-01-01')
    val_mask = (df['date'] >= '2024-01-01') & (df['date'] < '2024-07-01')
    test_mask = (df['date'] >= '2024-07-01')

    # 2. TACTICAL RELEVANCE FILTER (CRUCIAL)
    # We don't want to learn about encirclements in cities 300km from the front.
    # We filter to train ONLY with cities in the "Operations Zone" (< 20km).
    # This makes the model a close combat specialist.
    relevant_zone = df['dist_to_front_m'] < 20000 
    
    print(f"Total rows: {len(df)}")
    print(f"Rows in combat zone (<20km): {relevant_zone.sum()}")

    # Apply combined masks (Time + Zone)
    X_train = df.loc[train_mask & relevant_zone, FEATURES]
    y_train = df.loc[train_mask & relevant_zone, TARGET]
    w_train = df.loc[train_mask & relevant_zone, 'sample_weight']

    X_val = df.loc[val_mask & relevant_zone, FEATURES]
    y_val = df.loc[val_mask & relevant_zone, TARGET]

    X_test = df.loc[test_mask & relevant_zone, FEATURES]
    y_test = df.loc[test_mask & relevant_zone, TARGET]

    print(f"Training with {len(X_train)} real threat situations...")

    # --- TRAINING ---
    # We use Regressor because encirclement_score is continuous (0.0 to 1.0)
    model = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=5,
        early_stopping_rounds=50,
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
    
    # Clip to ensure prediction is between 0 and 1 (physical logic)
    preds = np.clip(preds, 0, 1)
    
    mae = mean_absolute_error(y_test, preds)
    
    print(f"\nModel B Results:")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print("Interpretation: A MAE of 0.05 means the model fails by 5% when estimating how surrounded a city is.")

    # --- FEATURE IMPORTANCE ---
    print("\n--- Key Factors for Encirclement ---")
    importance = pd.DataFrame({
        'feature': FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance.head(10))

    # --- SAVE ---
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"\nModel B successfully saved at: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model_b()