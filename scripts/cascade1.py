import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import os

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_to_train.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "model_A_front_dynamics.pkl")

# Define the features (Inputs)
# Exclude future targets and non-numeric metadata
FEATURES = [
    # --- Current State ---
    'dist_to_front_m', 'encirclement_score', 'pct_occupied',
    
    # --- Geography and Defense ---
    'terrain_score', 'fortification_level', 'is_transport_hub', 
    'population', 'geo_area_km2', 'symbolic_weight',
    
    # --- Inertia (CRUCIAL for your scenarios) ---
    'delta_dist_daily', 'momentum_7d', 'momentum_30d',
    
    # --- Global Attrition (Global Features) ---
    'ru_total_lag1', 'uk_total', 'ratio_total', 
    'ru_artillery_lag1', 'uk_artillery',
    'ru_missiles_launched_lag1',
    
    # --- Specific Lags (Recent History) ---
    'ru_total_lag3', 'ru_total_lag7'
]

TARGET = 'target_delta_dist_next'

def train_model_a():
    print("--- Loading data ---")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    
    # --- TEMPORAL SPLIT (As agreed) ---
    print("Splitting dataset by time...")
    
    # Train: May 2022 -> Dec 2023 (Learn base logic)
    train_mask = (df['date'] < '2024-01-01')
    
    # Val: Jan 2024 -> Jun 2024 (Fine-tune with current war/Avdiivka)
    val_mask = (df['date'] >= '2024-01-01') & (df['date'] < '2024-07-01')
    
    # Test: Jul 2024 -> End (Trial by fire)
    test_mask = (df['date'] >= '2024-07-01')

    X_train = df.loc[train_mask, FEATURES]
    y_train = df.loc[train_mask, TARGET]
    w_train = df.loc[train_mask, 'sample_weight'] # Your weights go here!

    X_val = df.loc[val_mask, FEATURES]
    y_val = df.loc[val_mask, TARGET]
    # We don't use weights in validation to see the real "human" error

    X_test = df.loc[test_mask, FEATURES]
    y_test = df.loc[test_mask, TARGET]

    print(f"Train rows: {len(X_train)}, Val rows: {len(X_val)}, Test rows: {len(X_test)}")

    # --- XGBOOST TRAINING ---
    print("\n--- Training Model A (Front Movement Regression) ---")
    
    model = xgb.XGBRegressor(
        n_estimators=1000,        # Maximum number of trees
        learning_rate=0.05,       # Learning speed (lower is more precise)
        max_depth=6,              # Tree depth (complexity)
        early_stopping_rounds=50, # Stop if no improvement in 50 iterations
        n_jobs=-1,                # Use all CPU cores
        random_state=42
    )

    model.fit(
        X_train, y_train,
        sample_weight=w_train,    # Apply laziness penalties
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100               # Print progress every 100 trees
    )

    # --- EVALUATION ---
    print("\n--- Results Evaluation ---")
    preds = model.predict(X_test)

    # Calculate RMSE manually for compatibility with modern scikit-learn
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)  # Square root of MSE is RMSE
    
    mae = mean_absolute_error(y_test, preds)
    
    print(f"MAE (Mean Absolute Error): {mae:.2f} meters")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f} meters")
    
    # Basic interpretation
    print("\nWhat does this mean?")
    print(f"On average, the model is off by {mae:.2f} meters when predicting daily advance.")

    # --- FEATURE IMPORTANCE ---
    print("\n--- What is the model learning? (Top 10 Factors) ---")
    importance = pd.DataFrame({
        'feature': FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(importance.head(10))
    
    # --- SAFE SAVE (FIX) ---
    # We use joblib instead of save_model to avoid type conflicts
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"\nModel successfully saved as Python object at: {MODEL_SAVE_PATH}")
    
    return model, X_test, y_test, preds

# --- RUN ---
if __name__ == "__main__":
    model, X_test, y_test, preds = train_model_a()

    # Small visualization of a real vs predicted case (Optional)
    # We take the first 50 days of the test set from any city
    plt.figure(figsize=(10,5))
    plt.plot(y_test.values[:100], label='Reality (Movement Meters)', alpha=0.7)
    plt.plot(preds[:100], label='Model Prediction', alpha=0.7)
    plt.legend()
    plt.title("Reality vs Prediction (First 100 samples from Test)")
    plt.show()