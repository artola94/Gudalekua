import pandas as pd
import numpy as np
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 1. Load datasets
df_main = pd.read_csv(os.path.join(DATA_DIR, 'dataset_preprocesado.csv'))
df_cities = pd.read_csv(os.path.join(DATA_DIR, 'ukranian_cities_final.csv'))
df_losses = pd.read_csv(os.path.join(DATA_DIR, 'daily_losses_processed.csv'))
df_missiles = pd.read_csv(os.path.join(DATA_DIR, 'missiles_cleaned_daily.csv'))

# 2. Date Standardization
for df in [df_main, df_losses, df_missiles]:
    df['date'] = pd.to_datetime(df['date'])

# ---------------------------------------------------------
# KEY STEP: LAG GENERATION (Temporal Delays)
# ---------------------------------------------------------
# We calculate lags BEFORE the merge for efficiency.
# Chosen lags: 
# - 1 day (immediate)
# - 3 days (your map correction)
# - 7 days (weekly trend)

# A. Lags for Losses
# We select key columns to avoid dimensionality explosion
cols_losses_lag = ['ru_total', 'ru_tanks', 'ru_artillery', 'ru_armor']
df_losses_lags = df_losses.copy()

for lag in [1, 3, 7]:
    for col in cols_losses_lag:
        if col in df_losses.columns: # Safety check
            df_losses_lags[f'{col}_lag{lag}'] = df_losses_lags[col].shift(lag)

# B. Lags for Missiles
cols_missiles_lag = ['ru_missiles_launched', 'ru_missiles_num_hit_location']
df_missiles_lags = df_missiles.copy()

for lag in [1, 3, 7]:
    for col in cols_missiles_lag:
        if col in df_missiles.columns:
            df_missiles_lags[f'{col}_lag{lag}'] = df_missiles_lags[col].shift(lag)

# ---------------------------------------------------------
# MERGE PROCESS (CASCADE)
# ---------------------------------------------------------

# 1. Base + Cities (Static)
df_master = pd.merge(df_main, df_cities, on='city_id', how='left')

# Redundancy cleanup
if 'city_name' in df_master.columns:
    df_master = df_master.drop(columns=['city_name'])

# 2. Base + Losses with Lags (Dynamic by Date)
df_master = pd.merge(df_master, df_losses_lags, on='date', how='left')

# 3. Base + Missiles with Lags (Dynamic by Date)
df_master = pd.merge(df_master, df_missiles_lags, on='date', how='left')

# ---------------------------------------------------------
# POST-PROCESSING AND QUALITATIVE LAYER
# ---------------------------------------------------------

# 1. Fill Nulls (Assumption: Missing data = 0 activity)
# We identify new numeric columns from the merges
cols_to_fill = [c for c in df_master.columns if 'ru_' in c or 'uk_' in c or 'missiles' in c]
df_master[cols_to_fill] = df_master[cols_to_fill].fillna(0)

# 2. Qualitative Layer: "Strategic Priority Score"
# Helps the model identify logical hot spots
def calculate_strategic_priority(row):
    # If less than 20km from front AND has high symbolic value (>7)
    if row['dist_to_front_m'] < 20000 and row['symbolic_weight'] >= 8:
        return 3 # CRITICAL
    # If in artillery zone (<30km)
    elif row['dist_to_front_m'] < 30000:
        return 2 # HIGH RISK
    # If in operational range (<50km)
    elif row['dist_to_front_m'] < 50000:
        return 1 # ACTIVE FRONT
    else:
        return 0 # REAR

# Apply the function (make sure there are no NaNs in dist_to_front or symbolic_weight before)
df_master['strategic_priority_score'] = df_master.apply(calculate_strategic_priority, axis=1)

# Save
df_master.to_csv(os.path.join(DATA_DIR, 'dataset.csv'), index=False)

print("Dataset generated successfully.")
print(f"Dimensions: {df_master.shape}")
print(df_master[['date', 'name', 'ru_total_lag3', 'strategic_priority_score']].head())