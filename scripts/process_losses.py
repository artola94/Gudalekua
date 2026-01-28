import pandas as pd
import glob
import os
import numpy as np

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_LOSSES_FOLDER = os.path.join(BASE_DIR, "data", "Russia-Ukraine-main", "data", "byType")
OUTPUT_LOSSES_CSV = os.path.join(BASE_DIR, "data", "daily_losses_processed.csv")

# 1. TRANSLATION DICTIONARY 
# Only what's here (or derived aggregations) will survive in the final CSV
TYPE_MAPPING = {
    # Tanks and Armored Vehicles
    'Tanks': 'tanks',
    'Armoured Fighting Vehicles': 'afv',
    'Infantry Fighting Vehicles': 'ifv',
    'Armoured Personnel Carriers': 'apc',
    'Mine-resistant ambush protected': 'mrap',
    'Infantry Mobility Vehicles': 'imv',
    'Engineering Vehicles': 'engineering',
    # Artillery
    'Self-Propelled Artillery': 'sp_artillery',
    'Towed Artillery': 'towed_artillery',
    'Multiple Rocket Launchers': 'mlrs',
    'Surface-to-air missile systems': 'sam', # Sometimes artillery, sometimes air defense
    'Anti-Aircraft Guns': 'aa_guns',
    'Self-Propelled Anti-Aircraft Guns': 'sp_aa_guns',
    # Aerial
    'Aircraft': 'aircraft',
    'Helicopters': 'helicopters',
    'Unmanned Aerial Vehicles': 'drones',
    'Combat Unmanned Aerial Vehicles': 'combat_drones',
    'Reconnaissance Unmanned Aerial Vehicles': 'recon_drones',
    # Naval
    'Naval Ships': 'ships',
    'Submarines': 'submarines',
    # Logistics and Others
    'Trucks, Vehicles and Jeeps': 'trucks',
    'Radars': 'radars',
    'Communications Stations': 'comms',
    'Command PostsAnd Communications Stations': 'command_posts', # Common typos in source
    'All Types': 'total'
}

# 2. GROUP DEFINITIONS (Aggregations)
AGGREGATIONS = {
    'artillery': ['sp_artillery', 'towed_artillery', 'mlrs'],
    'armor': ['tanks', 'afv', 'ifv', 'apc', 'mrap'],
    'air_defense': ['sam', 'aa_guns', 'sp_aa_guns'],
    'air': ['aircraft', 'helicopters']
}

def process_losses():
    print("🚀 Starting losses processing...")
    all_files = glob.glob(os.path.join(PATH_LOSSES_FOLDER, "**/*.csv"), recursive=True)
    
    if not all_files:
        print("❌ No CSV files found.")
        return

    daily_records = []

    # --- PHASE 1: READING AND CLEANING ---
    for file in all_files:
        try:
            df = pd.read_csv(file)
            
            # Get date
            if 'Date' in df.columns:
                date = pd.to_datetime(df['Date'].iloc[0])
            else:
                # Rescue attempt by filename
                filename = os.path.basename(file).split('.')[0]
                date = pd.to_datetime(filename)

            # Clean equipment names
            df['equipment_type'] = df['equipment_type'].astype(str).str.strip()
            
            row_data = {'date': date}
            
            # Pivoting (We only process what's in TYPE_MAPPING)
            for country in ['Russia', 'Ukraine']:
                prefix = 'ru' if country == 'Russia' else 'uk'
                country_data = df[df['country'] == country]
                
                for _, row in country_data.iterrows():
                    eq_type = row['equipment_type']
                    
                    # MAGIC: If not in our dictionary, we ignore it. 
                    # This automatically eliminates junk columns.
                    short_name = TYPE_MAPPING.get(eq_type)
                    
                    if short_name:
                        col_name = f"{prefix}_{short_name}"
                        row_data[col_name] = row['type_total']

            daily_records.append(row_data)
            
        except Exception as e:
            continue # Silently ignore corrupt files

    # Create DataFrame
    df_final = pd.DataFrame(daily_records)
    df_final = df_final.fillna(0)
    df_final = df_final.sort_values('date')

    # --- PHASE 2: AGGREGATIONS (SUMS) ---
    print("⚙️  Calculating aggregations...")
    for category, subcolumns in AGGREGATIONS.items():
        for prefix in ['ru', 'uk']:
            # Find which columns actually exist
            cols_to_sum = [f"{prefix}_{sub}" for sub in subcolumns if f"{prefix}_{sub}" in df_final.columns]
            
            if cols_to_sum:
                df_final[f"{prefix}_{category}"] = df_final[cols_to_sum].sum(axis=1)
            else:
                df_final[f"{prefix}_{category}"] = 0.0

    # --- PHASE 3: STRATEGIC RATIOS ---
    print("📊 Calculating ratios...")
    ratios_to_calc = ['tanks', 'armor', 'artillery', 'total', 'trucks']
    
    for eq in ratios_to_calc:
        ru_col = f"ru_{eq}"
        uk_col = f"uk_{eq}"
        
        if ru_col in df_final.columns and uk_col in df_final.columns:
            # Ratio with smoothing (+1) to avoid division by zero
            df_final[f'ratio_{eq}'] = df_final[ru_col] / (df_final[uk_col] + 1.0)
            df_final[f'ratio_{eq}'] = df_final[f'ratio_{eq}'].round(3)

    # --- PHASE 4: FINAL CLEANUP (THE GOLDEN FILTER) ---
    print("🧹 Cleaning junk columns...")
    
    # Explicit list of allowed columns
    # 1. Date
    valid_cols = ['date']
    
    # 2. Raw columns from mapping (ru_tanks, uk_tanks...)
    valid_suffixes = list(TYPE_MAPPING.values()) + list(AGGREGATIONS.keys())
    for suffix in valid_suffixes:
        valid_cols.append(f"ru_{suffix}")
        valid_cols.append(f"uk_{suffix}")
    
    # 3. Ratios
    for eq in ratios_to_calc:
        valid_cols.append(f"ratio_{eq}")

    # Filter: We only keep columns that exist and are valid
    final_cols = [c for c in valid_cols if c in df_final.columns]
    df_final = df_final[final_cols]

    # Remove date duplicates
    df_final = df_final.drop_duplicates(subset='date', keep='last')

    # Save
    df_final.to_csv(OUTPUT_LOSSES_CSV, index=False)
    print(f"✅ Done! CSV saved at: {OUTPUT_LOSSES_CSV}")
    print(f"   Rows: {len(df_final)}")
    print(f"   Columns: {len(df_final.columns)} (All clean and useful)")

if __name__ == "__main__":
    process_losses()