import pandas as pd
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 1. Load the dataset
input_file = os.path.join(DATA_DIR, 'missile_attacks_daily.csv')
output_file = os.path.join(DATA_DIR, 'missiles_cleaned_daily.csv')

try:
    df = pd.read_csv(input_file)
    print("✅ File loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: File {input_file} not found")
    exit()

# 2. Convert 'time_start' column to datetime
# FIX: Added format='mixed' to accept dates with and without time
# dayfirst=False is the standard for YYYY-MM-DD, we add it for safety.
df['date'] = pd.to_datetime(df['time_start'], format='mixed', dayfirst=False).dt.date

# 3. Select only necessary columns
cols_to_keep = ['date', 'launched', 'destroyed', 'num_hit_location']
# We verify columns exist before selecting to avoid another error
available_cols = [c for c in cols_to_keep if c in df.columns]
df_clean = df[available_cols].copy()

# 4. Cleaning null data (NaN)
df_clean = df_clean.fillna(0)

# 5. Group by day
df_grouped = df_clean.groupby('date')[['launched', 'destroyed', 'num_hit_location']].sum().reset_index()

# 6. Rename columns
new_names = {
    'launched': 'ru_missiles_launched',
    'destroyed': 'ru_missiles_destroyed',
    'num_hit_location': 'ru_missiles_num_hit_location'
}
df_grouped.rename(columns=new_names, inplace=True)

# 7. Save the new CSV
df_grouped.to_csv(output_file, index=False)

print(f"✅ Process completed. File saved as: {output_file}")
print("\n--- Preview of first 5 rows ---")
print(df_grouped.head())