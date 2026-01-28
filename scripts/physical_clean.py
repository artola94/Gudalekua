import pandas as pd
import os
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(BASE_DIR, "data", "ukranian_cities_physical.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "ukranian_cities_physical_clean.csv")

def merge_group(group):
    """
    Receives all rows with the same name (e.g.: 'Kyiv').
    Merges info from NODE to RELATION.
    """
    # If there's only one row, return it as is (whether node or relation)
    if len(group) == 1:
        return group.iloc[0]
    
    # 1. Identify the "Boss" (Target): We prefer Relation (Polygon)
    # Look for rows whose ID contains 'relation'
    boss_candidates = group[group['city_id'].astype(str).str.contains('relation', case=False, na=False)]
    
    if not boss_candidates.empty:
        # If there are relations, the boss is the one with largest area
        boss = boss_candidates.sort_values('geo_area_km2', ascending=False).iloc[0].copy()
    else:
        # If there are no relations, the boss is the node with largest population (or the first one)
        boss = group.sort_values('population', ascending=False).iloc[0].copy()
    
    # 2. Identify "Donors" (All others)
    # We want to rescue the population if the boss doesn't have it
    if pd.isna(boss['population']) or boss['population'] == 0:
        # Look for maximum population in the whole group
        max_pop = group['population'].max()
        if pd.notna(max_pop) and max_pop > 0:
            boss['population'] = max_pop
            
    # 3. Rescue the name source (for curiosity)
    if pd.isna(boss.get('_name_source')) or boss.get('_name_source') == 'Unknown':
        sources = group['_name_source'].unique()
        valid_sources = [f for f in sources if f != 'Unknown' and pd.notna(f)]
        if valid_sources:
            boss['_name_source'] = valid_sources[0]

    return boss

def main():
    print(f"🧹 Loading dirty physical file: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"   Total initial rows: {len(df)}")
    
    # 1. Group by NAME
    # Warning: This will merge different cities with the same name (e.g.: 2 Andriivkas).
    # For your manual project this is acceptable.
    print("🧛 Starting 'vampire' merge (Nodes -> Relations)...")
    
    tqdm.pandas()
    # group_keys=False so it doesn't create an annoying multi-index
    df_clean = df.groupby('city_name', group_keys=False).progress_apply(merge_group)
    
    # Reset index just in case
    df_clean = df_clean.reset_index(drop=True)
    
    print(f"   Rows after merge: {len(df_clean)}")
    print(f"   Absorbed duplicates: {len(df) - len(df_clean)}")
    
    # 2. ADD MANUAL COLUMNS (Empty for you to fill)
    print("📝 Adding columns for manual editing...")
    extra_columns = ['terrain_score', 'is_transport_hub', 'fortification_level']
    
    for col in extra_columns:
        if col not in df_clean.columns:
            df_clean[col] = 0 # Initialize to 0
            
    # 3. SORT TO FACILITATE YOUR WORK
    # Put large cities at top so you fill those first
    df_clean = df_clean.sort_values('population', ascending=False)
    
    # Reorder columns to make Excel editing comfortable
    ordered_cols = ['city_id', 'city_name', 'population', 'geo_area_km2', 
                      'terrain_score', 'is_transport_hub', 'fortification_level', 
                      '_name_source']
    
    # Make sure all columns exist
    final_cols = [c for c in ordered_cols if c in df_clean.columns]
    df_clean = df_clean[final_cols]
    
    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ DONE! File ready: {OUTPUT_CSV}")
    print("   -> Now open it in Excel.")
    print("   -> You'll see that 'relation/XXX' now has the population that the node previously had.")
    print("   -> Fill in the terrain/transport/fortification columns for key cities.")

if __name__ == "__main__":
    main()