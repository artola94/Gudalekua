import json
import pandas as pd
import geopandas as gpd
import os
from tqdm import tqdm
from shapely.geometry import shape
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_GEOJSON = os.path.join(BASE_DIR, "data", "ukranian_cities.geojson")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "ukranian_cities_physical.csv")
CRS_METRIC = "EPSG:3035"

# ==========================================
# GEOMETRY REPAIR LOGIC
# ==========================================
def repair_geo_json(geometry):
    if not geometry: return None
    coords = geometry.get("coordinates")
    geo_type = geometry.get("type")
    if not coords: return geometry

    def close_ring(ring):
        if len(ring) > 2 and ring[0] != ring[-1]:
            ring.append(ring[0])
        return ring

    if geo_type == "Polygon":
        geometry["coordinates"] = [close_ring(list(r)) for r in coords]
    elif geo_type == "MultiPolygon":
        new_coords = []
        for poly in coords:
            new_poly = [close_ring(list(r)) for r in poly]
            new_coords.append(new_poly)
        geometry["coordinates"] = new_coords
    return geometry

# ==========================================
# NEW LOGIC: DEEP EXTRACTION
# ==========================================
def get_real_properties(props_raw):
    """
    Analyzes the properties dictionary and if it detects the nested structure
    @relations -> reltags, extracts the data from there.
    """
    # 1. Normal Case: Data is at the root
    if "name" in props_raw or "name:en" in props_raw:
        return props_raw
    
    # 2. Nested Case (The one you discovered)
    if "@relations" in props_raw and isinstance(props_raw["@relations"], list):
        relations = props_raw["@relations"]
        if len(relations) > 0:
            # Look inside 'reltags' of the first element
            nested_tags = relations[0].get("reltags", {})
            
            # Merge: Keep the original @id but add the found tags
            # (We make a copy to not destructively alter the original)
            merged_props = props_raw.copy()
            merged_props.update(nested_tags)
            return merged_props
            
    return props_raw

# ==========================================
# MAIN SCRIPT
# ==========================================
def main():
    print(f"📍 Reading raw JSON: {PATH_GEOJSON}")
    try:
        with open(PATH_GEOJSON, 'r', encoding='utf-8') as f:
            data_raw = json.load(f)
    except Exception as e:
        print(f"❌ Fatal error reading file: {e}")
        return

    features = data_raw.get("features", [])
    print(f"   -> Found {len(features)} elements. Processing...")

    processed_data = []
    name_stats = {"name": 0, "name:uk": 0, "name:en": 0, "name:ru": 0, "Unknown": 0}

    for feat in tqdm(features, desc="Extracting data"):
        props_raw = feat.get("properties", {})
        geom_raw = feat.get("geometry")
        
        # --- HERE'S THE NEW MAGIC ---
        # Flatten properties before searching for anything
        props = get_real_properties(props_raw)
        # --------------------------------
        
        # 1. ID (Can be at root or come from props)
        cid = props.get("@id") or props.get("id")
        if not cid: continue
            
        # 2. NAME (Cascade)
        name = None
        name_source = "Unknown"
        keys_to_try = ["name:en", "name", "name:uk", "name:ru"] # Prioritize English to avoid weird characters if you prefer
        
        for k in keys_to_try:
            val = props.get(k)
            if val and isinstance(val, str) and len(val.strip()) > 0:
                name = val
                name_source = k
                break
        
        if not name:
            name = f"Unknown_{cid}"
            name_stats["Unknown"] += 1
        else:
            name_stats[name_source] += 1

        # 3. POPULATION
        pop_raw = props.get("population")
        population = None
        if pop_raw:
            try:
                # Clean "114 867" -> 114867
                population = int(str(pop_raw).replace(" ", ""))
            except:
                population = None 

        # 4. GEOMETRY
        if geom_raw:
            repaired_geom = repair_geo_json(geom_raw)
            geom_obj = shape(repaired_geom)
            
            processed_data.append({
                "city_id": cid,
                "city_name": name,
                "population": population,
                "geometry": geom_obj,
                "_name_source": name_source
            })

    # --- DATAFRAME ---
    print("\n📐 Calculating geometric areas...")
    gdf = gpd.GeoDataFrame(processed_data)
    
    if gdf.empty:
        print("❌ Error: No valid cities were extracted. Check the JSON structure.")
        return

    gdf.set_crs("EPSG:4326", inplace=True)
    gdf_metric = gdf.to_crs(CRS_METRIC)
    
    # Area in km2
    gdf["geo_area_km2"] = (gdf_metric.geometry.area / 1e6).round(3)

    # --- SAVE PHYSICAL CSV ---
    df_final = pd.DataFrame(gdf[[
        "city_id", "city_name", "population", "geo_area_km2", "_name_source"
    ]])
    
    df_final = df_final.sort_values("city_name")
    df_final.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*40)
    print("SUMMARY V3 (NESTED FIXED)")
    print("="*40)
    print(f"✅ Total cities: {len(df_final)}")
    print(f"✅ Cities with detected population: {df_final['population'].notna().sum()}")
    print("Name sources:")
    for k, v in name_stats.items():
        if v > 0: print(f"   - {k}: {v}")
            
    print(f"\n💾 File ready: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()