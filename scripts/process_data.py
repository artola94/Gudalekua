import os
import json
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from tqdm import tqdm
import warnings

# Ignore future version warnings from pandas/geopandas
warnings.filterwarnings('ignore')


# Paths based on the structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Repo root
PATH_CITIES = os.path.join(BASE_DIR, "data", "ukranian_cities.geojson")
DIR_DAILY_MAPS = os.path.join(BASE_DIR, "data", "nzz-maps-master", "data")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "dataset_preprocesado.csv")

# Projections
CRS_ORIGINAL = "EPSG:4326"  # Latitude/Longitude (what your jsons have)
CRS_METRIC = "EPSG:3035"    # ETRS89-extended / LAEA Europe (Real Meters)

# Optimization Parameters
MAX_ANALYSIS_DISTANCE_KM = 550  # If front is more than 500km away, ignore the city that day
ENCIRCLEMENT_RADIUS_KM = 5      # Buffer to calculate if it's surrounded


def load_custom_json(file_path):
    """
    Reads NZZ JSON files that have the structure {"value": {FeatureCollection}}
    and returns a GeoDataFrame.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        # Extract the real geospatial part that's inside "value"
        if "value" in content:
            geojson_data = content["value"]
        else:
            # Just in case the format changes someday and it's a normal geojson
            geojson_data = content
            
        # Create the GeoDataFrame from the features
        if "features" in geojson_data and geojson_data["features"]:
            gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
            gdf.set_crs(CRS_ORIGINAL, inplace=True)
            return gdf
        else:
            return gpd.GeoDataFrame() # Return empty if no geometries
            
    except Exception as e:
        print(f"⚠️ Error reading {os.path.basename(file_path)}: {e}")
        return gpd.GeoDataFrame()



# 3. BOARD PREPARATION (CITIES)

print(f"📍 Loading cities from: {PATH_CITIES}")

def repair_coordinates(geometry):
    """
    Recursive function to ensure all polygon rings are closed.
    """
    geo_type = geometry.get("type")
    coords = geometry.get("coordinates")

    if not coords:
        return geometry

    def close_ring(ring):
        # If first and last points are not equal, add the first at the end
        if len(ring) > 2 and ring[0] != ring[-1]:
            ring.append(ring[0])
        return ring

    if geo_type == "Polygon":
        # A Polygon is a list of rings (exterior and holes)
        geometry["coordinates"] = [close_ring(list(ring)) for ring in coords]
        
    elif geo_type == "MultiPolygon":
        # A MultiPolygon is a list of Polygons
        new_coords = []
        for poly in coords:
            new_poly = [close_ring(list(ring)) for ring in poly]
            new_coords.append(new_poly)
        geometry["coordinates"] = new_coords
        
    return geometry

# 1. Load as pure JSON to avoid GeoPandas crash
try:
    with open(PATH_CITIES, 'r', encoding='utf-8') as f:
        data_raw = json.load(f)

    # 2. Iterate and repair geometries one by one
    valid_features = []
    print("   -> Repairing topology of open polygons...")
    
    if "features" in data_raw:
        for feature in tqdm(data_raw["features"], desc="Validating cities"):
            if "geometry" in feature and feature["geometry"]:
                # Try to repair
                feature["geometry"] = repair_coordinates(feature["geometry"])
                valid_features.append(feature)
    
    # 3. Create GeoDataFrame from clean data in memory
    cities = gpd.GeoDataFrame.from_features(valid_features)
    
    # Assign original projection (we know it's 4326 lat/lon)
    cities.set_crs("EPSG:4326", inplace=True)

except Exception as e:
    print(f"❌ Fatal error loading cities manually: {e}")
    exit()

# --- NORMAL PROCESS CONTINUES ---

# Ensure we use the correct ID (relation/XXXX)
if "@id" in cities.columns:
    col_id = "@id"
elif "id" in cities.columns:
    col_id = "id"
else:
    # Fallback: use index if no ID column found
    cities["city_id_gen"] = cities.index
    col_id = "city_id_gen"

print(f"   -> Using ID column: {col_id}")

# Transform to meters (EPSG:3035) for precise area calculations
print("   -> Reprojecting to metric system (EPSG:3035)...")
cities = cities.to_crs(CRS_METRIC)

# Static pre-calculations (done only once)
# We use extra buffer(0) to fix self-intersections (bowtie geometries) that may remain
cities["geometry"] = cities.geometry.buffer(0)
cities["total_area_m2"] = cities.geometry.area
cities["geometry_encirclement_buffer"] = cities.geometry.buffer(ENCIRCLEMENT_RADIUS_KM * 1000)

print(f"✅ Total cities loaded and repaired: {len(cities)}")

# 4. TEMPORAL PROCESSING

json_files = sorted(glob.glob(os.path.join(DIR_DAILY_MAPS, "*.json")))
print(f"📂 Found {len(json_files)} daily war maps.")

data_for_csv = []

# Progress bar
pbar = tqdm(json_files, desc="Processing days")

for file in pbar:
    # A. Extract date from filename (e.g.: 2022-02-24.json)
    filename = os.path.basename(file)
    date_str = filename.replace(".json", "")
    
    # B. Load Russian Front
    front_gdf = load_custom_json(file)
    
    if front_gdf.empty:
        continue
        
    # Reproject front to meters
    if front_gdf.crs != CRS_METRIC:
        front_gdf = front_gdf.to_crs(CRS_METRIC)
    
    # Unify all Russian polygons into a single geometry (critical optimization)
    # This avoids comparing each city with 500 front pieces.
    try:
        occupied_zone = front_gdf.unary_union
    except:
        # If unary_union fails (sometimes happens with invalid geometries), try to fix it
        front_gdf["geometry"] = front_gdf.geometry.buffer(0)
        occupied_zone = front_gdf.unary_union

    if occupied_zone is None or occupied_zone.is_empty:
        continue

    # C. SPATIAL FILTER "Active Zone"
    # Only include cities within X km of occupied zone to save time with distant cities
    influence_zone = occupied_zone.buffer(MAX_ANALYSIS_DISTANCE_KM * 1000)
    
    # Spatial index for filtering
    # Find which cities intersect with the influence zone
    candidate_idx = cities.sindex.query(influence_zone, predicate='intersects')
    active_cities = cities.iloc[candidate_idx]
    
    # D. CALCULATIONS PER CITY
    batch_data = []
    
    for _, city in active_cities.iterrows():
        city_id = city[col_id]
        city_geom = city.geometry
        
        # 1. Percentage Occupied
        # Real geometric intersection
        intersection = city_geom.intersection(occupied_zone)
        occupied_area = intersection.area
        pct_occupied = occupied_area / city["total_area_m2"]
        
        # Floating decimal cleanup
        pct_occupied = max(0.0, min(1.0, pct_occupied))
        
        # 2. Distance to front
        if pct_occupied > 0.01:
            # If already touched/occupied, distance is 0 (or symbolic negative)
            distance = 0
        else:
            # Minimum euclidean distance in meters
            distance = city_geom.distance(occupied_zone)
            
        # 3. Encirclement Score
        # How much of the surrounding area (5km) is occupied
        buffer_geom = city["geometry_encirclement_buffer"]
        buffer_intersection = buffer_geom.intersection(occupied_zone)
        pct_surrounded = buffer_intersection.area / buffer_geom.area
        pct_surrounded = max(0.0, min(1.0, pct_surrounded))

        # 4. Binary Target (Did it fall?)
        is_captured = 1 if pct_occupied >= 0.95 else 0
        
        batch_data.append({
            "date": date_str,
            "city_id": city_id,
            "name": city.get("name", "Unknown"), # Save name just in case
            "pct_occupied": round(pct_occupied, 4),
            "dist_to_front_m": round(distance, 0),
            "encirclement_score": round(pct_surrounded, 4),
            "is_captured": is_captured
        })
    
    data_for_csv.extend(batch_data)


# 5. FINAL SAVE

print(f"\nGenerating master CSV with {len(data_for_csv)} records...")
df_final = pd.DataFrame(data_for_csv)

# Sort by date and city
df_final = df_final.sort_values(by=["date", "city_id"])

df_final.to_csv(OUTPUT_CSV, index=False)
print(f" Dataset saved at: {OUTPUT_CSV}")
