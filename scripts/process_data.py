import os
import json
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from tqdm import tqdm
import warnings

# Ignorar advertencias de versiones futuras de pandas/geopandas
warnings.filterwarnings('ignore')


# Rutas basadas en la estructura
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Raíz del repo
PATH_CIUDADES = os.path.join(BASE_DIR, "data", "ukranian_cities.geojson")
DIR_MAPAS_DIARIOS = os.path.join(BASE_DIR, "data", "nzz-maps-master", "data")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "dataset_ml_final.csv")

# Proyecciones
CRS_ORIGINAL = "EPSG:4326"  # Latitud/Longitud (lo que tienen tus jsons)
CRS_METRICO = "EPSG:3035"   # ETRS89-extended / LAEA Europe (Metros reales)

# Parámetros de Optimización
DISTANCIA_MAX_ANALISIS_KM = 50  # Si el frente está a más de 50km, ignoramos la ciudad ese día
RADIO_CERCO_KM = 5              # Buffer para calcular si está rodeada


def cargar_json_custom(ruta_archivo):
    """
    Lee los archivos JSON de NZZ que tienen la estructura {"value": {FeatureCollection}}
    y devuelve un GeoDataFrame.
    """
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            contenido = json.load(f)
        
        # Extraemos la parte geoespacial real que está dentro de "value"
        if "value" in contenido:
            geojson_data = contenido["value"]
        else:
            # Por si acaso algún día el formato cambia y es un geojson normal
            geojson_data = contenido
            
        # Creamos el GeoDataFrame desde las features
        if "features" in geojson_data and geojson_data["features"]:
            gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
            gdf.set_crs(CRS_ORIGINAL, inplace=True)
            return gdf
        else:
            return gpd.GeoDataFrame() # Devuelve vacío si no hay geometrías
            
    except Exception as e:
        print(f"⚠️ Error leyendo {os.path.basename(ruta_archivo)}: {e}")
        return gpd.GeoDataFrame()



# 3. PREPARACIÓN DEL TABLERO (CIUDADES)

print(f"📍 Cargando ciudades desde: {PATH_CIUDADES}")

def reparar_coordenadas(geometry):
    """
    Función recursiva para asegurar que todos los anillos de los polígonos estén cerrados.
    """
    tipo = geometry.get("type")
    coords = geometry.get("coordinates")

    if not coords:
        return geometry

    def cerrar_anillo(anillo):
        # Si el primer y último punto no son iguales, añadimos el primero al final
        if len(anillo) > 2 and anillo[0] != anillo[-1]:
            anillo.append(anillo[0])
        return anillo

    if tipo == "Polygon":
        # Un Polígono es una lista de anillos (el exterior y los agujeros)
        geometry["coordinates"] = [cerrar_anillo(list(ring)) for ring in coords]
        
    elif tipo == "MultiPolygon":
        # Un MultiPolígono es una lista de Polígonos
        new_coords = []
        for poly in coords:
            new_poly = [cerrar_anillo(list(ring)) for ring in poly]
            new_coords.append(new_poly)
        geometry["coordinates"] = new_coords
        
    return geometry

# 1. Cargar como JSON puro para evitar el crash de GeoPandas
try:
    with open(PATH_CIUDADES, 'r', encoding='utf-8') as f:
        data_raw = json.load(f)

    # 2. Iterar y reparar geometrías una por una
    features_validas = []
    print("   -> Reparando topología de polígonos abiertos...")
    
    if "features" in data_raw:
        for feature in tqdm(data_raw["features"], desc="Validando ciudades"):
            if "geometry" in feature and feature["geometry"]:
                # Intentamos reparar
                feature["geometry"] = reparar_coordenadas(feature["geometry"])
                features_validas.append(feature)
    
    # 3. Crear el GeoDataFrame desde los datos limpios en memoria
    cities = gpd.GeoDataFrame.from_features(features_validas)
    
    # Asignar proyección original (sabemos que es 4326 lat/lon)
    cities.set_crs("EPSG:4326", inplace=True)

except Exception as e:
    print(f"❌ Error fatal cargando ciudades manualmente: {e}")
    exit()

# --- CONTINÚA EL PROCESO NORMAL ---

# Aseguramos que usamos el ID correcto (relation/XXXX)
if "@id" in cities.columns:
    col_id = "@id"
elif "id" in cities.columns:
    col_id = "id"
else:
    # Fallback: usar el índice si no encuentra columna ID
    cities["city_id_gen"] = cities.index
    col_id = "city_id_gen"

print(f"   -> Usando columna ID: {col_id}")

# Transformar a metros (EPSG:3035) para cálculos de área precisos
print("   -> Reproyectando a sistema métrico (EPSG:3035)...")
cities = cities.to_crs(CRS_METRICO)

# Pre-cálculos estáticos (se hacen una sola vez)
# Usamos buffer(0) extra para arreglar auto-intersecciones (geometrías bowtie) que puedan quedar
cities["geometry"] = cities.geometry.buffer(0)
cities["area_total_m2"] = cities.geometry.area
cities["geometry_buffer_cerco"] = cities.geometry.buffer(RADIO_CERCO_KM * 1000)

print(f"✅ Total ciudades cargadas y reparadas: {len(cities)}")

# 4. PROCESAMIENTO TEMPORAL

archivos_json = sorted(glob.glob(os.path.join(DIR_MAPAS_DIARIOS, "*.json")))
print(f"📂 Encontrados {len(archivos_json)} mapas diarios de la guerra.")

datos_para_csv = []

# Barra de progreso
pbar = tqdm(archivos_json, desc="Procesando días")

for archivo in pbar:
    # A. Extraer fecha del nombre (ej: 2022-02-24.json)
    nombre_archivo = os.path.basename(archivo)
    fecha_str = nombre_archivo.replace(".json", "")
    
    # B. Cargar Frente Ruso
    frente_gdf = cargar_json_custom(archivo)
    
    if frente_gdf.empty:
        continue
        
    # Reproyectar frente a metros
    if frente_gdf.crs != CRS_METRICO:
        frente_gdf = frente_gdf.to_crs(CRS_METRICO)
    
    # Unificar todos los polígonos rusos en una sola geometría (optimización crítica)
    # Esto evita comparar cada ciudad con 500 trozos de frente.
    try:
        zona_ocupada = frente_gdf.unary_union
    except:
        # Si falla unary_union (a veces pasa con geometrías inválidas), intentamos arreglarlo
        frente_gdf["geometry"] = frente_gdf.geometry.buffer(0)
        zona_ocupada = frente_gdf.unary_union

    if zona_ocupada is None or zona_ocupada.is_empty:
        continue

    # C. FILTRO ESPACIAL "Zona Activa"
    # Solo se incluyen ciudades a X km de la zona ocupada para no perder tiempo con ciudades lejanas
    zona_influencia = zona_ocupada.buffer(DISTANCIA_MAX_ANALISIS_KM * 1000)
    
    # Índice espacial para filtrar
    # Buscamos qué ciudades intersectan con la zona de influencia
    idx_candidatas = cities.sindex.query(zona_influencia, predicate='intersects')
    ciudades_activas = cities.iloc[idx_candidatas]
    
    # D. CÁLCULOS POR CIUDAD
    batch_datos = []
    
    for _, ciudad in ciudades_activas.iterrows():
        id_ciudad = ciudad[col_id]
        geom_ciudad = ciudad.geometry
        
        # 1. Porcentaje Ocupado
        # Intersección geométrica real
        interseccion = geom_ciudad.intersection(zona_ocupada)
        area_ocupada = interseccion.area
        pct_ocupado = area_ocupada / ciudad["area_total_m2"]
        
        # Limpieza de decimales flotantes
        pct_ocupado = max(0.0, min(1.0, pct_ocupado))
        
        # 2. Distancia al frente
        if pct_ocupado > 0.01:
            # Si ya está tocada/ocupada, la distancia es 0 (o negativa simbólica)
            distancia = 0
        else:
            # Distancia euclidiana mínima en metros
            distancia = geom_ciudad.distance(zona_ocupada)
            
        # 3. Cerco (Encirclement Score)
        # Cuánto del área alrededor (5km) está ocupada
        geom_buffer = ciudad["geometry_buffer_cerco"]
        interseccion_buffer = geom_buffer.intersection(zona_ocupada)
        pct_rodeado = interseccion_buffer.area / geom_buffer.area
        pct_rodeado = max(0.0, min(1.0, pct_rodeado))

        # 4. Target Binario (¿Cayó?)
        is_captured = 1 if pct_ocupado >= 0.95 else 0
        
        batch_datos.append({
            "date": fecha_str,
            "city_id": id_ciudad,
            "name": ciudad.get("name", "Unknown"), # Guardamos nombre por si acaso
            "pct_occupied": round(pct_ocupado, 4),
            "dist_to_front_m": round(distancia, 0),
            "encirclement_score": round(pct_rodeado, 4),
            "is_captured": is_captured
        })
    
    datos_para_csv.extend(batch_datos)


# 5. GUARDADO FINAL

print(f"\nGenerando CSV maestro con {len(datos_para_csv)} registros...")
df_final = pd.DataFrame(datos_para_csv)

# Ordenar por fecha y ciudad
df_final = df_final.sort_values(by=["date", "city_id"])

df_final.to_csv(OUTPUT_CSV, index=False)
print(f" Dataset guardado en: {OUTPUT_CSV}")
