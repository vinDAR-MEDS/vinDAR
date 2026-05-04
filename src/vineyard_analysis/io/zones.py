import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from vineyard_analysis.config import DATA_DIR, TILES_FILE, TARGET_CRS, PRIORITY_MAP

def load_zones():
    zones = pd.read_csv(os.path.join(DATA_DIR, TILES_FILE)).dropna(subset=["bbox"])
    zones["geometry"] = zones["bbox"].apply(lambda s: box(*map(float, s.split())))
    zones = gpd.GeoDataFrame(zones, geometry="geometry", crs="EPSG:4326").to_crs(TARGET_CRS)
    return zones
