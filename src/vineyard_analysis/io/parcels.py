import os
import pandas as pd
import geopandas as gpd
from vineyard_analysis.config import DATA_DIR, PARCELS_FILE, TARGET_CRS, PRIORITY_MAP

def load_parcels():
    """Load vineyard parcels and reproject to the target CRS."""
    return gpd.read_file(os.path.join(DATA_DIR, PARCELS_FILE)).to_crs(TARGET_CRS)

def asign_aoc_to_parcels(parcels, aoc):
    joined = gpd.sjoin(parcels, aoc, predicate="intersects", how="inner")

    # Keep only highest designation
    joined['tier'] = joined['Designation_Level'].map(PRIORITY_MAP)
    joined = joined.reset_index()
    highest = joined.groupby('IDU')['tier'].idxmax()
    return joined.loc[highest].reset_index(drop=True)