import os
import pandas as pd
import geopandas as gpd
from vineyard_analysis.config import TARGET_CRS, DATA_DIR, AOC_FILE, SPACING_FILE, WINE_REGION, ADMINISTRATIVE_REGION, PDO_NAME, DESIGNATION_LEVEL

def load_aoc():
    aoc = gpd.read_file(os.path.join(DATA_DIR, AOC_FILE)).to_crs(TARGET_CRS)
    
    return aoc

def load_spacing():
    spacing = pd.read_csv(os.path.join(DATA_DIR, SPACING_FILE))
    
    return spacing

def filter_aoc(aoc, 
               wine_region=WINE_REGION,
               administrative_region=ADMINISTRATIVE_REGION,
               pdonam = PDO_NAME,
               designation_level = DESIGNATION_LEVEL):
    """Filter AOC data by wine region and/or administrative region."""
    if wine_region is not None:
        aoc = aoc[aoc.Wine_Region.str.contains(wine_region)]
    if administrative_region is not None:
        aoc = aoc[aoc.Administrative_Region.str.contains(administrative_region)]
    if designation_level is not None:
        aoc = aoc[aoc.Designation_Level.str.contains(pdonam)]
    if pdonam is not None:
        aoc = aoc[aoc.PDOnam.str.contains(designation_level)]
    return aoc