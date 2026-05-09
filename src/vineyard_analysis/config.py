"""
Configuration for the vineyard analysis pipeline.

Only constants are present here

Edit this file to point at different data,
change the target CRS, or adjust filtering criteria.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join("..", "data")

PARCELS_FILE = "AllVDR.gpkg"     # Parcels / plots
AOC_FILE = "france_aoc.gpkg"             # Geometry for all AOC/PDOs in France
TILES_FILE = "lidar_tiles.csv"           # LiDAR tile geometries
SPACING_FILE = "aoc_regulations.csv"     # AOC spacing regulations

OUTPUT_CSV = "parcel_results.csv"

# ---------------------------------------------------------------------------
# Coordinate reference system
# ---------------------------------------------------------------------------

TARGET_CRS = 2154  # Lambert-93

# ---------------------------------------------------------------------------
# PDO priority ranking
# ---------------------------------------------------------------------------

PRIORITY_MAP = {
    'Grand Cru': 5, 'Grand Cru Monopole': 5, 'Communal Grand Cru': 5,
    'Cru': 4, 'Vin Doux Naturel Cru': 4, 'Sub-regional Cru': 4,
    'Communal': 3, 'Communal Monopole': 3, 'Communal Vin Jaune': 3,
    'VDN Communal': 3, 'Communal Rosé': 3, 'Communal Sparkling': 3,
    'Communal Sweet': 3,
    'Sub-regional': 2, 'VDN Sub-regional': 2, 'Sub-regional Rosé': 2,
    'Sub-regional Sparkling': 2, 'Sub-regional Sweet': 2,
    'Regional': 1, 'Regional Supérieur': 1, 'Crémant': 1,
    'VDN Regional': 1, 'Vin Doux Naturel': 1, 'Vin de Liqueur': 1,
}

# ---------------------------------------------------------------------------
# Filtering criteria
# ---------------------------------------------------------------------------

# Two-digit department code regex (e.g. r"^84"). Set to None to skip.
DEPT_PREFIX = None

# Wine region of interest. Set to None to skip filtering.
WINE_REGION = "Rhône"

# Administrative region of interest. Set to None to skip filtering.
ADMINISTRATIVE_REGION = None

# PDO/AOC of interest. Set to None to skip filtering.
PDO_NAME = None

# Desognation level to filter to. Set to None to skip filtering.
DESIGNATION_LEVEL = None

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
SAMPLE_SIZE = 30

# ---------------------------------------------------------------------------
# Row Fitting Variables
# ---------------------------------------------------------------------------

DEFAULT_ROW_MIN = 1
DEFAULT_ROW_MAX = 2.5
DEFAULT_PLANT_MIN = 0.5
DEFAULT_PLANT_MAX = 2.0
STEP = 0.1
DECIMALS = 1