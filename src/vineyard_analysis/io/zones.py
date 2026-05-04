from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

LAMBERT_93 = 2154


def get_zones_data(gdf, tiles_csv):
    assert gdf.crs.to_epsg() == LAMBERT_93, f"CRS must be EPSG:{LAMBERT_93}"

    zones = pd.read_csv(Path(tiles_csv)).dropna(subset=["bbox"])
    zones["geometry"] = zones["bbox"].apply(lambda s: box(*map(float, s.split())))
    zones_gdf = gpd.GeoDataFrame(
        zones, geometry="geometry", crs="EPSG:4326"
    ).to_crs(epsg=LAMBERT_93)

    joined = gpd.sjoin(gdf, zones_gdf, predicate="intersects", how="left")
    return joined.zone.unique()
