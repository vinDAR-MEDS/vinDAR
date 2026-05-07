import geopandas as gpd
from shapely.geometry import box
import pandas as pd
import numpy as np

def lidar_file_urls(parcels, zones):
    """Return sorted, deduped LiDAR tile URLs covering `parcels`.

    Both inputs must be in EPSG:2154. `zones` must have a `title` column.
    """
    LIDAR_URL = (
        "https://data.geopf.fr/telechargement/download/LiDARHD-NUALID/"
        "{title}/LHD_FXX_{x:04d}_{y:04d}_PTS_LAMB93_IGN69.copc.laz"
    )
    
    TILE_M = 1000  # LiDAR tiles are 1km x 1km in Lambert-93

    pairs = gpd.sjoin(
        parcels[["geometry"]],
        zones[["geometry", "title"]],
        predicate="intersects",
        how="inner",
    )

    urls = set()
    for title, group in pairs.groupby("title"):
        sindex = group.sindex
        minx, miny, maxx, maxy = group.total_bounds
        x_range = range(int(minx // TILE_M), int(-(-maxx // TILE_M)))
        y_range = range(int(miny // TILE_M), int(-(-maxy // TILE_M)))

        for x_km in x_range:
            for y_km in y_range:
                tile = box(
                    x_km * TILE_M, y_km * TILE_M,
                    (x_km + 1) * TILE_M, (y_km + 1) * TILE_M,
                )
                hits = list(sindex.intersection(tile.bounds))
                if not hits or not group.iloc[hits].intersects(tile).any():
                    continue
                urls.add(LIDAR_URL.format(title=title, x=x_km, y=y_km + 1))

    return sorted(urls)