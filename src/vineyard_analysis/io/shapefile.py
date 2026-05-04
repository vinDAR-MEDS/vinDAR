from pathlib import Path

import geopandas as gpd

LAMBERT_93 = 2154


def load_shapefile(file_name, value, data_dir, id_column="IDU"):
    path = Path(data_dir) / file_name / f"{file_name}.shp"
    data = gpd.read_file(path)

    if id_column not in data.columns or not data[id_column].isin([value]).any():
        raise ValueError(
            f"Value {value!r} not found in column {id_column!r} of {path.name}"
        )

    parcels = data[data[id_column] == value]
    return parcels.to_crs(epsg=LAMBERT_93)
