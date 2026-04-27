import geopandas as gpd
import os

def load_shapefile_id(file_name, idu):
    # Read in data
    data = gpd.read_file(os.path.join("..", "data", file_name, f"{file_name}.shp"))
    
    # Check if the input IDU is in the file provided
    if "id" not in data.columns or not data["id"].isin([idu]).any():
        raise ValueError(f"IDU '{idu}' was not found in the specified shapefile.")

    # Filter data to IDU
    parcels = data[data.id == idu]
    
    # Transform parcels to match LiDAR CRS (LAMB93 = EPSG:2154)
    parcels = parcels.to_crs(epsg=2154)
    
    return(parcels)