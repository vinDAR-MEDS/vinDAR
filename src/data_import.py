import geopandas as gpd
import os

def load_shapefile(file_name, idu):
    # Check if file is in correct format
    if ".shp" not in file_name:
        raise ValueError("Incompatible file type. File must be in .shp format") 
    # Read in data
    data = gpd.read_file(os.path.join("..", "data", file_name))
    
    # Check if the input IDU is in the file provided
    if ~data.IDU.isin([idu]).any():
        raise ValueError("Incompatible file type. File must be in .shp format") 

    # Filter data to IDU
    parcels = data[data.IDU == idu]
    
    # Transform parcels to match LiDAR CRS (LAMB93 = EPSG:2154)
    parcels = parcels.to_crs(epsg=2154)
    
    return(parcels)