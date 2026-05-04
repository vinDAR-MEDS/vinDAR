def lidar_tile_urls(parcels, zones):
    parcels = parcels.to_crs(2154)
    zones = zones.to_crs(2154)
    
    # One row per (parcel, zone) pair the parcel touches.
    pairs = gpd.sjoin(
        parcels[["geometry"]],
        zones[["geometry", "title"]],
        predicate="intersects",
        how="inner",
    )
    
    urls = set()
