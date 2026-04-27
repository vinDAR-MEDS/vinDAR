import numpy as np
import open3d as o3d

def cluster_points(data, spacing = 0.75, min_points = 5):
    points = np.vstack((data.x, data.y, data.z)).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Cluster
    labels = np.array(pcd.cluster_dbscan(eps= spacing,
                                         min_points=min_points))

    pts = np.asarray(pcd.points)
    centroids = []
    
    # Isolate centroids
    for i in range(labels.max() + 1):
        cluster_pts = pts[labels == i]
        centroid = cluster_pts.mean(axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)
    
    return(centroids)