import numpy as np
import open3d as o3d

def cluster_points(data, spacing=0.75, min_points=5):
    if len(data) == 0:
        return np.empty((0, 3))

    points = np.vstack((data.x, data.y, data.z)).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    labels = np.array(pcd.cluster_dbscan(eps=float(spacing), min_points=min_points))

    if labels.size == 0 or labels.max() < 0:
        return np.empty((0, 3))

    pts = np.asarray(pcd.points)
    centroids = []
    for i in range(labels.max() + 1):
        cluster_pts = pts[labels == i]
        centroids.append(cluster_pts.mean(axis=0))

    return np.array(centroids)