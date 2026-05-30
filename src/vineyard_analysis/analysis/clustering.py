
import numpy as np
import open3d as o3d
 
 
def cluster_points(
    data,
    eps=0.20,
    min_points=3,
    top_fraction=0.30,
    min_top_points=2,
    min_z_range=0.1,
    centre_band=(0.0, 1.0),
    voxel_size=None,
    spacing=None,
):
    # ── Back-compat: honour the old `spacing` kwarg if present ────────────────
    if spacing is not None:
        eps = float(spacing)
 
    if len(data) == 0:
        return np.empty((0, 3))
 
    points = np.vstack((data.x, data.y, data.z)).T
 
    # ── Optional voxel downsample (cheap density normaliser + speedup) ────────
    if voxel_size is not None and voxel_size > 0:
        pcd_full = o3d.geometry.PointCloud()
        pcd_full.points = o3d.utility.Vector3dVector(points)
        pcd_full = pcd_full.voxel_down_sample(voxel_size=float(voxel_size))
        points = np.asarray(pcd_full.points)
        if len(points) == 0:
            return np.empty((0, 3))
 
    # ── 2D DBSCAN: feed (x, y, 0) so distance is purely horizontal ────────────
    # Open3D's DBSCAN works on Vector3dVector; zeroing Z removes it from the
    # distance metric without forcing us off the o3d backend.
    pts_xy0 = np.column_stack([points[:, 0], points[:, 1],
                               np.zeros(len(points))])
    pcd2d = o3d.geometry.PointCloud()
    pcd2d.points = o3d.utility.Vector3dVector(pts_xy0)
    labels = np.array(pcd2d.cluster_dbscan(eps=float(eps),
                                           min_points=int(min_points)))
 
    if labels.size == 0 or labels.max() < 0:
        return np.empty((0, 3))
 
    refined_centres = []
    n_labels = int(labels.max()) + 1
    for i in range(n_labels):
        cluster_pts = points[labels == i]
        if len(cluster_pts) < min_points:
            continue  # defensive — DBSCAN should already guarantee this
 
        z = cluster_pts[:, 2]
 
        # ── Ground-cluster guard: needs real vertical extent ──────────────────
        z_range = float(z.max() - z.min())
        if z_range < min_z_range:
            continue
 
        # ── Canopy slice (for acceptance + fallback) ──────────────────────────
        n_top = max(min_top_points, int(np.ceil(len(cluster_pts) * top_fraction)))
        n_top = min(n_top, len(cluster_pts))
        top_idx = np.argpartition(z, -n_top)[-n_top:]
        top_pts = cluster_pts[top_idx]
        if len(top_pts) < min_top_points:
            continue
 
        # ── XY centre from the configured Z band ──────────────────────────────
        # Mid-band by default — closer to the trunk than the canopy median.
        lo_q, hi_q = float(centre_band[0]), float(centre_band[1])
        if lo_q >= hi_q or (lo_q <= 0.0 and hi_q >= 1.0):
            band_pts = cluster_pts
        else:
            z_lo = np.quantile(z, lo_q)
            z_hi = np.quantile(z, hi_q)
            mask = (z >= z_lo) & (z <= z_hi)
            band_pts = cluster_pts[mask] if mask.any() else cluster_pts
 
        # Fall back to the canopy slice if the centre band is too sparse to
        # give a stable median (e.g. very small cluster).
        if len(band_pts) < min_top_points:
            band_pts = top_pts
 
        centre_xy = np.median(band_pts[:, :2], axis=0)
        mean_z = float(top_pts[:, 2].mean())
 
        refined_centres.append(np.array([centre_xy[0], centre_xy[1], mean_z]))
 
    if not refined_centres:
        return np.empty((0, 3))
    return np.array(refined_centres)
 