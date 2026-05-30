"""Row-orientation estimation for vineyard centroids.
 
Returns the orientation of either the row OR the plant direction — the FFT
can't reliably distinguish them since they're perpendicular and both
periodic. The caller is expected to try both `angle` and `angle + 90` and
pick the better one by grid-fit score.
 
A DC-disk mask suppresses the parcel envelope, which would otherwise
collapse the arctan2 result to exact 0 / ±90 degrees.
"""
import numpy as np
 
 
def find_row_orientation(data, bins=256, min_snr=2.5):
    """Return angle_deg of the brightest non-DC spectral peak.
 
    The angle is in (-90, 90] and corresponds to either the row direction
    or the plant direction. Returns None when the spectrum is too noisy
    to trust (peak SNR below `min_snr`); callers should skip the parcel.
    """
    x = data["x"].to_numpy()
    y = data["y"].to_numpy()
    if len(x) < 8:
        return None
 
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    if x_range <= 0 or y_range <= 0:
        return None
 
    # Square-pixel histogram: same physical bin size in x and y, so spectral
    # coordinates are isotropic and the peak angle is meaningful.
    if x_range >= y_range:
        bins_x = bins
        bins_y = max(8, int(round(bins * y_range / x_range)))
    else:
        bins_y = bins
        bins_x = max(8, int(round(bins * x_range / y_range)))
 
    hist, _, _ = np.histogram2d(x, y, bins=(bins_x, bins_y))
    hist = hist.T
 
    # Window to reduce edge leakage from the parcel boundary.
    wy = np.hanning(hist.shape[0])[:, None]
    wx = np.hanning(hist.shape[1])[None, :]
    hist_w = hist * (wy * wx)
 
    F = np.fft.fftshift(np.fft.fft2(hist_w))
    magnitude = np.abs(F)
    cy, cx = np.array(magnitude.shape) // 2
 
    # Zero out the DC component and a small disk around it. These low
    # frequencies represent the parcel envelope, not vine periodicity.
    yy, xx = np.indices(magnitude.shape)
    dc_mask = np.hypot(yy - cy, xx - cx) >= 3
    masked = np.where(dc_mask, magnitude, 0)
 
    peak_y, peak_x = np.unravel_index(np.argmax(masked), masked.shape)
    peak_val = masked[peak_y, peak_x]
 
    # SNR check: if the peak doesn't stand out from the background, the
    # spectrum is dominated by noise and the argmax is effectively random.
    bg = np.median(magnitude[dc_mask])
    if peak_val / max(bg, 1e-12) < min_snr:
        return None
 
    # The peak vector in spectral space points along the periodicity
    # direction; the perpendicular real-space direction is the row OR
    # plant axis. (Square-pixel binning means we can read the angle
    # straight off the pixel offsets without rescaling by dx/dy.)
    peak_angle = np.degrees(np.arctan2(peak_y - cy, peak_x - cx))
    line_angle = ((peak_angle + 90 + 90) % 180) - 90
 
    return float(line_angle)