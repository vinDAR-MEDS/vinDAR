def find_row_orientation(data, bins=256):
    """Return (line_angle_deg, row_spacing) for a points dataframe with x, y columns."""
    hist, xedges, yedges = np.histogram2d(data["x"], data["y"], bins=bins)
    hist = hist.T

    F = np.fft.fftshift(np.fft.fft2(hist))
    magnitude = np.abs(F)

    cy, cx = np.array(magnitude.shape) // 2
    magnitude[cy, cx] = 0  # zero out DC component

    peak_y, peak_x = np.unravel_index(np.argmax(magnitude), magnitude.shape)

    dx = (xedges[-1] - xedges[0]) / bins
    dy = (yedges[-1] - yedges[0]) / bins
    fy = (peak_y - cy) / (bins * dy)
    fx = (peak_x - cx) / (bins * dx)

    line_angle = np.degrees(np.arctan2(fy, fx)) + 90  # rows are perpendicular to peak
    spacing = 1.0 / np.sqrt(fx**2 + fy**2)
    return line_angle, spacing