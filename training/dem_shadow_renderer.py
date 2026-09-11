import numpy as np

def render_dem_hillshade(dem_array: np.ndarray, azimuth_deg: float, altitude_deg: float) -> np.ndarray:
    """
    Ray-traced hillshade generator from LOLA DEM at dynamic solar incidence angles (5° to 60°)
    to simulate extreme lunar shadow transitions for training contrast-invariant models.
    """
    azimuth = np.radians(azimuth_deg)
    altitude = np.radians(altitude_deg)
    gy, gx = np.gradient(dem_array)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(gx*gx + gy*gy))
    aspect = np.arctan2(-gx, gy)
    shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    return np.clip(shaded, 0, 1)
