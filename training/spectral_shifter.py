import numpy as np

def simulate_spectral_shift(visible_img: np.ndarray) -> np.ndarray:
    """
    Simulates short-wave infrared (SWIR) response (Chandrayaan-2 IIRS) from visible reflectance.
    """
    return visible_img.astype(np.float32) * 0.85
