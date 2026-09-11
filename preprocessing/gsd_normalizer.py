import cv2
import numpy as np

def normalize_gsd_scale(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Downsamples the higher resolution image using anti-aliased Gaussian filtering
    so both source and target images match effective Ground Sample Distance (GSD).
    """
    if np.isclose(scale_factor, 1.0, atol=0.05):
        return image
    if scale_factor < 1.0:
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)
        # Apply Gaussian blur before decimation to suppress high-frequency aliasing
        blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=0.8 / scale_factor)
        return cv2.resize(blurred, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image
