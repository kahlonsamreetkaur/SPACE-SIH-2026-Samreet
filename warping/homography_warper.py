import cv2
import numpy as np

def warp_homography(image: np.ndarray, H: np.ndarray, output_shape: tuple) -> np.ndarray:
    """Planar homography perspective warp fallback."""
    return cv2.warpPerspective(image, H, output_shape)
