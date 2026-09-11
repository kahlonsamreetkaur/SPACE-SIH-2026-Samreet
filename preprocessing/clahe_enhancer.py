import cv2
import numpy as np

def apply_lunar_clahe(image: np.ndarray, clip_limit: float = 2.5, grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to attenuate harsh polar crater shadows and boost mare ridge contrast.
    """
    if image.dtype != np.uint8:
        norm_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        norm_img = image

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(norm_img)
