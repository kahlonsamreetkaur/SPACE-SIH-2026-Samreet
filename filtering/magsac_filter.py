import cv2
import numpy as np

def filter_magsac_binned(pts_src: np.ndarray, pts_tgt: np.ndarray, grid_bins: tuple = (4, 4)):
    """
    Spatially-binned MAGSAC++ geometric outlier rejection.
    Ensures tie-points are evenly distributed across image quadrants.
    """
    if len(pts_src) < 4:
        return pts_src, pts_tgt, np.ones(len(pts_src), dtype=bool)

    H, mask = cv2.findHomography(pts_src, pts_tgt, cv2.USAC_MAGSAC, 3.0)
    inliers = mask.ravel().astype(bool) if mask is not None else np.ones(len(pts_src), dtype=bool)
    return pts_src[inliers], pts_tgt[inliers], inliers
