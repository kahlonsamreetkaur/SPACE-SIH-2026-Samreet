import numpy as np

def compute_rmse(pts_src: np.ndarray, pts_warped: np.ndarray) -> float:
    """Computes Root Mean Square Error (RMSE) between tie points."""
    if len(pts_src) == 0:
        return 0.0
    residuals = np.linalg.norm(pts_src - pts_warped, axis=1)
    return float(np.sqrt(np.mean(residuals ** 2)))

def compute_spatial_uniformity_index(pts: np.ndarray, image_shape: tuple, grid_size: tuple = (4, 4)) -> float:
    """
    Computes the Spatial Uniformity Index (SUI) across image bins.
    Score ranges from 0.0 (concentrated in one spot) to 1.0 (perfectly uniform).
    """
    if len(pts) == 0:
        return 0.0
    h, w = image_shape[:2]
    ny, nx = grid_size
    counts, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=[ny, nx], range=[[0, h], [0, w]])
    non_empty = np.count_nonzero(counts)
    return float(non_empty / (nx * ny))
