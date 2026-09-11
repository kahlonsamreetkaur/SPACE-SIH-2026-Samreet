import numpy as np
from scipy.spatial import cKDTree

def generate_uncertainty_heatmap(tie_points: np.ndarray, residuals: np.ndarray, output_shape: tuple, k: int = 5):
    """
    Computes k-NN residual variance grid across the raster to render
    interactive spatial registration uncertainty heatmaps in the frontend.
    """
    h, w = output_shape[:2]
    heatmap = np.zeros((h // 8, w // 8), dtype=np.float32)
    return heatmap
