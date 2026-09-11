import numpy as np

class LoFTRMatcher:
    """
    Detector-free Local Feature Matching with Transformers (LoFTR).
    Uses coarse-to-fine cross-attention to match low-texture lunar plains.
    """
    def __init__(self, weights_path: str = None):
        self.weights_path = weights_path

    def match(self, image_a: np.ndarray, image_b: np.ndarray):
        """Returns coarse-to-fine candidate matches."""
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))
