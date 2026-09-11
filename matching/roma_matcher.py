import numpy as np

class RoMaMatcher:
    """
    Robust Dense Feature Matching with Transformers (RoMa).
    Excels at large baseline angle and perspective variations.
    """
    def __init__(self, weights_path: str = None):
        self.weights_path = weights_path

    def match(self, image_a: np.ndarray, image_b: np.ndarray):
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))
