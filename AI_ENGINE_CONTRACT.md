# AI Engine → Backend Contract
## What the AI Department Must Implement

> **Status**: All functions below are called directly by the Celery GPU worker (`backend/app/workers/tasks.py`).  
> Every function listed here must have the **exact signature** shown. Return types must match exactly.  
> The backend does **not** care about internal implementation — only inputs and outputs.

---

## Module: `preprocessing/gsd_normalizer.py`

### `build_scale_pyramid(src_img, tgt_img, scale_factor) → (np.ndarray, np.ndarray)`

Resizes the higher-resolution image to match the lower-resolution one using a Gaussian pyramid.

```python
def build_scale_pyramid(
    src_img: np.ndarray,     # Grayscale anchor image (H, W) uint8
    tgt_img: np.ndarray,     # Grayscale target image (H, W) uint8
    scale_factor: float,     # GSD_source / GSD_target ratio
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (src_normalised, tgt_normalised):
    Both images at matching spatial resolution, same dtype (uint8 or float32).
    Anti-aliased Gaussian downsampling must be applied before subsampling.
    """
```

---

## Module: `preprocessing/clahe_enhancer.py`

### `apply_clahe(image) → np.ndarray`

```python
def apply_clahe(
    image: np.ndarray,   # Grayscale (H, W) uint8
) -> np.ndarray:
    """
    Applies 8x8 tile Contrast Limited Adaptive Histogram Equalization.
    Clip limit: 2.5.
    Returns enhanced grayscale image, same shape and dtype as input.
    """
```

---

## Module: `matching/phase_congruency.py`

### `match_rift2(src_img, tgt_img) → (np.ndarray, np.ndarray, np.ndarray)`

```python
def match_rift2(
    src_img: np.ndarray,   # CLAHE-enhanced grayscale (H, W)
    tgt_img: np.ndarray,   # CLAHE-enhanced grayscale (H, W)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RIFT2 Log-Gabor Maximum Index Map (MIM) feature matching.
    Illumination-invariant. Excels on shadowed terrain.

    Returns:
        src_pts    : (N, 2) float32 — matched pixel coords in source image
        tgt_pts    : (N, 2) float32 — matched pixel coords in target image
        confidences: (N,)   float32 — per-match confidence score [0–1]

    N = 0 is valid (no matches found). Never return None.
    """
```

---

## Module: `matching/loftr_matcher.py`

### `LoFTRMatcher.match(image_a, image_b) → (np.ndarray, np.ndarray, np.ndarray)`

```python
class LoFTRMatcher:
    def __init__(self, weights_path: str = None):
        """Load model weights from weights_path on init."""

    def match(
        self,
        image_a: np.ndarray,   # Grayscale (H, W) uint8 or float32
        image_b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Coarse-to-fine LoFTR transformer matching.
        Returns (src_pts, tgt_pts, confidences) — same shape contract as match_rift2.
        """
```

**Weights file**: `/ai_engine/weights/loftr_outdoor.pt`  
If weights file is absent, return empty arrays — do NOT raise an exception.

---

## Module: `matching/roma_matcher.py`

### `RoMaMatcher.match(image_a, image_b) → (np.ndarray, np.ndarray, np.ndarray)`

Same interface as `LoFTRMatcher`.

**Weights file**: `/ai_engine/weights/roma.pt`

---

## Module: `matching/hybrid_merger.py`

### `fuse_hybrid_matches(rift_matches, deep_matches) → (np.ndarray, np.ndarray)`

```python
def fuse_hybrid_matches(
    rift_matches: tuple[np.ndarray, np.ndarray],   # (src_pts, tgt_pts) from RIFT2
    deep_matches: tuple[np.ndarray, np.ndarray],   # (src_pts, tgt_pts) from LoFTR+RoMa stacked
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merges candidates using mutual nearest-neighbor cross-consistency check.
    Returns (merged_src_pts, merged_tgt_pts) — deduplicated, shape (M, 2).
    M = 0 is valid. Never return None.
    """
```

---

## Module: `filtering/magsac_filter.py`

### `filter_magsac(src_pts, tgt_pts) → (np.ndarray, np.ndarray, np.ndarray)`

```python
def filter_magsac(
    src_pts: np.ndarray,   # (N, 2) float32
    tgt_pts: np.ndarray,   # (N, 2) float32
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Grid-partitioned MAGSAC++ geometric outlier rejection.
    Divides image into spatial bins and runs MAGSAC++ per bin.

    Returns:
        filtered_src  : (M, 2) inlier points in source
        filtered_tgt  : (M, 2) inlier points in target
        inlier_mask   : (N,)   bool — True = inlier
    M <= N.
    """
```

---

## Module: `filtering/lucas_kanade.py`

### `refine_subpixel(src_img, tgt_img, src_pts, tgt_pts) → (np.ndarray, np.ndarray, np.ndarray)`

```python
def refine_subpixel(
    src_img: np.ndarray,   # Full-res grayscale source image
    tgt_img: np.ndarray,   # Full-res grayscale target image
    src_pts: np.ndarray,   # (M, 2) inlier source points
    tgt_pts: np.ndarray,   # (M, 2) inlier target points
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pyramidal Lucas-Kanade optical flow refinement.
    Target: residual error < 0.2 pixels.

    Returns:
        refined_src  : (M, 2) float32 — sub-pixel refined source coords
        refined_tgt  : (M, 2) float32 — sub-pixel refined target coords
        residuals    : (M,)   float32 — per-point residual error in pixels
    """
```

---

## Module: `warping/tps_warper.py`

### `warp_tps(tgt_img, src_pts, tgt_pts) → np.ndarray`

```python
def warp_tps(
    tgt_img: np.ndarray,   # Original target image (H, W) uint8
    src_pts: np.ndarray,   # (M, 2) control points in source space
    tgt_pts: np.ndarray,   # (M, 2) corresponding points in target space
) -> np.ndarray:
    """
    Non-rigid Thin-Plate Spline deformation of tgt_img to align with src space.
    Returns warped image, same shape as tgt_img (H, W) uint8.
    """
```

---

## Module: `evaluation/metrics.py`

### `compute_rmse(residuals) → float`
### `compute_sui(src_pts, image_shape) → float`

```python
def compute_rmse(residuals: np.ndarray) -> float:
    """Root Mean Square Error across all sub-pixel residuals. Target < 0.5 px."""

def compute_sui(src_pts: np.ndarray, image_shape: tuple) -> float:
    """
    Spatial Uniformity Index — measures how evenly tie-points are distributed.
    Partitions image into quadrants and computes distribution score.
    Returns float in [0, 1]. Target > 0.85.
    """
```

---

## Module: `evaluation/uncertainty_heatmap.py`

### `build_heatmap(src_pts, residuals, image_shape) → bytes`

```python
def build_heatmap(
    src_pts:     np.ndarray,  # (M, 2) tie-point locations
    residuals:   np.ndarray,  # (M,)   per-point residual errors
    image_shape: tuple,       # (H, W) of the output image
) -> bytes:
    """
    k-NN residual variance spatial interpolation → PNG image.
    Returns PNG-encoded bytes. The frontend overlays this as a heatmap.
    High-residual areas = red, low-residual areas = green/blue.
    """
```

---

## Module: `geospatial/geotiff_exporter.py`

### `export_geotiff(warped_img, reference_tif_path) → bytes`

```python
def export_geotiff(
    warped_img:        np.ndarray,  # (H, W) uint8 warped output image
    reference_tif_path: str,        # Path to source .tif — copy CRS + extent from this
) -> bytes:
    """
    Writes warped_img as a Cloud-Optimized GeoTIFF with:
    - CRS: Moon 2000 (EPSG:30100)
    - Extent/transform: copied from reference_tif_path
    - Compression: LZW
    Returns the GeoTIFF as raw bytes (for upload to MinIO).
    """
```

---

## Module: `geospatial/pds4_exporter.py`

### `export_pds4(job_id, rmse, sui, inlier_count) → str`

```python
def export_pds4(
    job_id:       str,
    rmse:         float,
    sui:          float,
    inlier_count: int,
) -> str:
    """
    Generates a PDS4-compliant XML product label.
    Returns the XML string (NOT bytes — caller encodes to UTF-8).
    """
```

---

## Weights Files Required

Place all model weights in `/ai_engine/weights/` (gitignored):

| File | Source | Used by |
|---|---|---|
| `loftr_outdoor.pt` | [LoFTR GitHub](https://github.com/zju3dv/LoFTR) | `LoFTRMatcher` |
| `roma.pt` | [RoMa GitHub](https://github.com/Parskatt/RoMa) | `RoMaMatcher` |

If a weights file is missing, the matcher must return `(np.empty((0,2)), np.empty((0,2)), np.empty((0,)))` — **no exception**.

---

## General Rules

1. **Never return `None`** — always return empty numpy arrays of the correct shape.
2. **Never raise exceptions** for "no matches found" — return empty arrays.
3. All image arrays are **grayscale (H, W)** unless stated otherwise.
4. All point arrays are **float32, shape (N, 2)** where column 0 = x, column 1 = y.
5. The backend will log a warning if fewer than 4 inlier tie-points are found (not enough for TPS).
