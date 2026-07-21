import matplotlib.pyplot as plt
import numpy as np


def build_plume_mask_percentile(
    plume: np.ndarray,
    base_threshold_ppm_m: float = 0.0,
    lower_percentile: float = 5.0,
    min_pixels: int = 10,
) -> tuple[np.ndarray, float]:
    """Build a plume mask from a base threshold and lower percentile."""
    candidate_mask = np.isfinite(plume) & (plume > base_threshold_ppm_m)
    candidate_values = plume[candidate_mask]

    if candidate_values.size < min_pixels:
        return candidate_mask, float(base_threshold_ppm_m)

    percentile_threshold = float(np.percentile(candidate_values, lower_percentile))
    effective_threshold = max(float(base_threshold_ppm_m), percentile_threshold)
    mask = np.isfinite(plume) & (plume > effective_threshold)

    if np.count_nonzero(mask) < min_pixels:
        return candidate_mask, float(base_threshold_ppm_m)

    return mask, effective_threshold


def plume_mask_area_m2(mask: np.ndarray, px_area_m2: float) -> float:
    if mask is None:
        return 0.0
    return float(np.count_nonzero(mask) * px_area_m2)


def plume_length_sqrt_area_m(
    mask: np.ndarray,
    px_area_m2: float,
    plot: bool = True,
    sat_name: str | None = None,
    datetime_str: str | None = None,
) -> tuple[float, float]:
    """Calculate the characteristic plume length as L = sqrt(A_M)."""
    area_m2 = plume_mask_area_m2(mask, px_area_m2)
    if area_m2 <= 0:
        return 0.0, 0.0

    length_m = float(np.sqrt(area_m2))
    if plot:
        sensor = {"PRS": "PRISMA", "ENMAP": "EnMAP"}.get(sat_name, sat_name)
        plt.figure(figsize=(6, 5))
        plt.imshow(mask, origin="upper")
        plt.title(
            f"{sensor} - {datetime_str}\n"
            f"A_M = {area_m2:.1f} m2; L = sqrt(A_M) = {length_m:.1f} m"
        )
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.tight_layout()
        plt.show()

    return length_m, area_m2
