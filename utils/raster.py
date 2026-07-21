import numpy as np
import rasterio
from rasterio.transform import Affine


def meters_per_degree_lat(phi_rad: float) -> float:
    return (
        111132.92
        - 559.82 * np.cos(2 * phi_rad)
        + 1.175 * np.cos(4 * phi_rad)
        - 0.0023 * np.cos(6 * phi_rad)
    )


def meters_per_degree_lon(phi_rad: float) -> float:
    return (
        111412.84 * np.cos(phi_rad)
        - 93.5 * np.cos(3 * phi_rad)
        + 0.118 * np.cos(5 * phi_rad)
    )


def pixel_metrics_from_dataset(
    src: rasterio.io.DatasetReader,
) -> tuple[float, float, float]:
    """Return pixel area and resolution in metres."""
    transform: Affine = src.transform
    resx = abs(transform.a)
    resy = abs(transform.e)

    crs = src.crs
    if crs is None:
        raise ValueError("Missing CRS: pixel size cannot be estimated.")

    if not crs.is_geographic:
        return float(resx * resy), float(resx), float(resy)

    center_row = src.height / 2.0
    center_col = src.width / 2.0
    _, lat_deg = transform * (center_col, center_row)
    phi = np.deg2rad(lat_deg)

    px_resx_m = resx * meters_per_degree_lon(phi)
    px_resy_m = resy * meters_per_degree_lat(phi)
    px_area_m2 = float(px_resx_m * px_resy_m)
    return px_area_m2, float(px_resx_m), float(px_resy_m)
