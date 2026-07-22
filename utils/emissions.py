import numpy as np
import rasterio

from utils.plume import build_plume_mask_percentile, plume_length_sqrt_area_m
from utils.raster import pixel_metrics_from_dataset


def calc_ime_kg(plume_ppm_m: np.ndarray, px_area_m2: float) -> float:
    """Calculate integrated mass enhancement in kg CH4."""
    molar_volume_l_per_mol = 22.4
    molar_mass_ch4_kg_per_mol = 0.016043
    kg = (
        plume_ppm_m
        * px_area_m2
        * 1e-6
        * 1000.0
        * (1.0 / molar_volume_l_per_mol)
        * molar_mass_ch4_kg_per_mol
    )
    return float(np.nansum(kg))


def effective_wind_speed_prisma_enmap(u10_mps: float) -> float:
    """
    For PRISMA and EnMAP:
    Reference url:
    1. Guanter et al., 2021  https://www.sciencedirect.com/science/article/pii/S0034425721003916
    2. Rogers et al., 2024  https://ieeexplore.ieee.org/abstract/document/10387469
        Ueff = 0.34 * U10 + 0.44
    """
    if u10_mps is None or not np.isfinite(u10_mps):
        return np.nan

    ueff = 0.34 * float(u10_mps) + 0.44
    return float(max(ueff, 0.0))


def effective_wind_speed_ghgsat(
    u10_mps: float,
    min_u10_mps: float = 0.1,
) -> float:
    """
    GHGSat:
    Reference url:
    Varon et al., 2018 - https://amt.copernicus.org/articles/11/5673/2018/
        Ueff = 0.9 * ln(U10) + 0.6
    """
    if u10_mps is None or not np.isfinite(u10_mps):
        return np.nan

    u10_safe = max(float(u10_mps), min_u10_mps)
    ueff = 0.9 * np.log(u10_safe) + 0.6
    return float(max(ueff, 0.0))


def effective_wind_speed_by_sensor(
    u10_mps: float,
    sat_name: str,
    min_u10_mps: float = 0.1,
) -> tuple[float, str]:
    sat_upper = "" if sat_name is None else str(sat_name).upper()

    if sat_upper in ["PRS", "PRISMA", "ENMAP"]:
        return (
            effective_wind_speed_prisma_enmap(u10_mps),
            "Ueff = 0.34 * U10 + 0.44",
        )

    if "GHGSAT" in sat_upper:
        return (
            effective_wind_speed_ghgsat(u10_mps, min_u10_mps),
            "Ueff = 0.9 * ln(U10) + 0.6",
        )

    raise ValueError(
        f"Sensor not recognized for Ueff calculation: {sat_name}. "
        "PRS/PRISMA, ENMAP or GHGSat."
    )


def emission_rate_from_plume_tif(
    tif_path: str,
    u10_mps: float,
    plume_threshold_ppm_m: float = 0.0,
    percentile_filter: float = 0.0,
    nodata_to_nan: bool = True,
    px_area_def: float | None = None,
    sat_name: str | None = None,
    datetime_str: str | None = None,
    min_u10_mps: float = 0.1,
    plot_mask: bool = True,
) -> dict:
    with rasterio.open(tif_path) as src:
        plume = src.read(1).astype(np.float64) * 8
        if nodata_to_nan and src.nodata is not None:
            plume[plume == src.nodata] = np.nan

        px_area_src, _, _ = pixel_metrics_from_dataset(src)
        px_area_m2 = px_area_src if px_area_def is None else float(px_area_def)

    mask, _ = build_plume_mask_percentile(
        plume=plume,
        base_threshold_ppm_m=plume_threshold_ppm_m,
        lower_percentile=percentile_filter,
        min_pixels=10,
    )
    n_plume_pixels = int(np.count_nonzero(mask))
    plume_masked = np.where(mask, plume, np.nan)
    ime_kg = calc_ime_kg(plume_masked, px_area_m2)

    length_m, area_m2 = plume_length_sqrt_area_m(
        mask=mask,
        px_area_m2=px_area_m2,
        plot=plot_mask,
        sat_name=sat_name,
        datetime_str=datetime_str,
    )
    ueff_mps, ueff_method = effective_wind_speed_by_sensor(
        u10_mps=u10_mps,
        sat_name=sat_name,
        min_u10_mps=min_u10_mps,
    )

    if length_m <= 0 or not np.isfinite(ueff_mps):
        q_kg_s = np.nan
        q_kg_h = np.nan
    else:
        q_kg_s = ime_kg * ueff_mps / length_m
        q_kg_h = q_kg_s * 3600.0

    return {
        "IME_kg": ime_kg,
        "L_m": length_m,
        "A_m2": area_m2,
        "U10_mps": float(u10_mps),
        "Ueff_mps": ueff_mps,
        "Ueff_method": ueff_method,
        "Q_kg_s": q_kg_s,
        "Q_kg_h": q_kg_h,
        "n_plume_pixels": n_plume_pixels,
    }
