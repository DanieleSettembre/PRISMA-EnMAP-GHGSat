import os

import numpy as np
import pandas as pd

from utils.comparison import build_detailed_comparison_table, build_paired_tables
from utils.emissions import emission_rate_from_plume_tif
from utils.ghgsat_public import append_ghgsat_public_results
from utils.metadata import (
    event_key,
    plume_directory_from_tif,
    sat_group_from_sat_name,
    sat_name_from_tif,
)
from utils.outputs import write_analysis_outputs
from utils.plotting import create_comparison_plots
from utils.wind import (
    acquisition_midpoint_utc,
    nearest_era5_datetime_utc,
    windspeed,
)


def _error_result(tif_path: str, error: Exception) -> dict:
    sensor = sat_name_from_tif(tif_path)
    return {
        "event": event_key(tif_path),
        "sat": sensor,
        "sat_group": sat_group_from_sat_name(sensor),
        "tif": tif_path,
        "datetime": None,
        "U10 (m/s)": np.nan,
        "Ueff (m/s)": np.nan,
        "Ueff method": None,
        "Mask threshold (ppm m)": np.nan,
        "N plume pixels": np.nan,
        "A plume mask (m2)": np.nan,
        "L sqrt(A) (m)": np.nan,
        "IME (kg)": np.nan,
        "Q (kg/s)": np.nan,
        "Q (kg/h)": np.nan,
        "error": str(error),
    }


def process_plume_tif(
    tif_path: str,
    plume_threshold_ppm_m: float,
    percentile_filter: float,
    plot_mask: bool,
    wind_by_event_hour: dict,
) -> dict:
    print(f"Processing: {tif_path}")
    sensor = sat_name_from_tif(tif_path)
    plume_dir = plume_directory_from_tif(tif_path)

    try:
        midpoint_utc = acquisition_midpoint_utc(tif_path)
        wind_key = (event_key(tif_path), nearest_era5_datetime_utc(midpoint_utc))
        datetime_str = midpoint_utc.strftime("%Y/%m/%d %H:%M:%S")
        if wind_key in wind_by_event_hour:
            wind_speed = wind_by_event_hour[wind_key]
            print(
                "Reusing ERA5 wind for the same event and nearest hour: "
                f"{wind_speed} m/s"
            )
        else:
            wind_speed, datetime_str = windspeed(tif_path, plume_dir)
            wind_by_event_hour[wind_key] = wind_speed
        metrics = emission_rate_from_plume_tif(
            tif_path=tif_path,
            u10_mps=wind_speed,
            plume_threshold_ppm_m=plume_threshold_ppm_m,
            percentile_filter=percentile_filter,
            sat_name=sensor,
            datetime_str=datetime_str,
            px_area_def=None,
            plot_mask=plot_mask,
        )
        result = {
            "event": event_key(tif_path),
            "sat": sensor,
            "sat_group": sat_group_from_sat_name(sensor),
            "tif": tif_path,
            "datetime": datetime_str,
            "U10 (m/s)": metrics["U10_mps"],
            "Ueff (m/s)": metrics["Ueff_mps"],
            "Ueff method": metrics["Ueff_method"],
            "Mask threshold (ppm m)": metrics["mask_threshold_ppm_m"],
            "N plume pixels": metrics["n_plume_pixels"],
            "A plume mask (m2)": metrics["A_m2"],
            "L sqrt(A) (m)": metrics["L_m"],
            "IME (kg)": metrics["IME_kg"],
            "Q (kg/s)": metrics["Q_kg_s"],
            "Q (kg/h)": metrics["Q_kg_h"],
        }
        print(f"Event: {result['event']}")
        print(f"Satellite: {sensor}")
        print(f"U10 = {metrics['U10_mps']:.3f} m/s")
        print(f"Ueff = {metrics['Ueff_mps']:.3f} m/s")
        print(f"Ueff method: {metrics['Ueff_method']}")
        print(f"Mask threshold = {metrics['mask_threshold_ppm_m']:.6f} ppm m")
        print(f"A_M = {metrics['A_m2']:.2f} m2")
        print(f"L = {metrics['L_m']:.2f} m")
        print(f"IME = {metrics['IME_kg']:.6f} kg")
        print(f"Q = {metrics['Q_kg_s']:.6f} kg/s")
        print(f"Q = {metrics['Q_kg_h']:.3f} kg/h")
        return result
    except Exception as exc:
        print(f"ERROR on file: {tif_path}")
        print(exc)
        return _error_result(tif_path, exc)


def run_analysis(
    tif_paths: list[str],
    ghgsat_public_csv: str,
    output_dir: str,
    plume_threshold_ppm_m: float = 0.0,
    percentile_filter: float = 5.0,
    plot_mask: bool = False,
    create_plots: bool = True,
    show_plots: bool = False,
) -> pd.DataFrame:
    if not tif_paths:
        raise FileNotFoundError("No PRISMA or EnMAP GeoTIFF files were found.")
    if not os.path.exists(ghgsat_public_csv):
        raise FileNotFoundError(
            f"Public GHGSat metrics CSV not found: {ghgsat_public_csv}"
        )

    os.makedirs(output_dir, exist_ok=True)
    wind_by_event_hour = {}
    results = [
        process_plume_tif(
            tif_path,
            plume_threshold_ppm_m,
            percentile_filter,
            plot_mask,
            wind_by_event_hour,
        )
        for tif_path in tif_paths
    ]
    append_ghgsat_public_results(
        results,
        ghgsat_public_csv,
        wind_by_event_hour,
    )

    result_table = pd.DataFrame(results)
    pair_results, wide_ime, wide_q = build_paired_tables(result_table)
    comparison = build_detailed_comparison_table(pair_results, wide_ime)

    print(f"\nPaired IME events used: {len(wide_ime)}")
    print(wide_ime)
    print(f"\nPaired Q events used: {len(wide_q)}")

    write_analysis_outputs(result_table, comparison, output_dir)
    if create_plots:
        create_comparison_plots(
            wide_ime,
            wide_q,
            output_dir,
            show=show_plots,
        )
    return result_table
