import os

import numpy as np
import pandas as pd

from utils.emissions import effective_wind_speed_by_sensor
from utils.metadata import event_key
from utils.wind import (
    acquisition_midpoint_utc,
    nearest_era5_datetime_utc,
    windspeed_from_time_and_area,
)


def load_ghgsat_public_metrics(csv_path: str) -> dict:
    """Load public GHGSat metrics used when private TIFFs are unavailable."""
    if not os.path.exists(csv_path):
        return {}

    table = pd.read_csv(csv_path)
    lookup = {}
    for row in table.to_dict("records"):
        event = str(row.get("event", ""))
        file_name = os.path.basename(str(row.get("file", "")))
        if event:
            lookup[(event, file_name)] = row
            lookup[(event, "")] = row
    return lookup


def ghgsat_public_metrics_for_tif(tif_path: str, lookup: dict) -> dict | None:
    event = event_key(tif_path)
    file_name = os.path.basename(tif_path)
    return lookup.get((event, file_name)) or lookup.get((event, ""))


def result_from_ghgsat_public_row(
    row: dict,
    tif_path: str | None = None,
) -> dict:
    file_name = os.path.basename(str(row.get("file", "")))
    return {
        "event": str(row.get("event", "")),
        "sat": "GHGSat",
        "sat_group": "GHGSat",
        "tif": tif_path if tif_path is not None else file_name,
        "datetime": None,
        "U10 (m/s)": np.nan,
        "Ueff (m/s)": np.nan,
        "Ueff method": None,
        "N plume pixels": np.nan,
        "A plume mask (m2)": np.nan,
        "L sqrt(A) (m)": float(row["L_m"]),
        "IME (kg)": float(row["IME_kg"]),
        "Q (kg/s)": np.nan,
        "Q (kg/h)": np.nan,
        "error": "GHGSat TIFF unavailable; metrics loaded from public CSV",
    }


def fill_ghgsat_public_wind(
    result: dict,
    time_source_path: str | None,
    reference_tif: str | None,
    wind_speed_override: float | None = None,
) -> dict:
    if not time_source_path or not reference_tif:
        return result

    try:
        midpoint_utc = acquisition_midpoint_utc(time_source_path)
        datetime_str = midpoint_utc.strftime("%Y/%m/%d %H:%M:%S")
        if wind_speed_override is None:
            u10_mps, datetime_str = windspeed_from_time_and_area(
                time_source_path,
                reference_tif,
            )
        else:
            u10_mps = float(wind_speed_override)
    except Exception as exc:
        result["error"] = (
            f"{result.get('error', '')}; GHGSat wind unavailable: {exc}"
        )
        return result

    ueff_mps, ueff_method = effective_wind_speed_by_sensor(u10_mps, "GHGSat")
    ime_kg = result["IME (kg)"]
    length_m = result["L sqrt(A) (m)"]

    if length_m <= 0 or not np.isfinite(ueff_mps):
        q_kg_s = np.nan
        q_kg_h = np.nan
    else:
        q_kg_s = ime_kg * ueff_mps / length_m
        q_kg_h = q_kg_s * 3600.0

    result.update(
        {
            "datetime": datetime_str,
            "U10 (m/s)": float(u10_mps),
            "Ueff (m/s)": ueff_mps,
            "Ueff method": ueff_method,
            "Q (kg/s)": q_kg_s,
            "Q (kg/h)": q_kg_h,
        }
    )
    return result


def reference_raster_by_event(results: list[dict]) -> dict:
    raster_by_event = {}
    for sensor_group in ["PRS", "ENMAP"]:
        for row in results:
            event = str(row.get("event", ""))
            tif_path = str(row.get("tif", ""))
            if (
                event
                and event not in raster_by_event
                and row.get("sat_group") == sensor_group
                and os.path.exists(tif_path)
            ):
                raster_by_event[event] = tif_path
    return raster_by_event


def append_ghgsat_public_results(
    results: list[dict],
    csv_path: str,
    wind_by_event_hour: dict | None = None,
) -> None:
    """Append public GHGSat rows for events represented by public raster data."""
    if not os.path.exists(csv_path):
        return

    table = pd.read_csv(csv_path)
    if table.empty or "event" not in table.columns:
        return

    requested_events = {
        str(row.get("event", ""))
        for row in results
        if row.get("sat_group") in ["PRS", "ENMAP"]
    }
    existing_events = {
        str(row.get("event", ""))
        for row in results
        if row.get("sat_group") == "GHGSat"
    }
    raster_by_event = reference_raster_by_event(results)
    wind_by_event_hour = wind_by_event_hour if wind_by_event_hour is not None else {}

    for row in table.to_dict("records"):
        event = str(row.get("event", ""))
        if not event or event not in requested_events or event in existing_events:
            continue

        result = result_from_ghgsat_public_row(row)
        time_source_path = str(row.get("file", ""))
        midpoint_utc = acquisition_midpoint_utc(time_source_path)
        wind_key = (event, nearest_era5_datetime_utc(midpoint_utc))
        fill_ghgsat_public_wind(
            result,
            time_source_path,
            raster_by_event.get(event),
            wind_speed_override=wind_by_event_hour.get(wind_key),
        )
        if np.isfinite(result.get("U10 (m/s)", np.nan)):
            wind_by_event_hour.setdefault(wind_key, result["U10 (m/s)"])
        results.append(result)
        existing_events.add(event)
