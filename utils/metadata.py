import os
import re
from pathlib import Path


def sat_name_from_tif(tif_path: str | Path) -> str:
    base = os.path.basename(str(tif_path))
    sat = base.split("_")[0]
    base_upper = base.upper()

    if sat == "PRS" or base_upper.startswith("PRS_"):
        return "PRS"
    if sat == "ENMAP" or "ENMAP" in base_upper:
        return "ENMAP"
    if sat.upper().startswith("GHGSAT") or "GHGSAT" in base_upper:
        return "GHGSat"
    return sat + "_GHGSat"


def event_key(tif_path: str | Path) -> str:
    parts = re.split(r"[\\/]", str(tif_path))
    date = next((part for part in parts if re.fullmatch(r"\d{8}", part)), "UNKNOWNDATE")
    plume = next(
        (part for part in parts if re.fullmatch(r"(?i)plume\d+", part)),
        None,
    )

    if plume is None:
        match = re.search(r"(?i)plume\d+", os.path.basename(str(tif_path)))
        plume = match.group(0) if match else "UNKNOWNPLUME"

    return f"{date}_{plume.upper()}"


def plume_directory_from_tif(tif_path: str | Path) -> str:
    tif_path = str(tif_path)
    date, plume = event_key(tif_path).split("_", 1)

    if date != "UNKNOWNDATE" and plume != "UNKNOWNPLUME":
        date_dir = os.path.abspath(os.path.dirname(tif_path))
        while os.path.basename(date_dir) != date:
            parent = os.path.dirname(date_dir)
            if parent == date_dir:
                date_dir = None
                break
            date_dir = parent

        if date_dir:
            for folder_name in (plume.capitalize(), plume.upper()):
                candidate = os.path.join(date_dir, folder_name)
                if os.path.isdir(candidate):
                    return candidate

    return os.path.dirname(tif_path)


def sat_group_from_sat_name(sat_name: str) -> str:
    if sat_name == "PRS":
        return "PRS"
    if sat_name == "ENMAP":
        return "ENMAP"
    if "GHGSat" in sat_name or sat_name.endswith("_GHGSat"):
        return "GHGSat"
    return "OTHER"


def discover_public_tifs(data_dir: str | Path) -> list[str]:
    """Find PRISMA and EnMAP plume GeoTIFFs in the public data tree."""
    paths = []
    for path in Path(data_dir).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        if "ERA5" in {part.upper() for part in path.parts}:
            continue
        if sat_name_from_tif(path) in {"PRS", "ENMAP"}:
            paths.append(str(path.resolve()))

    sensor_order = {"PRS": 0, "ENMAP": 1}
    return sorted(
        paths,
        key=lambda path: (
            event_key(path),
            sensor_order.get(sat_name_from_tif(path), 99),
            path,
        ),
    )
