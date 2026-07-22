from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Geod
from rasterio.warp import transform
from shapely.geometry import LineString
from shapely.ops import linemerge

from utils.metadata import discover_public_tifs, event_key, sat_name_from_tif


GEOD = Geod(ellps="WGS84")
PRISMA_BAND = 4
MIN_FLAT_FRACTION = 0.5
ENDPOINT_FRACTION = 0.1


def discover_centerline_jobs(data_dir: str | Path) -> list[tuple[str, Path]]:
    rasters_by_event = {}
    for raster_path in discover_public_tifs(data_dir):
        if sat_name_from_tif(raster_path) != "PRS":
            continue
        raster_path = Path(raster_path)
        if not (raster_path.parent / "plume_centerline_PRS.shp").is_file():
            continue
        event = event_key(raster_path)
        if event in rasters_by_event:
            raise ValueError(f"Multiple PRISMA rasters found for {event}.")
        rasters_by_event[event] = raster_path

    if not rasters_by_event:
        raise FileNotFoundError(
            f"No PRISMA raster and centerline pairs found under {data_dir}"
        )
    return sorted(rasters_by_event.items())


def read_centerline(
    centerline_path: str | Path,
    target_crs,
) -> LineString:
    centerline = gpd.read_file(centerline_path)
    if centerline.empty:
        raise ValueError(f"Centerline has no features: {centerline_path}")
    if centerline.crs is None:
        raise ValueError(f"Centerline CRS is undefined: {centerline_path}")

    centerline = centerline.to_crs(target_crs)
    geometries = centerline.geometry.dropna()
    geometries = geometries[~geometries.is_empty]
    if geometries.empty:
        raise ValueError(f"Centerline has no valid geometry: {centerline_path}")
    geometry = geometries.iloc[0]
    if geometry.geom_type == "MultiLineString":
        geometry = linemerge(geometry)
    if not isinstance(geometry, LineString):
        raise TypeError(f"Centerline must be a LineString: {centerline_path}")
    return geometry


def cumulative_distances_km(coords: np.ndarray, source_crs) -> np.ndarray:
    longitudes, latitudes = transform(
        source_crs,
        "EPSG:4326",
        coords[:, 0].tolist(),
        coords[:, 1].tolist(),
    )
    if len(longitudes) < 2:
        return np.zeros(len(longitudes), dtype=float)

    _, _, segment_lengths_m = GEOD.inv(
        longitudes[:-1],
        latitudes[:-1],
        longitudes[1:],
        latitudes[1:],
    )
    return np.concatenate(([0.0], np.cumsum(segment_lengths_m) / 1000.0))


def sample_profile(
    raster_path: str | Path,
    centerline_path: str | Path,
    band_index: int = 4,
    num_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")

    with rasterio.open(raster_path) as source:
        if source.crs is None:
            raise ValueError(f"Raster CRS is undefined: {raster_path}")
        if band_index > source.count:
            raise ValueError(
                f"Band {band_index} is unavailable in {raster_path}; "
                f"the raster has {source.count} band(s)."
            )

        centerline = read_centerline(centerline_path, source.crs)
        positions = np.linspace(0.0, centerline.length, num_points)
        points = [centerline.interpolate(position) for position in positions]
        coords = np.asarray([(point.x, point.y) for point in points])
        samples = list(source.sample(coords, indexes=band_index, masked=True))
        values = np.ma.vstack(samples).filled(np.nan).astype(float).ravel()/8
        distances_km = cumulative_distances_km(coords, source.crs)

    values[values == 0] = np.nan
    valid = np.isfinite(values)
    if not np.any(valid):
        raise ValueError(
            f"No valid samples in {raster_path} along {centerline_path}"
        )
    return distances_km[valid], values[valid]


def orient_profile(
    distances_km: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(values) < 3:
        return distances_km, values

    window = max(2, int(np.ceil(len(values) * ENDPOINT_FRACTION)))
    start_concentration = float(np.median(values[:window]))
    end_concentration = float(np.median(values[-window:]))
    reverse = end_concentration > start_concentration
    if not reverse:
        return distances_km, values

    reversed_distances = distances_km[-1] - distances_km[::-1]
    return reversed_distances, values[::-1]


def reduce_flat_profile(
    distances_km: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if len(values) < 3:
        return distances_km, values, False

    flat_segments = np.isclose(
        values[1:],
        values[:-1],
        rtol=1e-6,
        atol=1e-9,
    )
    if float(np.mean(flat_segments)) < MIN_FLAT_FRACTION:
        return distances_km, values, False

    run_starts = np.concatenate(([0], np.flatnonzero(~flat_segments) + 1))
    keep = run_starts
    if keep[-1] != len(values) - 1:
        keep = np.append(keep, len(values) - 1)
    return distances_km[keep], values[keep], True


def plot_prisma_centerline(
    raster_path: str | Path,
    output_file: str | Path,
    num_points: int = 100,
    show: bool = False,
) -> Path:
    raster_path = Path(raster_path)
    if not raster_path.is_file():
        raise FileNotFoundError(f"PRISMA raster not found: {raster_path}")
    if sat_name_from_tif(raster_path) != "PRS":
        raise ValueError(f"Raster is not PRISMA: {raster_path}")

    centerline_path = raster_path.parent / "plume_centerline_PRS.shp"
    if not centerline_path.is_file():
        raise FileNotFoundError(f"PRISMA centerline not found: {centerline_path}")

    distances_km, values = sample_profile(
        raster_path,
        centerline_path,
        band_index=PRISMA_BAND,
        num_points=num_points,
    )
    distances_km, values = orient_profile(distances_km, values)
    distances_km, values, reduced_profile = reduce_flat_profile(
        distances_km,
        values,
    )
    maximum_distance = distances_km.max()

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        distances_km,
        values,
        label="PRISMA",
        color="blue",
        marker="o",
        markersize=6,
        drawstyle="steps-post" if reduced_profile else "default",
    )

    axis.set_xlabel("Distance along PRISMA centerline (km)")
    axis.set_ylabel(r"CH$_4$ concentration (ppb)")
    axis.set_xlim(0, maximum_distance)
    axis.set_title(r"CH$_4$ concentrations along PRISMA plume centerline")
    axis.legend()
    axis.tick_params(axis="both", direction="in", length=6, width=1.5)
    for spine in axis.spines.values():
        spine.set_linewidth(1.5)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_file, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return output_file
