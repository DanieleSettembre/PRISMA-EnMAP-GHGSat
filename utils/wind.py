import os
import re
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from utils.metadata import sat_name_from_tif


LOCAL_UTC_OFFSET_HOURS = 0
BBOX_BUFFER_DEG = 0.0
MAX_AREA_DEG2 = 50.0


def meteorological_wind_direction_from(u10, v10):
    """Return the direction the wind comes from, clockwise from North."""
    return np.mod(180.0 + np.degrees(np.arctan2(u10, v10)), 360.0)


def open_era5_dataset(nc_path: str) -> xr.Dataset:
    """Open ERA5 NetCDF data, with an h5py fallback for limited environments."""
    try:
        return xr.open_dataset(nc_path)
    except ValueError:
        import h5py

        with h5py.File(nc_path, "r") as h5:
            required = {"u10", "v10", "latitude", "longitude"}
            if not required.issubset(h5.keys()):
                raise

            u10 = np.asarray(h5["u10"][:])
            v10 = np.asarray(h5["v10"][:])
            coords = {
                "latitude": np.asarray(h5["latitude"][:]),
                "longitude": np.asarray(h5["longitude"][:]),
            }
            dims = ["latitude", "longitude"]

            if u10.ndim == 3:
                time_name = "valid_time" if "valid_time" in h5 else "time"
                raw_time = (
                    np.asarray(h5[time_name][:])
                    if time_name in h5
                    else np.arange(u10.shape[0])
                )
                coords[time_name] = (
                    raw_time.astype("datetime64[s]")
                    if time_name in h5
                    else raw_time
                )
                dims = [time_name, "latitude", "longitude"]

            return xr.Dataset(
                {"u10": (dims, u10), "v10": (dims, v10)},
                coords=coords,
            )


def download_era5_land_cached(
    request: dict,
    out_zip_path: str,
    force: bool = False,
) -> str:
    cached_file_exists = (
        os.path.exists(out_zip_path) and os.path.getsize(out_zip_path) > 0
    )
    if not force and cached_file_exists:
        return out_zip_path

    import cdsapi

    temporary_path = f"{out_zip_path}.download"
    client = cdsapi.Client()
    try:
        client.retrieve("reanalysis-era5-land", request).download(temporary_path)
        os.replace(temporary_path, out_zip_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return out_zip_path


def plume_point_from_raster(
    plume_raster_path: str,
    method: str = "centroid_mask",
) -> tuple[float, float]:
    """Return a representative plume point as latitude and longitude."""
    with rasterio.open(plume_raster_path) as src:
        sensor = sat_name_from_tif(plume_raster_path)
        if sensor == "PRS":
            if src.count < 4:
                raise ValueError("PRISMA raster does not contain the required band 4.")
            band_index = 4
        elif sensor == "ENMAP":
            band_index = 4 if src.count >= 4 else 1
        else:
            band_index = 1

        array = src.read(band_index)
        valid_mask = (src.read_masks(band_index) > 0) & np.isfinite(array)
        if src.nodata is not None:
            valid_mask &= array != src.nodata

        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {plume_raster_path}")
        if not np.any(valid_mask):
            raise ValueError("Raster contains no valid plume pixels.")

        if method == "max":
            valid_values = np.where(valid_mask, array, -np.inf)
            row, col = np.unravel_index(np.argmax(valid_values), valid_values.shape)
        elif method == "centroid_mask":
            plume_mask = valid_mask & (array > 0)
            if not np.any(plume_mask):
                raise ValueError("No plume pixels greater than zero were found.")
            rows, cols = np.where(plume_mask)
            row = int(np.round(rows.mean()))
            col = int(np.round(cols.mean()))
        else:
            raise ValueError("method must be 'max' or 'centroid_mask'")

        x, y = rasterio.transform.xy(src.transform, row, col, offset="center")
        if str(src.crs).upper() == "EPSG:4326":
            lon, lat = x, y
        else:
            from pyproj import Transformer

            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(x, y)

    return float(lat), float(lon)


def _drop_size1_dims(data_array: xr.DataArray) -> xr.DataArray:
    for dimension in list(data_array.dims):
        if data_array.sizes.get(dimension, 1) == 1 and dimension not in {
            "latitude",
            "longitude",
        }:
            data_array = data_array.isel({dimension: 0})
    return data_array


def era5_wind_at_point(
    nc_path: str,
    target_lat: float,
    target_lon: float,
    target_time_utc: datetime | None = None,
) -> dict:
    """Extract nearest ERA5 wind and meteorological direction from North."""
    dataset = open_era5_dataset(nc_path)
    try:
        if "u10" not in dataset.variables or "v10" not in dataset.variables:
            raise KeyError(
                f"Wind variables not found. Available: {list(dataset.variables)}"
            )

        u = dataset["u10"]
        v = dataset["v10"]
        if target_time_utc is not None:
            time_coord = next(
                (coord for coord in ("time", "valid_time") if coord in u.coords),
                None,
            )
            if time_coord is not None:
                target_time_naive = target_time_utc.replace(tzinfo=None)
                u = u.sel({time_coord: target_time_naive}, method="nearest")
                v = v.sel({time_coord: target_time_naive}, method="nearest")

        lon_values = dataset["longitude"].values
        lon_query = float(target_lon)
        if lon_values.max() > 180 and lon_query < 0:
            lon_query = (lon_query + 360) % 360

        u_point = u.sel(
            latitude=float(target_lat),
            longitude=lon_query,
            method="nearest",
        )
        v_point = v.sel(
            latitude=float(target_lat),
            longitude=lon_query,
            method="nearest",
        )
        u_point = _drop_size1_dims(u_point)
        v_point = _drop_size1_dims(v_point)

        u10 = float(np.asarray(u_point.values))
        v10 = float(np.asarray(v_point.values))
        speed = float(np.sqrt(u10**2 + v10**2))
        direction_from = float(meteorological_wind_direction_from(u10, v10))

        lat_nearest = float(np.asarray(u_point["latitude"].values))
        lon_nearest = float(np.asarray(u_point["longitude"].values))
        if lon_values.max() > 180 and lon_nearest > 180:
            lon_nearest -= 360.0

        return {
            "lat_nearest": lat_nearest,
            "lon_nearest": lon_nearest,
            "u10": u10,
            "v10": v10,
            "speed": speed,
            "direction_from_deg": direction_from,
        }
    finally:
        dataset.close()


def era5_subset_covers_point(
    nc_path: str,
    target_lat: float,
    target_lon: float,
) -> bool:
    dataset = open_era5_dataset(nc_path)
    try:
        latitudes = np.asarray(dataset["latitude"].values, dtype=float)
        longitudes = np.asarray(dataset["longitude"].values, dtype=float)
        lon_query = float(target_lon)
        if longitudes.max() > 180 and lon_query < 0:
            lon_query = (lon_query + 360) % 360

        def coordinate_is_covered(values: np.ndarray, query: float) -> bool:
            unique_values = np.unique(values)
            if unique_values.size > 1:
                tolerance = float(np.median(np.diff(unique_values))) / 2.0
            else:
                tolerance = 0.05
            return bool(
                unique_values.min() - tolerance <= query
                <= unique_values.max() + tolerance
            )

        return coordinate_is_covered(
            latitudes,
            float(target_lat),
        ) and coordinate_is_covered(longitudes, lon_query)
    finally:
        dataset.close()


def parse_acquisition_times(file_path: str) -> tuple[datetime, datetime]:
    base = os.path.basename(file_path)
    match = re.search(r"_(\d{14})_(\d{14})_", base)
    if not match:
        raise ValueError(f"Cannot extract acquisition times from: {base}")
    return (
        datetime.strptime(match.group(1), "%Y%m%d%H%M%S"),
        datetime.strptime(match.group(2), "%Y%m%d%H%M%S"),
    )


def local_to_utc(dt_local: datetime, utc_offset_hours: int) -> datetime:
    local_timezone = timezone(timedelta(hours=utc_offset_hours))
    return dt_local.replace(tzinfo=local_timezone).astimezone(timezone.utc)


def acquisition_interval_utc(file_path: str) -> tuple[datetime, datetime]:
    start_local, stop_local = parse_acquisition_times(file_path)
    return (
        local_to_utc(start_local, LOCAL_UTC_OFFSET_HOURS),
        local_to_utc(stop_local, LOCAL_UTC_OFFSET_HOURS),
    )


def nearest_acquisition_path(
    target_path: str,
    candidate_paths: list[str],
) -> str:
    """Match the target start time, preferring the earlier interval on ties."""
    if not candidate_paths:
        return target_path

    target_start, _ = acquisition_interval_utc(target_path)
    candidates = []
    for candidate_path in candidate_paths:
        try:
            start_utc, stop_utc = acquisition_interval_utc(candidate_path)
        except ValueError:
            continue

        if start_utc <= target_start <= stop_utc:
            distance_seconds = 0.0
        elif target_start < start_utc:
            distance_seconds = (start_utc - target_start).total_seconds()
        else:
            distance_seconds = (target_start - stop_utc).total_seconds()

        candidates.append(
            (
                distance_seconds,
                start_utc,
                stop_utc,
                str(candidate_path),
            )
        )

    if not candidates:
        return target_path
    return min(candidates)[3]


def nearest_era5_hour_utc(time_utc: datetime) -> str:
    return nearest_era5_datetime_utc(time_utc).strftime("%H:00")


def nearest_era5_datetime_utc(time_utc: datetime) -> datetime:
    lower_hour = time_utc.replace(minute=0, second=0, microsecond=0)
    upper_hour = lower_hour + timedelta(hours=1)
    return (
        lower_hour
        if (time_utc - lower_hour) <= (upper_hour - time_utc)
        else upper_hour
    )


def acquisition_midpoint_utc(file_path: str) -> datetime:
    start_utc, stop_utc = acquisition_interval_utc(file_path)
    return start_utc + (stop_utc - start_utc) / 2


def raster_bbox_latlon(
    raster_path: str,
    buffer_deg: float = 0.0,
) -> tuple[float, float, float, float]:
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("GeoTIFF has no CRS.")
        bounds = src.bounds
        crs = src.crs

    west, south, east, north = transform_bounds(
        crs,
        "EPSG:4326",
        bounds.left,
        bounds.bottom,
        bounds.right,
        bounds.top,
        densify_pts=21,
    )
    north += buffer_deg
    south -= buffer_deg
    east += buffer_deg
    west -= buffer_deg

    approximate_area = abs((east - west) * (north - south))
    if approximate_area > MAX_AREA_DEG2:
        raise ValueError(
            f"Requested area is too large ({approximate_area:.2f} deg2)."
        )
    return north, west, south, east


def raster_bbox_union_latlon(
    raster_paths: list[str],
    buffer_deg: float = 0.0,
) -> tuple[float, float, float, float]:
    if not raster_paths:
        raise ValueError("At least one raster is required for the ERA5 area.")

    bounds = [raster_bbox_latlon(path) for path in raster_paths]
    north = max(bound[0] for bound in bounds) + buffer_deg
    west = min(bound[1] for bound in bounds) - buffer_deg
    south = min(bound[2] for bound in bounds) - buffer_deg
    east = max(bound[3] for bound in bounds) + buffer_deg

    approximate_area = abs((east - west) * (north - south))
    if approximate_area > MAX_AREA_DEG2:
        raise ValueError(
            f"Requested area is too large ({approximate_area:.2f} deg2)."
        )
    return north, west, south, east


def build_cds_request(
    acquisition_time_utc: datetime,
    area_nwse: tuple[float, float, float, float],
) -> dict:
    return {
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year": acquisition_time_utc.strftime("%Y"),
        "month": acquisition_time_utc.strftime("%m"),
        "day": [acquisition_time_utc.strftime("%d")],
        "time": nearest_era5_hour_utc(acquisition_time_utc),
        "data_format": "netcdf",
        "download_format": "zip",
        "area": list(area_nwse),
    }


def find_netcdf_in_zip(zip_path: str, out_dir: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as archive:
        nc_files = [name for name in archive.namelist() if name.lower().endswith(".nc")]
        if not nc_files:
            raise ValueError("Downloaded archive contains no NetCDF file.")
        nc_name = nc_files[0]
        archive.extract(nc_name, out_dir)
    return os.path.join(out_dir, nc_name)


def compute_speed_direction_tiff(
    nc_path: str,
    area_nwse: tuple[float, float, float, float],
    out_tif_path: str,
    target_time_utc: datetime | None = None,
) -> bool:
    """Write speed and meteorological wind direction when the subset is at least 2x2."""
    north, west, south, east = area_nwse
    dataset = open_era5_dataset(nc_path)
    try:
        if "u10" not in dataset.variables or "v10" not in dataset.variables:
            raise KeyError(
                f"Wind variables not found. Available: {list(dataset.variables)}"
            )

        lon_values = dataset["longitude"].values
        if lon_values.max() > 180 and west < 0:
            west = (west + 360) % 360
            east = (east + 360) % 360

        latitude = dataset["latitude"]
        longitude = dataset["longitude"]
        selected_latitudes = latitude.where(
            (latitude >= south) & (latitude <= north),
            drop=True,
        )
        selected_longitudes = longitude.where(
            (longitude >= west) & (longitude <= east),
            drop=True,
        )
        subset = dataset.sel(
            latitude=selected_latitudes,
            longitude=selected_longitudes,
        )
        u = subset["u10"]
        v = subset["v10"]

        if target_time_utc is not None:
            time_coord = next(
                (coord for coord in ("time", "valid_time") if coord in u.coords),
                None,
            )
            if (
                time_coord is not None
                and time_coord in u.dims
                and u.sizes.get(time_coord, 1) > 1
            ):
                target_time_naive = (
                    target_time_utc.astimezone(timezone.utc).replace(tzinfo=None)
                    if target_time_utc.tzinfo is not None
                    else target_time_utc
                )
                u = u.sel({time_coord: target_time_naive}, method="nearest")
                v = v.sel({time_coord: target_time_naive}, method="nearest")

        u = _drop_size1_dims(u)
        v = _drop_size1_dims(v)
        if u.ndim != 2 or v.ndim != 2:
            raise ValueError(
                f"Wind arrays are not 2D: u={u.dims}/{u.shape}, v={v.dims}/{v.shape}"
            )

        u_values = u.values.astype(np.float32)
        v_values = v.values.astype(np.float32)
        speed = np.sqrt(u_values**2 + v_values**2).astype(np.float32)
        direction = meteorological_wind_direction_from(
            u_values,
            v_values,
        ).astype(np.float32)
        latitudes = u["latitude"].values
        longitudes = u["longitude"].values

        if len(longitudes) < 1 or len(latitudes) < 1:
            raise ValueError("ERA5 subset is empty for the requested area.")
        if len(latitudes) > 1 and latitudes[0] < latitudes[-1]:
            latitudes = latitudes[::-1]
            speed = speed[::-1, :]
            direction = direction[::-1, :]
        if len(longitudes) < 2 or len(latitudes) < 2:
            return False

        x_resolution = float(abs(longitudes[1] - longitudes[0]))
        y_resolution = float(abs(latitudes[1] - latitudes[0]))
        transform = from_origin(
            west=float(min(longitudes)) - x_resolution / 2.0,
            north=float(max(latitudes)) + y_resolution / 2.0,
            xsize=x_resolution,
            ysize=y_resolution,
        )
        profile = {
            "driver": "GTiff",
            "height": speed.shape[0],
            "width": speed.shape[1],
            "count": 2,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": transform,
            "compress": "deflate",
        }
        temporary_path = f"{out_tif_path}.tmp.tif"
        try:
            with rasterio.open(temporary_path, "w", **profile) as destination:
                destination.write(speed, 1)
                destination.set_band_description(1, "wind_speed_10m_m_s")
                destination.write(direction, 2)
                destination.set_band_description(
                    2,
                    "wind_direction_from_north_deg",
                )
            os.replace(temporary_path, out_tif_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        return True
    finally:
        dataset.close()


def era5_output_dir_for_raster(raster_path: str) -> str:
    current = Path(raster_path).resolve().parent
    for directory in (current, *current.parents):
        if re.fullmatch(r"\d{8}", directory.name):
            return str(directory / "ERA5")
    return str(current.parent / "ERA5")


def windspeed_from_time_and_area(
    time_source_path: str,
    plume_raster_path: str,
    area_nwse: tuple[float, float, float, float] | None = None,
) -> tuple[float, str]:
    area = area_nwse or raster_bbox_latlon(
        plume_raster_path,
        buffer_deg=BBOX_BUFFER_DEG,
    )
    output_dir = era5_output_dir_for_raster(plume_raster_path)
    os.makedirs(output_dir, exist_ok=True)

    start_utc, stop_utc = acquisition_interval_utc(time_source_path)
    midpoint_utc = start_utc + (stop_utc - start_utc) / 2

    request = build_cds_request(start_utc, area)
    tag = (
        f"{start_utc.strftime('%Y%m%dT%H%M%SZ')}_"
        f"{stop_utc.strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_zip = os.path.join(output_dir, f"ERA5L_{tag}.zip")
    out_tif = os.path.join(output_dir, f"ERA5L_wind_speed_dir_{tag}.tif")
    download_era5_land_cached(request, out_zip)

    nc_path = find_netcdf_in_zip(out_zip, output_dir)
    plume_lat, plume_lon = plume_point_from_raster(
        plume_raster_path,
        method="centroid_mask",
    )
    if not era5_subset_covers_point(nc_path, plume_lat, plume_lon):
        download_era5_land_cached(request, out_zip, force=True)
        nc_path = find_netcdf_in_zip(out_zip, output_dir)
        if not era5_subset_covers_point(nc_path, plume_lat, plume_lon):
            raise ValueError(
                "Downloaded ERA5 subset does not cover the plume point."
            )

    wind_point = era5_wind_at_point(
        nc_path,
        plume_lat,
        plume_lon,
        target_time_utc=midpoint_utc,
    )
    try:
        compute_speed_direction_tiff(
            nc_path,
            area,
            out_tif,
            target_time_utc=midpoint_utc,
        )
    except PermissionError:
        pass

    return wind_point["speed"], midpoint_utc.strftime("%Y/%m/%d %H:%M:%S")


def windspeed(raster_path: str, plume_dir: str | None = None) -> tuple[float, str]:
    return windspeed_from_time_and_area(raster_path, raster_path)
