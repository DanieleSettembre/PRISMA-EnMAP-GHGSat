from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
    transform,
)
from scipy.ndimage import shift as ndi_shift
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.metrics import structural_similarity


RGB_BANDS = (1, 2, 3)
DEFAULT_STEPS = (1.0, 0.75, 0.5, 0.25, 0.01)


def _normalized(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    minimum = np.nanmin(data)
    span = np.nanmax(data) - minimum
    if not np.isfinite(span) or span <= 0:
        return np.zeros_like(data)
    return (data - minimum) / span


def structural_similarity_score(data0: np.ndarray, data1: np.ndarray) -> float:
    return float(
        structural_similarity(
            _normalized(data0),
            _normalized(data1),
            data_range=1.0,
        )
    )
def _three_band_profile(source: rasterio.io.DatasetReader, **updates) -> dict:
    profile = source.profile.copy()
    for key in ("blockxsize", "blockysize", "tiled", "interleave"):
        profile.pop(key, None)
    profile.update(
        count=3,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        **updates,
    )
    if np.issubdtype(np.dtype(profile["dtype"]), np.floating):
        profile["predictor"] = 3
    return profile


def reproject_reference(
    input_file: str | Path,
    output_file: str | Path,
    target_crs: str = "EPSG:4326",
) -> Path:
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        with rasterio.open(input_file) as source:
            if source.count != 3:
                raise ValueError(f"Expected a three-band EnMAP raster: {input_file}")
            if source.crs is None:
                raise ValueError(f"EnMAP raster has no CRS: {input_file}")
            if source.crs.to_string() == target_crs:
                return input_file

            dst_transform, width, height = calculate_default_transform(
                source.crs,
                target_crs,
                source.width,
                source.height,
                *source.bounds,
            )
            profile = _three_band_profile(
                source,
                crs=target_crs,
                transform=dst_transform,
                width=width,
                height=height,
            )

            with rasterio.open(output_file, "w", **profile) as destination:
                for band_index in RGB_BANDS:
                    reproject(
                        source=rasterio.band(source, band_index),
                        destination=rasterio.band(destination, band_index),
                        src_transform=source.transform,
                        src_crs=source.crs,
                        dst_transform=dst_transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest,
                    )
                destination.colorinterp = source.colorinterp[:3]
                destination.update_tags(**source.tags())

    return output_file


def resample_prisma_to_reference(
    prisma_file: str | Path,
    reference_file: str | Path,
    output_file: str | Path,
) -> Path:
    prisma_file = Path(prisma_file)
    reference_file = Path(reference_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        with rasterio.open(prisma_file) as prisma, rasterio.open(
            reference_file
        ) as reference:
            if prisma.count != 3:
                raise ValueError(f"Expected a three-band PRISMA raster: {prisma_file}")
            if prisma.crs is None or reference.crs is None:
                raise ValueError("Both coregistration rasters must define a CRS.")

            dst_transform, width, height = calculate_default_transform(
                prisma.crs,
                reference.crs,
                prisma.width,
                prisma.height,
                *prisma.bounds,
                resolution=(
                    abs(reference.transform.a),
                    abs(reference.transform.e),
                ),
            )
            profile = _three_band_profile(
                prisma,
                dtype="float32",
                width=width,
                height=height,
                transform=dst_transform,
                crs=reference.crs,
            )

            with rasterio.open(output_file, "w", **profile) as destination:
                for band_index in RGB_BANDS:
                    reproject(
                        source=rasterio.band(prisma, band_index),
                        destination=rasterio.band(destination, band_index),
                        src_transform=prisma.transform,
                        src_crs=prisma.crs,
                        dst_transform=dst_transform,
                        dst_crs=reference.crs,
                        src_nodata=prisma.nodata,
                        dst_nodata=prisma.nodata,
                        resampling=Resampling.nearest,
                    )
                destination.colorinterp = prisma.colorinterp[:3]
                destination.update_tags(**prisma.tags())

    return output_file


def _read_rgb(raster_file: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(raster_file) as source:
        if source.count != 3:
            raise ValueError(f"Expected a three-band raster: {raster_file}")
        data = source.read().astype(np.float32)
        rgb = np.moveaxis(data, 0, -1)
        return rgb, data


def _correction_factor(prisma_data: np.ndarray, enmap_data: np.ndarray) -> float:
    enmap_values = enmap_data.reshape(-1).astype(np.float64)
    prisma_values = prisma_data.reshape(-1).astype(np.float64)
    size = min(enmap_values.size, prisma_values.size)
    enmap_values = enmap_values[:size]
    prisma_values = prisma_values[:size]

    valid = np.isfinite(enmap_values) & np.isfinite(prisma_values)
    if not np.any(valid):
        return 1.0
    percentile_90 = np.percentile(prisma_values[valid], 90)
    valid &= prisma_values < percentile_90
    denominator = np.sum(prisma_values[valid] ** 2)
    if denominator <= 1e-8:
        return 1.0
    return float(np.sum(enmap_values[valid] * prisma_values[valid]) / denominator)


def _gradient(rgb: np.ndarray) -> np.ndarray:
    return sobel(rgb2gray(rgb))


def _matching_prisma_pixel(
    enmap_file: str | Path,
    prisma_file: str | Path,
    x_pixel: int,
    y_pixel: int,
) -> tuple[int, int]:
    with rasterio.open(enmap_file) as enmap, rasterio.open(prisma_file) as prisma:
        x_coord, y_coord = rasterio.transform.xy(
            enmap.transform,
            y_pixel,
            x_pixel,
        )
        x_prisma, y_prisma = transform(
            enmap.crs,
            prisma.crs,
            [x_coord],
            [y_coord],
        )
        col_prisma, row_prisma = ~prisma.transform * (
            x_prisma[0],
            y_prisma[0],
        )
    return int(round(row_prisma)), int(round(col_prisma))


def _extract_roi(
    image: np.ndarray,
    row: int,
    col: int,
    patch_size: int,
    margin: int,
) -> tuple[np.ndarray, int, int]:
    row_start = int(np.floor(row)) - margin
    col_start = int(np.floor(col)) - margin
    row_end = row_start + patch_size + 2 * margin
    col_end = col_start + patch_size + 2 * margin

    clipped_row_start = max(0, row_start)
    clipped_col_start = max(0, col_start)
    clipped_row_end = min(image.shape[0], row_end)
    clipped_col_end = min(image.shape[1], col_end)
    roi = image[
        clipped_row_start:clipped_row_end,
        clipped_col_start:clipped_col_end,
    ]
    return (
        roi,
        int(np.floor(row)) - clipped_row_start,
        int(np.floor(col)) - clipped_col_start,
    )


def find_best_shift_multiscale(
    patch_enmap: np.ndarray,
    gradient_prisma: np.ndarray,
    row_prisma: int,
    col_prisma: int,
    patch_size: int = 10,
    search_range: float = 10.0,
    steps: tuple[float, ...] = DEFAULT_STEPS,
    refine_window: float = 1.0,
    interpolation_order: int = 3,
) -> tuple[tuple[float, float], float]:
    margin = int(np.ceil(search_range)) + 2
    roi, row_offset, col_offset = _extract_roi(
        gradient_prisma,
        row_prisma,
        col_prisma,
        patch_size,
        margin,
    )
    best_score = -np.inf
    best_shift = (0.0, 0.0)

    for stage_index, step in enumerate(steps):
        if stage_index == 0:
            center_dy, center_dx = 0.0, 0.0
            current_range = float(search_range)
        else:
            center_dy, center_dx = best_shift
            current_range = refine_window

        dx_values = np.arange(
            center_dx - current_range,
            center_dx + current_range + 1e-9,
            step,
        )
        dy_values = np.arange(
            center_dy - current_range,
            center_dy + current_range + 1e-9,
            step,
        )
        stage_score = -np.inf
        stage_shift = best_shift

        for dx in dx_values:
            for dy in dy_values:
                shifted_roi = ndi_shift(
                    roi,
                    shift=(dy, dx),
                    order=interpolation_order,
                    mode="constant",
                    cval=0.0,
                )
                patch_prisma = shifted_roi[
                    row_offset : row_offset + patch_size,
                    col_offset : col_offset + patch_size,
                ]
                if patch_prisma.shape != patch_enmap.shape:
                    continue
                current_score = structural_similarity_score(
                    patch_enmap,
                    patch_prisma,
                )
                if np.isfinite(current_score) and current_score > stage_score:
                    stage_score = current_score
                    stage_shift = (float(dy), float(dx))

        if stage_score > best_score:
            best_score = stage_score
            best_shift = stage_shift

    if not np.isfinite(best_score):
        raise ValueError("No valid coregistration shift was found.")

    return best_shift, float(best_score)


def coregister_prisma(
    enmap_file: str | Path,
    prisma_file: str | Path,
    working_dir: str | Path,
    x0_pixel: int = 512,
    y0_pixel: int = 731,
    patch_size: int = 10,
    search_range: float = 10.0,
    step: float = 0.001,
    enmap_bands: tuple[int, ...] = (0, 1, 2),
    prisma_bands: tuple[int, ...] = (0, 1, 2),
) -> dict:
    enmap_file = Path(enmap_file)
    prisma_file = Path(prisma_file)
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    if enmap_bands != (0, 1, 2) or prisma_bands != (0, 1, 2):
        raise ValueError("The public coregistration workflow uses bands 0, 1 and 2.")

    resampled_file = working_dir / f"{prisma_file.stem}_resampled.tif"
    resample_prisma_to_reference(prisma_file, enmap_file, resampled_file)

    prisma_rgb, prisma_data = _read_rgb(resampled_file)
    enmap_rgb, enmap_data = _read_rgb(enmap_file)
    correction_factor = _correction_factor(prisma_data, enmap_data)
    prisma_gradient = _gradient(prisma_rgb * correction_factor)
    enmap_gradient = _gradient(enmap_rgb)

    row_prisma, col_prisma = _matching_prisma_pixel(
        enmap_file,
        resampled_file,
        x0_pixel,
        y0_pixel,
    )
    patch_enmap = enmap_gradient[
        y0_pixel : y0_pixel + patch_size,
        x0_pixel : x0_pixel + patch_size,
    ]
    if patch_enmap.shape != (patch_size, patch_size):
        raise ValueError("The selected EnMAP patch falls outside the raster.")

    (dy_shift, dx_shift), best_score = find_best_shift_multiscale(
        patch_enmap=patch_enmap,
        gradient_prisma=prisma_gradient,
        row_prisma=row_prisma,
        col_prisma=col_prisma,
        patch_size=patch_size,
        search_range=search_range,
    )

    return {
        "dx_pixels": dx_shift,
        "dy_pixels": dy_shift,
        "SSIM": best_score,
    }
