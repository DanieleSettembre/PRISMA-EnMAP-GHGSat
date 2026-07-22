import argparse
import csv
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coregistration import coregister_prisma, reproject_reference


DEFAULT_ENMAP = (
    PROJECT_ROOT
    / "coregistration_data"
    / "enmap"
    / "ENMAP_20240911_RGB.tif"
)
DEFAULT_PRISMA_DIR = PROJECT_ROOT / "coregistration_data" / "prisma"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "coregistration"
ENMAP_BANDS = (0, 1, 2)
PRISMA_BANDS = (0, 1, 2)
PATCH_SIZE = 10
SEARCH_RANGE = 10.0
X0_PIXEL = 512
Y0_PIXEL = 731
STEP = 0.001


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate SSIM-based PRISMA-to-EnMAP coregistration shifts.",
    )
    parser.add_argument("--enmap", type=Path, default=DEFAULT_ENMAP)
    parser.add_argument("--prisma-files", nargs="+", type=Path)
    parser.add_argument("--prisma-dir", type=Path, default=DEFAULT_PRISMA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--x0-pixel", type=int, default=X0_PIXEL)
    parser.add_argument("--y0-pixel", type=int, default=Y0_PIXEL)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--search-range", type=float, default=SEARCH_RANGE)
    parser.add_argument("--step", type=float, default=STEP)
    return parser


def discover_prisma_files(prisma_dir: Path) -> list[Path]:
    files = sorted(
        path
        for path in prisma_dir.rglob("*.tif")
        if path.is_file() and path.name.upper().startswith("PRS_")
    )
    if not files:
        raise FileNotFoundError(f"No PRISMA inputs found under {prisma_dir}")
    return files


def acquisition_date(raster_file: Path) -> str:
    match = re.search(r"\d{8}", raster_file.name)
    return match.group(0) if match else "unknown_date"


def write_summary(results: list[dict], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date",
        "file",
        "dx_pixels",
        "dy_pixels",
        "SSIM",
    ]
    with output_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    args = build_parser().parse_args()
    prisma_files = args.prisma_files or discover_prisma_files(args.prisma_dir)
    if not args.enmap.is_file():
        raise FileNotFoundError(f"EnMAP input not found: {args.enmap}")

    results = []
    with TemporaryDirectory(prefix="prisma_coregistration_") as temp_dir:
        temp_root = Path(temp_dir)
        reference_file = reproject_reference(
            args.enmap,
            temp_root / "ENMAP_20240911_RGB_EPSG4326.tif",
        )
        for prisma_file in prisma_files:
            if not prisma_file.is_file():
                raise FileNotFoundError(f"PRISMA input not found: {prisma_file}")
            date = acquisition_date(prisma_file)
            result = coregister_prisma(
                enmap_file=reference_file,
                prisma_file=prisma_file,
                working_dir=temp_root / date,
                x0_pixel=args.x0_pixel,
                y0_pixel=args.y0_pixel,
                patch_size=args.patch_size,
                search_range=args.search_range,
                step=args.step,
                enmap_bands=ENMAP_BANDS,
                prisma_bands=PRISMA_BANDS,
            )
            results.append(
                {
                    "date": date,
                    "file": prisma_file.name,
                    **result,
                }
            )

    write_summary(results, args.output_dir / "Coregistration_SSIM.csv")


if __name__ == "__main__":
    main()
