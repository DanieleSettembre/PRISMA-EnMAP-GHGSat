import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.transects import (
    discover_centerline_jobs,
    plot_prisma_centerline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare CH4 profiles along plume centerlines.",
    )
    parser.add_argument(
        "--raster",
        type=Path,
        help="Optional PRISMA GeoTIFF for a single analysis.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output figure for a single comparison.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Data directory used for automatic discovery.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "centerlines",
        help="Output directory used for automatic discovery.",
    )
    parser.add_argument("--num-points", type=int, default=100)
    parser.add_argument("--show-plot", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.raster:
        jobs = [(None, args.raster)]
    else:
        if args.output is not None:
            raise ValueError("--output requires --raster.")
        jobs = discover_centerline_jobs(args.data_dir)

    for event, raster in jobs:
        output_file = args.output
        if output_file is None:
            output_name = (
                "PRISMA_centerline_comparison.jpeg"
                if event is None
                else f"PRISMA_centerline_{event}.jpeg"
            )
            output_file = args.output_dir / output_name
        plot_prisma_centerline(
            raster,
            output_file,
            num_points=args.num_points,
            show=args.show_plot,
        )


if __name__ == "__main__":
    main()
