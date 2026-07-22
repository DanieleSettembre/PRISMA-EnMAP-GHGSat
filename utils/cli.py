import argparse
from pathlib import Path

from utils.metadata import discover_public_tifs
from utils.pipeline import run_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate and compare PRISMA, EnMAP and GHGSat plume metrics."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory containing PRISMA and EnMAP (20240911) GeoTIFFs.",
    )
    parser.add_argument(
        "--ghgsat-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "data_ghgsat_public.csv",
        help="Read-only CSV containing public GHGSat metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory for generated tables and figures.",
    )
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--percentile", type=float, default=5.0)
    parser.add_argument("--plot-mask", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tif_paths = discover_public_tifs(args.data_dir)
    run_analysis(
        tif_paths=tif_paths,
        ghgsat_public_csv=str(args.ghgsat_csv.resolve()),
        output_dir=str(args.output_dir.resolve()),
        plume_threshold_ppm_m=args.threshold,
        percentile_filter=args.percentile,
        plot_mask=args.plot_mask,
        create_plots=not args.no_plots,
        show_plots=args.show_plots,
    )
