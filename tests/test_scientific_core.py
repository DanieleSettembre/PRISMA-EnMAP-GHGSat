import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils.emissions import (
    effective_wind_speed_ghgsat,
    effective_wind_speed_prisma_enmap,
)
from utils.metadata import event_key, sat_name_from_tif
from utils.outputs import write_analysis_outputs
from utils.pipeline import process_plume_tif
from utils.wind import (
    acquisition_midpoint_utc,
    meteorological_wind_direction_from,
    nearest_era5_datetime_utc,
)


class EffectiveWindSpeedTests(unittest.TestCase):
    def test_prisma_enmap_formula(self):
        self.assertAlmostEqual(
            effective_wind_speed_prisma_enmap(8.0),
            0.34 * 8.0 + 0.44,
        )

    def test_ghgsat_formula(self):
        self.assertAlmostEqual(
            effective_wind_speed_ghgsat(8.0),
            0.9 * np.log(8.0) + 0.6,
        )


class WindDirectionTests(unittest.TestCase):
    def test_cardinal_directions_from(self):
        cases = [
            (0.0, -1.0, 0.0),
            (-1.0, 0.0, 90.0),
            (0.0, 1.0, 180.0),
            (1.0, 0.0, 270.0),
        ]
        for u10, v10, expected in cases:
            with self.subTest(u10=u10, v10=v10):
                self.assertAlmostEqual(
                    float(meteorological_wind_direction_from(u10, v10)),
                    expected,
                )


class MetadataTests(unittest.TestCase):
    def test_event_and_sensor_from_path(self):
        path = (
            "data/20240911/Plume1/"
            "PRS_L1_STD_OFFL_20240911071147_20240911071151_Plume1.tif"
        )
        self.assertEqual(event_key(path), "20240911_PLUME1")
        self.assertEqual(sat_name_from_tif(path), "PRS")

    def test_files_round_to_the_same_era5_hour(self):
        prisma = "PRS_20240911071147_20240911071151_Plume1.tif"
        ghgsat = "GHGSat_20240911063311_20240911063311_Plume1.tif"
        self.assertEqual(
            nearest_era5_datetime_utc(acquisition_midpoint_utc(prisma)),
            nearest_era5_datetime_utc(acquisition_midpoint_utc(ghgsat)),
        )


class PipelineTests(unittest.TestCase):
    @patch("utils.pipeline.acquisition_midpoint_utc")
    @patch("utils.pipeline.windspeed")
    @patch("utils.pipeline.emission_rate_from_plume_tif")
    def test_prisma_result_is_populated_without_threshold_metric(
        self,
        emission_rate,
        windspeed,
        acquisition_midpoint,
    ):
        acquisition_midpoint.return_value = datetime(
            2024, 9, 11, 7, 11, 49, tzinfo=timezone.utc
        )
        windspeed.return_value = (4.0, "2024/09/11 07:11:49")
        emission_rate.return_value = {
            "IME_kg": 100.0,
            "L_m": 200.0,
            "A_m2": 40000.0,
            "U10_mps": 4.0,
            "Ueff_mps": 1.8,
            "Ueff_method": "test",
            "Q_kg_s": 0.9,
            "Q_kg_h": 3240.0,
            "n_plume_pixels": 20,
        }

        result = process_plume_tif(
            "PRS_20240911071147_20240911071151_Plume1.tif",
            plume_threshold_ppm_m=0.0,
            percentile_filter=5.0,
            plot_mask=False,
            wind_by_event_hour={},
        )

        self.assertEqual(result["IME (kg)"], 100.0)
        self.assertEqual(result["Q (kg/h)"], 3240.0)
        self.assertNotIn("error", result)
        self.assertNotIn("Mask threshold (ppm m)", result)


class OutputTests(unittest.TestCase):
    def test_outputs_are_csv_and_exclude_threshold_columns(self):
        results = pd.DataFrame(
            [
                {
                    "sat_group": "PRS",
                    "tif": "plume_400_2488.tif",
                    "U10 (m/s)": 4.0,
                    "Ueff (m/s)": 1.8,
                    "IME (kg)": 100.0,
                    "L sqrt(A) (m)": 200.0,
                    "Q (kg/h)": 3240.0,
                    "mask_threshold_ppm_m": 5.0,
                }
            ]
        )
        comparison = pd.DataFrame(
            [
                {
                    "event": "20240911_PLUME1",
                    "PRS_tif": "plume_400_2488.tif",
                    "Mask threshold": 5.0,
                }
            ]
        )

        with TemporaryDirectory() as output_dir:
            write_analysis_outputs(results, comparison, output_dir)
            expected_files = {
                "Flux_results_all_sensors.csv",
                "Flux_comparison_GHGSat_PRISMA_EnMAP.csv",
                "Flux_comparison_PRS-GHGSat.csv",
            }
            self.assertEqual(
                {path.name for path in Path(output_dir).iterdir()},
                expected_files,
            )

            for file_name in expected_files:
                table = pd.read_csv(Path(output_dir) / file_name)
                self.assertFalse(
                    any("threshold" in column.lower() for column in table.columns)
                )
                self.assertFalse(
                    table.astype(str)
                    .apply(lambda column: column.str.contains("_400_2488").any())
                    .any()
                )


if __name__ == "__main__":
    unittest.main()
