import unittest

import numpy as np

from utils.emissions import (
    effective_wind_speed_ghgsat,
    effective_wind_speed_prisma_enmap,
)
from utils.metadata import event_key, sat_name_from_tif
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


if __name__ == "__main__":
    unittest.main()
