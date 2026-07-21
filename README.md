# PRISMA-EnMAP-GHGSat comparison

This repository calculates integrated mass enhancement (IME), characteristic
plume length, effective wind speed, and methane emission rate from PRISMA and
EnMAP plume GeoTIFFs. Public GHGSat metrics are read from a CSV because the
original GHGSat imagery cannot be redistributed.

## Project structure

```text
scripts/IME.py                 Command-line entry point
utils/                         Scientific and processing modules
data/YYYYMMDD/                 Public PRISMA and EnMAP plume GeoTIFFs
data/data_ghgsat_public.csv    Read-only public GHGSat metrics
outputs/                       Generated tables and figures
```

ERA5-Land cache files are created under `data/YYYYMMDD/ERA5/` and are excluded
from Git.

## Environment

Create and activate the Conda environment:

```powershell
conda env create -f environment.yml
conda activate prisma-enmap-ghgsat
```

ERA5-Land downloads use the Climate Data Store API. Configure CDS credentials
before the first run.

## Run

From the repository root:

```powershell
python scripts/IME.py
```

Useful options:

```powershell
python scripts/IME.py --help
python scripts/IME.py --no-plots
python scripts/IME.py --show-plots
python scripts/IME.py --data-dir path\to\data --output-dir path\to\outputs
```

The script recursively discovers PRISMA and EnMAP GeoTIFFs under `data`. The
public GHGSat CSV is an input file and is never overwritten.

## Method

IME is calculated from the masked column enhancement and pixel area. The
characteristic plume length is:

```text
L = sqrt(A_M)
```

For PRISMA and EnMAP:

```text
Ueff = 0.34 * U10 + 0.44
```

References:

1. Guanter et al., 2021: https://www.sciencedirect.com/science/article/pii/S0034425721003916
2. Rogers et al., 2024: https://ieeexplore.ieee.org/abstract/document/10387469

For GHGSat:

```text
Ueff = 0.9 * ln(U10) + 0.6
```

Reference: Varon et al., 2018: https://amt.copernicus.org/articles/11/5673/2018/

The emission rate is:

```text
Q = IME * Ueff / L
```

Acquisition timestamps in the filenames are interpreted as UTC
(`LOCAL_UTC_OFFSET_HOURS = 0`). ERA5-Land is selected at the nearest hourly
timestamp. Wind direction is the meteorological direction the wind comes from,
measured clockwise from North.

## Outputs

The compact multi-sensor CSV contains only:

```text
file, wind_speed_m_s, effective_wind_speed_m_s, IME_kg, L_m, Q_kg_h
```

Additional Excel tables and PRISMA-GHGSat comparison figures are written to
`outputs/`.
