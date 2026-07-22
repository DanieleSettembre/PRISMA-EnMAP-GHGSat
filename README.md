# PRISMA-EnMAP-GHGSat comparison

This project calculates and compares methane plume metrics from PRISMA and
EnMAP GeoTIFFs and public GHGSat data. ERA5-Land provides the wind data.

Follow the steps below from top to bottom. Commands are provided for Windows
PowerShell and macOS Terminal.

## 1. Install the required software

Python 3.12, Git, Git LFS, and GitHub CLI are required.

### Windows PowerShell

Open PowerShell and run:

```powershell
winget install --exact --id Python.Python.3.12
winget install --exact --id Git.Git
winget install --exact --id GitHub.GitLFS
winget install --exact --id GitHub.cli
```

Close and reopen PowerShell, then verify the installation:

```powershell
python --version
git --version
git lfs version
gh --version
git lfs install
```

`python --version` must report Python 3.12.x.

### macOS Terminal

Install the Xcode command-line tools:

```bash
xcode-select --install
```

Install [Homebrew](https://brew.sh/) if `brew --version` is not available:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the PATH instructions printed by the Homebrew installer. Then run:

```bash
brew install python@3.12 git git-lfs gh
python3.12 --version
git --version
git lfs version
gh --version
git lfs install
```

`python3.12 --version` must report Python 3.12.x.

## 2. Clone the repository

Authenticate with GitHub:

```bash
gh auth login --web --git-protocol https
gh auth status
```

Choose a parent directory, then run:

```bash
gh repo clone DanieleSettembre/PRISMA-EnMAP-GHGSat
cd PRISMA-EnMAP-GHGSat
git lfs pull
git lfs ls-files
```

Complete the browser login when requested. `git lfs ls-files` must list the
GeoTIFFs. If the TIFFs contain only small text pointers, run `git lfs pull`
again.

## 3. Create the Python environment

The environment is stored inside `.venv` and does not require Conda.

### Windows PowerShell

Run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verify the environment:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
```

### macOS Terminal

Run from the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Verify the environment:

```bash
.venv/bin/python --version
.venv/bin/python -m pip check
```

`pip check` should print `No broken requirements found`.

## 4. Configure Climate Data Store access

ERA5-Land downloads require personal CDS credentials. Credentials are stored
outside the repository and must never be shared or committed.

1. Register or log in at the
   [Climate Data Store](https://cds.climate.copernicus.eu/).
2. Open the [CDS API page](https://cds.climate.copernicus.eu/how-to-api).
3. Copy the personal access token shown on that page.
4. Create the credential file described below.
5. Open the [ERA5-Land download page](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download)
   and accept the dataset terms.

The credential file must contain exactly these fields:

```yaml
url: https://cds.climate.copernicus.eu/api
key: <PERSONAL-ACCESS-TOKEN>
```

Replace `<PERSONAL-ACCESS-TOKEN>` with the real token.

### Windows PowerShell

Create the file in the Windows user profile:

```powershell
notepad "$env:USERPROFILE\.cdsapirc"
```

Paste the two YAML lines, replace the token, save, and close Notepad. The final
path is normally `C:\Users\USERNAME\.cdsapirc`.

### macOS Terminal

Create the file in the home directory:

```bash
nano ~/.cdsapirc
```

Paste the two YAML lines, replace the token, save with `Ctrl+O`, press Enter,
and exit with `Ctrl+X`. Protect the file:

```bash
chmod 600 ~/.cdsapirc
```

Do not create `.cdsapirc` inside the repository.

## 5. Run the complete analysis

The first complete run tests the CDS credentials and downloads missing
ERA5-Land data.

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe scripts\Flux_comparison.py --no-plots
```

### macOS Terminal

```bash
.venv/bin/python scripts/Flux_comparison.py --no-plots
```

The analysis:

1. The program finds 15 public PRISMA/EnMAP GeoTIFFs.
2. Missing ERA5-Land ZIP files are downloaded.
3. ERA5 files are cached under `data/YYYYMMDD/ERA5/`.
4. Public GHGSat metrics are read from `data/data_ghgsat_public.csv`.
5. Results are written under `outputs/`.

After the command finishes, check the generated files described in Step 9.

## 6. Generate figures

Create figures without opening interactive windows:

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\Flux_comparison.py
```

macOS Terminal:

```bash
.venv/bin/python scripts/Flux_comparison.py
```

Create figures and display them during the run:

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\Flux_comparison.py --show-plots
```

macOS Terminal:

```bash
.venv/bin/python scripts/Flux_comparison.py --show-plots
```

Use `--help` to list all command-line options.

## 7. Compare plume centerlines

All plume folders are processed automatically using only the PRISMA raster and
its centerline. Profile direction and sampling density are adjusted
automatically.

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\plume_centerline_analysis.py
```

macOS Terminal:

```bash
.venv/bin/python scripts/plume_centerline_analysis.py
```

Figures are saved under `outputs/centerlines/`. Use `--help` to list the
optional controls for a single comparison.

## 8. Coregister PRISMA images

This script coregisters the three-band PRISMA rasters to the EnMAP reference
using SSIM.

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\Coregistration_PRS.py
```

macOS Terminal:

```bash
.venv/bin/python scripts/Coregistration_PRS.py
```

## 9. Check the outputs

The compact CSV is:

```text
outputs/Flux_comparison_GHGSat_PRISMA_EnMAP.csv
```

It contains:

```text
file, wind_speed_m_s, effective_wind_speed_m_s, IME_kg, L_m, Q_kg_h
```

The complete results table is:

```text
outputs/Flux_results_all_sensors.csv
```

The PRISMA-GHGSat comparison table is:

```text
outputs/Flux_comparison_PRS-GHGSat.csv
```

The `outputs/` directory also contains the comparison figures.
Outputs and ERA5 cache files are excluded from Git.

## 10. Troubleshooting

### CDS authentication error or HTTP 401

Check that `.cdsapirc` is in the user home directory, uses the current URL, and
contains the personal token without quotes.

### CDS licence error or HTTP 403

Log in to CDS and accept the ERA5-Land dataset terms.

## 11. Data and method

```text
data/YYYYMMDD/                 PRISMA and EnMAP plume GeoTIFFs
data/YYYYMMDD/Plume*/plume_centerline_PRS.*  PRISMA centerlines
data/data_ghgsat_public.csv    Public GHGSat metrics (read-only)
coregistration_data/           Three-band PRISMA and EnMAP RGB inputs
outputs/                       Generated tables and figures
```

Private GHGSat imagery is not distributed. The public GHGSat CSV is read-only
and is never overwritten.

```text
L = sqrt(A_M)
Q = IME * Ueff / L

PRISMA and EnMAP: Ueff = 0.34 * U10 + 0.44
GHGSat:           Ueff = 0.9 * ln(U10) + 0.6
```

References: [Guanter et al., 2021](https://www.sciencedirect.com/science/article/pii/S0034425721003916),
[Rogers et al., 2024](https://ieeexplore.ieee.org/abstract/document/10387469), and
[Varon et al., 2018](https://amt.copernicus.org/articles/11/5673/2018/).

Filename timestamps are interpreted as UTC. ERA5-Land is selected at the
nearest hour. Wind direction is measured clockwise from North and indicates
where the wind comes from.
