import os

import pandas as pd


def display_file_name(file_path: str) -> str:
    return os.path.basename(str(file_path)).replace("_400_2488", "")


def prepare_output_table(table: pd.DataFrame) -> pd.DataFrame:
    threshold_columns = [
        column for column in table.columns if "threshold" in str(column).lower()
    ]
    output = table.drop(columns=threshold_columns).copy()
    for column in output.select_dtypes(include=["object", "string"]).columns:
        output[column] = output[column].map(
            lambda value: value.replace("_400_2488", "")
            if isinstance(value, str)
            else value
        )
    return output


def write_flux_csv(
    results: pd.DataFrame,
    output_csv: str,
) -> None:
    multi_sensor = results[
        results["sat_group"].isin(["GHGSat", "PRS", "ENMAP"])
    ].copy()
    if multi_sensor.empty:
        return

    output = multi_sensor.rename(
        columns={
            "tif": "file",
            "U10 (m/s)": "wind_speed_m_s",
            "Ueff (m/s)": "effective_wind_speed_m_s",
            "IME (kg)": "IME_kg",
            "L sqrt(A) (m)": "L_m",
            "Q (kg/h)": "Q_kg_h",
        }
    )
    output = output[
        [
            "file",
            "wind_speed_m_s",
            "effective_wind_speed_m_s",
            "IME_kg",
            "L_m",
            "Q_kg_h",
        ]
    ].reset_index(drop=True)
    output["file"] = output["file"].map(display_file_name)
    output.to_csv(output_csv, index=False, encoding="utf-8-sig")


def write_analysis_outputs(
    results: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    prepare_output_table(results).to_csv(
        os.path.join(output_dir, "Flux_results_all_sensors.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    write_flux_csv(
        results,
        os.path.join(
            output_dir,
            "Flux_comparison_GHGSat_PRISMA_EnMAP.csv",
        ),
    )
    comparison_path = os.path.join(
        output_dir,
        "Flux_comparison_PRS-GHGSat.csv",
    )
    prepare_output_table(comparison).to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )
