import numpy as np
import pandas as pd


PAIR_VALUES = [
    "IME (kg)",
    "Q (kg/s)",
    "Q (kg/h)",
    "L sqrt(A) (m)",
    "A plume mask (m2)",
    "U10 (m/s)",
    "Ueff (m/s)",
]


def paired_table(
    table: pd.DataFrame,
    required_cols: list[tuple[str, str]],
) -> pd.DataFrame:
    missing = [column for column in required_cols if column not in table.columns]
    if missing:
        return table.iloc[0:0].copy()
    return table.dropna(subset=required_cols, how="any")


def build_paired_tables(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_results = results[results["sat_group"].isin(["PRS", "GHGSat"])].copy()
    wide = pair_results.pivot_table(
        index="event",
        columns="sat_group",
        values=PAIR_VALUES,
        aggfunc="mean",
    )
    wide_ime = paired_table(
        wide,
        [("IME (kg)", "PRS"), ("IME (kg)", "GHGSat")],
    )
    wide_q = paired_table(
        wide,
        [("Q (kg/h)", "PRS"), ("Q (kg/h)", "GHGSat")],
    )
    return pair_results, wide_ime, wide_q


def rma_regression(x, y) -> tuple[float, float, float]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[valid]
    y_values = y_values[valid]

    correlation = np.corrcoef(x_values, y_values)[0, 1]
    slope = (
        np.sign(correlation)
        * np.std(y_values, ddof=1)
        / np.std(x_values, ddof=1)
    )
    intercept = np.mean(y_values) - slope * np.mean(x_values)
    return float(slope), float(intercept), float(correlation)


def format_regression_equation(
    slope: float,
    intercept: float,
    x_name: str = "x",
    y_name: str = "y",
) -> str:
    sign = "+" if intercept >= 0 else "-"
    return f"{y_name} = {slope:.4f} {x_name} {sign} {abs(intercept):.4f}"


def flatten_column_name(column) -> str:
    if isinstance(column, tuple):
        parts = [
            str(part)
            for part in column
            if part is not None and str(part) != ""
        ]
        return "_".join(parts)
    return str(column)


def build_detailed_comparison_table(
    pair_results: pd.DataFrame,
    wide_ime: pd.DataFrame,
) -> pd.DataFrame:
    output = wide_ime.copy().reset_index()
    output.columns = [flatten_column_name(column) for column in output.columns]

    if "event" not in output.columns:
        event_columns = [
            column for column in output.columns if str(column).startswith("event")
        ]
        if not event_columns:
            raise KeyError(
                "Column 'event' not found after reset_index(). "
                f"Available columns: {list(output.columns)}"
            )
        output = output.rename(columns={event_columns[0]: "event"})

    output = output.rename(
        columns={
            "IME (kg)_PRS": "PRS_IME_kg",
            "IME (kg)_GHGSat": "GHGSat_IME_kg",
            "Q (kg/s)_PRS": "PRS_Q_kg_s",
            "Q (kg/s)_GHGSat": "GHGSat_Q_kg_s",
            "Q (kg/h)_PRS": "PRS_Q_kg_h",
            "Q (kg/h)_GHGSat": "GHGSat_Q_kg_h",
            "L sqrt(A) (m)_PRS": "PRS_L_sqrtA_m",
            "L sqrt(A) (m)_GHGSat": "GHGSat_L_sqrtA_m",
            "A plume mask (m2)_PRS": "PRS_A_mask_m2",
            "A plume mask (m2)_GHGSat": "GHGSat_A_mask_m2",
            "U10 (m/s)_PRS": "PRS_U10_m_s",
            "U10 (m/s)_GHGSat": "GHGSat_U10_m_s",
            "Ueff (m/s)_PRS": "PRS_Ueff_m_s",
            "Ueff (m/s)_GHGSat": "GHGSat_Ueff_m_s",
        }
    )
    output["date"] = output["event"].astype(str).str.slice(0, 8)
    output["plume"] = output["event"].astype(str).str.split("_", n=1).str[1]

    tif_map = (
        pair_results.groupby(["event", "sat_group"])["tif"]
        .first()
        .unstack("sat_group")
        .rename(columns={"PRS": "PRS_tif", "GHGSat": "GHGSat_tif"})
        .reset_index()
    )
    method_map = (
        pair_results.groupby(["event", "sat_group"])["Ueff method"]
        .first()
        .unstack("sat_group")
        .rename(
            columns={
                "PRS": "PRS_Ueff_method",
                "GHGSat": "GHGSat_Ueff_method",
            }
        )
        .reset_index()
    )
    output = output.merge(tif_map, on="event", how="left")
    output = output.merge(method_map, on="event", how="left")

    metric_pairs = [
        ("PRS_Q_kg_h", "GHGSat_Q_kg_h", "Q_diff_PRS_minus_GHGSat_kg_h", "Q_ratio_PRS_over_GHGSat"),
        ("PRS_Q_kg_s", "GHGSat_Q_kg_s", "Q_diff_PRS_minus_GHGSat_kg_s", None),
        ("PRS_IME_kg", "GHGSat_IME_kg", "IME_diff_PRS_minus_GHGSat_kg", "IME_ratio_PRS_over_GHGSat"),
        ("PRS_L_sqrtA_m", "GHGSat_L_sqrtA_m", "L_diff_PRS_minus_GHGSat_m", "L_ratio_PRS_over_GHGSat"),
    ]
    for left, right, difference, ratio in metric_pairs:
        if {left, right}.issubset(output.columns):
            output[difference] = output[left] - output[right]
            if ratio:
                output[ratio] = output[left] / output[right]

    column_order = [
        "date",
        "plume",
        "event",
        "PRS_tif",
        "GHGSat_tif",
        "PRS_IME_kg",
        "GHGSat_IME_kg",
        "IME_diff_PRS_minus_GHGSat_kg",
        "IME_ratio_PRS_over_GHGSat",
        "PRS_Q_kg_s",
        "GHGSat_Q_kg_s",
        "Q_diff_PRS_minus_GHGSat_kg_s",
        "PRS_Q_kg_h",
        "GHGSat_Q_kg_h",
        "Q_diff_PRS_minus_GHGSat_kg_h",
        "Q_ratio_PRS_over_GHGSat",
        "PRS_L_sqrtA_m",
        "GHGSat_L_sqrtA_m",
        "L_diff_PRS_minus_GHGSat_m",
        "L_ratio_PRS_over_GHGSat",
        "PRS_A_mask_m2",
        "GHGSat_A_mask_m2",
        "PRS_U10_m_s",
        "GHGSat_U10_m_s",
        "PRS_Ueff_m_s",
        "GHGSat_Ueff_m_s",
        "PRS_Ueff_method",
        "GHGSat_Ueff_method",
    ]
    available_columns = [
        column for column in column_order if column in output.columns
    ]
    return (
        output[available_columns]
        .sort_values(["date", "plume"])
        .reset_index(drop=True)
    )
