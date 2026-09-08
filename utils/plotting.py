import os

import matplotlib.pyplot as plt
import numpy as np

from utils.comparison import format_regression_equation, rma_regression


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 16,
            "axes.labelsize": 22,
            "axes.titlesize": 24,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "xtick.major.size": 7,
            "ytick.major.size": 7,
            "xtick.minor.size": 4,
            "ytick.minor.size": 4,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _axis_maximum(x_values: np.ndarray, y_values: np.ndarray, step: float) -> float:
    maximum = max(np.nanmax(x_values), np.nanmax(y_values))
    if maximum > step:
        return float(np.ceil((maximum * 1.08) / step) * step)
    return float(maximum * 1.12)


def _plot_comparison(
    x_values,
    y_values,
    x_label: str,
    y_label: str,
    output_stem: str,
    output_dir: str,
    axis_step: float,
    x_name: str,
    y_name: str,
    annotation: str,
    show: bool,
) -> dict | None:
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[valid]
    y_array = y_array[valid]
    if len(x_array) <= 1:
        return None

    maximum = _axis_maximum(x_array, y_array, axis_step)
    slope, intercept, correlation = rma_regression(x_array, y_array)
    x_line = np.linspace(0, maximum, 300)
    r_squared = float(correlation**2)

    fig, axis = plt.subplots(figsize=(7.2, 6.8), dpi=300)
    axis.scatter(
        x_array,
        y_array,
        s=40,
        color="navy",
        edgecolor="navy",
        linewidth=0.5,
        alpha=0.95,
        label="Plume observations",
        zorder=3,
    )
    axis.plot(
        [0, maximum],
        [0, maximum],
        color="black",
        linewidth=2.4,
        label="1:1 line",
        zorder=2,
    )
    axis.plot(
        x_line,
        intercept + slope * x_line,
        color="red",
        linewidth=2.2,
        label="Regression line",
        zorder=2,
    )
    axis.set_xlim(0, maximum)
    axis.set_ylim(0, maximum)
    axis.set_xlabel(x_label, labelpad=14)
    axis.set_ylabel(y_label, labelpad=14)
    axis.minorticks_on()
    axis.grid(False)

    for spine in axis.spines.values():
        spine.set_linewidth(1.2)
    axis.tick_params(axis="both", which="major", direction="out")
    axis.tick_params(axis="both", which="minor", direction="out")
    axis.legend(
        loc="upper left",
        frameon=False,
        handlelength=2.7,
        borderpad=0.2,
        labelspacing=0.6,
    )
    axis.text(
        0.97,
        0.05,
        annotation.format(
            n=len(x_array),
            r=correlation,
            r2=r_squared,
            slope=slope,
        ),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"{output_stem}.png"),
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(output_dir, f"{output_stem}.pdf"),
        bbox_inches="tight",
    )
    if show:
        plt.show()
    else:
        plt.close(fig)

    equation = format_regression_equation(
        slope,
        intercept,
        x_name=x_name,
        y_name=y_name,
    )
    return {
        "n": len(x_array),
        "r": correlation,
        "r2": r_squared,
        "slope": slope,
        "intercept": intercept,
        "equation": equation,
    }


def plot_ime_comparison(wide_ime, output_dir: str, show: bool = True) -> dict | None:
    if len(wide_ime) <= 1:
        return None
    return _plot_comparison(
        x_values=wide_ime[("IME (kg)", "GHGSat")],
        y_values=wide_ime[("IME (kg)", "PRS")],
        x_label="GHGSat IME (kg)",
        y_label="PRISMA IME (kg)",
        output_stem="IME_GHGSat_vs_PRISMA",
        output_dir=output_dir,
        axis_step=10,
        x_name="GHGSat IME",
        y_name="PRISMA IME",
        annotation="$n$ = {n}\n$r$ = {r:.2f}\nslope = {slope:.2f}",
        show=show,
    )


def plot_flux_comparison(wide_q, output_dir: str, show: bool = True) -> dict | None:
    if len(wide_q) <= 1:
        return None
    return _plot_comparison(
        x_values=wide_q[("Q (kg/h)", "GHGSat")],
        y_values=wide_q[("Q (kg/h)", "PRS")],
        x_label="GHGSat emission rate (kg h$^{-1}$)",
        y_label="PRISMA emission rate (kg h$^{-1}$)",
        output_stem="Flux_GHGSat_vs_PRISMA",
        output_dir=output_dir,
        axis_step=100,
        x_name="GHGSat Q",
        y_name="PRISMA Q",
        annotation="$r$ = {r:.2f}\n$R^2$ = {r2:.2f}",
        show=show,
    )


def create_comparison_plots(
    wide_ime,
    wide_q,
    output_dir: str,
    show: bool = True,
) -> dict:
    configure_plot_style()
    ime_stats = plot_ime_comparison(wide_ime, output_dir, show=show)
    flux_stats = plot_flux_comparison(wide_q, output_dir, show=show)
    return {"ime": ime_stats, "flux": flux_stats}
