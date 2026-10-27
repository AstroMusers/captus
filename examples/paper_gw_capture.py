"""Recompute a coarse neutron-star GW-capture grid from the paper."""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from matplotlib.colors import LogNorm

from captus.paths import get_visuals_path
from captus.gw_capture.simulation import GWCapture

plt.rcParams["text.usetex"] = False


def _plot_positive_grid(axis, masses_msun, velocities_kms, values, label):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & (values > 0)
    if not np.any(valid):
        axis.text(0.5, 0.5, "No positive values", ha="center", va="center")
        return

    lower = float(values[valid].min())
    upper = float(values[valid].max())
    if lower == upper:
        upper = lower * 10
    image = axis.pcolormesh(
        masses_msun,
        velocities_kms,
        np.ma.masked_where(~valid, values),
        shading="auto",
        cmap="magma",
        norm=LogNorm(vmin=lower, vmax=upper),
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"PBH mass [$M_\odot$]")
    axis.set_ylabel(r"$v_\infty$ [km s$^{-1}$]")
    axis.axhline(550, color="cyan", linestyle="--", linewidth=1)  # [km/s]
    axis.grid(False)
    colorbar = axis.figure.colorbar(image, ax=axis)
    colorbar.set_label(label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=8, help="Samples per axis")
    parser.add_argument(
        "--output",
        type=Path,
        default=get_visuals_path() / "captus_gw_capture_neutron_star.png",
        help="Figure output path",
    )
    args = parser.parse_args()
    if args.grid_size < 2:
        parser.error("--grid-size must be at least 2")

    masses_msun = np.logspace(-8, 2, args.grid_size)
    velocities_kms = np.logspace(0, 3, args.grid_size)
    model = GWCapture(
        {
            "name": "Typical_Neutron_Star",
            "M_star": 1.4 * const.M_sun.value,  # [kg]
            "R_star": 11e3,  # [m]
            "gridsize": args.grid_size,
            "M_pbh_values": masses_msun * const.M_sun.value,  # [kg]
            "v_inf_values": velocities_kms * 1e3,  # [m/s]
            "r": 8.0,  # [kpc]
        }
    )

    cross_section = model.get_capture_cross_section()
    coalescence_time = model.get_coalescence_time()
    grids = [cross_section, coalescence_time]
    titles = ["GW capture cross-section", "Post-capture lifetime"]
    labels = [r"$\sigma_{\rm cap}$ [AU$^2$]", "Coalescence time [yr]"]
    if os.environ.get("PBHBOUNDS_PATH"):
        grids.append(model.get_systems_in_eq())
        titles.append("Equilibrium abundance")
        labels.append(r"Bound $N_{\rm eq}$ contribution")

    figure, axes = plt.subplots(
        1, len(grids), figsize=(4.5 * len(grids), 4), layout="constrained"
    )
    for axis, grid, title, label in zip(axes, grids, titles, labels):
        _plot_positive_grid(axis, masses_msun, velocities_kms, grid, label)
        axis.set_title(title)
    figure.suptitle("Typical neutron star with 1.4 solar masses, 11 km radius, and r = 8 kpc")
    if len(grids) == 2:
        print("PBHBOUNDS_PATH is unset; skipping the equilibrium-abundance panel.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {args.output}...")
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()