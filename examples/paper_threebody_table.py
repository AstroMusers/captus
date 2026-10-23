"""Plot selected values reported in the paper's Table 2."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from captus.paths import get_visuals_path


PAPER_TABLES = {
    "sun-jupiter": {
        "title": "Sun-Jupiter",
        "mass_msun": np.array([1e-8, 1e-4, 1e-3, 1e-2]),  # [M_sun]
        "capture_rate": np.array([0.27, 0.31, 0.76, 7.46]),  # [Myr^-1]
        "capture_rate_err": np.array([0.01, 0.01, 0.02, 0.20]),  # [Myr^-1]
        "neq_per_density": np.array([0.15, 0.17, 4.17, 41.79]),  # [AU^3]
        "neq_per_density_err": np.array([0.01, 0.01, 0.14, 1.34]),  # [AU^3]
        "neq_full": np.array([2.1e-11, 2.4e-15, 5.9e-15, 5.9e-15]),
        "neq_full_err": np.array([0.1e-11, 0.1e-15, 0.2e-15, 0.2e-15]),
        "neq_bound": np.array([2.0e-13, 4.5e-18, 1.3e-18, 1.5e-19]),
        "neq_bound_err": np.array([0.1e-13, 0.2e-18, 0.0, 0.0]),
    },
    "beta-pictoris": {
        "title": "Beta Pictoris",
        "mass_msun": np.array([1e-8, 1e-4, 1e-3, 1e-2]),  # [M_sun]
        "capture_rate": np.array([13, 14, 30, 173]),  # [Myr^-1]
        "capture_rate_err": np.array([1, 1, 3, 13]),  # [Myr^-1]
        "neq_per_density": np.array([5, 4, 122, 816]),  # [AU^3]
        "neq_per_density_err": np.array([1, 1, 14, 80]),  # [AU^3]
        "neq_full": np.array([7.5e-10, 5.8e-15, 1.7e-14, 1.1e-14]),
        "neq_full_err": np.array([2.1e-10, 1.6e-15, 0.2e-14, 0.1e-14]),
        "neq_bound": np.array([7.3e-12, 1.3e-18, 4.3e-19, 3.8e-20]),
        "neq_bound_err": np.array([2.0e-12, 0.3e-18, 0.5e-19, 0.4e-20]),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        choices=PAPER_TABLES,
        default="sun-jupiter",
        help="Paper configuration to plot (default: sun-jupiter)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Figure output path",
    )
    args = parser.parse_args()
    data = PAPER_TABLES[args.system]
    output_path = args.output or get_visuals_path() / f"captus_{args.system}_table2.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True, layout="constrained")
    axes[0].errorbar(
        data["mass_msun"],
        data["capture_rate"],
        yerr=data["capture_rate_err"],
        fmt="o",
        color="tab:blue",
        capsize=3,
    )
    axes[0].set_ylabel(r"$R_{\rm cap}$ [Myr$^{-1}$]")

    axes[1].errorbar(
        data["mass_msun"],
        data["neq_per_density"],
        yerr=data["neq_per_density_err"],
        fmt="s",
        color="tab:orange",
        capsize=3,
    )
    axes[1].set_ylabel(r"$N_{\rm eq}/n_\infty$ [AU$^3$]")

    axes[2].errorbar(
        data["mass_msun"],
        data["neq_full"],
        yerr=data["neq_full_err"],
        fmt="o",
        color="tab:green",
        capsize=3,
        label=r"$f_{\rm total}$",
    )
    axes[2].errorbar(
        data["mass_msun"],
        data["neq_bound"],
        yerr=data["neq_bound_err"],
        fmt="s",
        color="tab:red",
        capsize=3,
        label=r"$f_{\rm bound}$",
    )
    axes[2].set_ylabel(r"$N_{\rm eq}$")
    axes[2].set_xlabel(r"PBH mass [$M_\odot$]")
    axes[2].set_yscale("log")
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.set_xscale("log")
        axis.grid(False)
    figure.suptitle(f"{data['title']} capture results at 8 kpc")

    print(f"Writing to {output_path}...")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()