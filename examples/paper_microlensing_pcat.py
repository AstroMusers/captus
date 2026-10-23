#!/usr/bin/env python3
"""Compare single- and binary-lens catalogs with PCAT on synthetic Roman data."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from pcat.diagnostics import posterior_convergence
from pcat.fixed import sample_fixed_chains


EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = EXAMPLE_DIRECTORY / "visuals" / "paper_microlensing_pcat"
SURVEY_DURATION = 72.0  # [day]
ROMAN_CADENCE = 15.0 / 1440.0  # [day]
PHOTOMETRIC_UNCERTAINTY = 0.01  # [relative flux]
INJECTED_BASE = np.array([36.0, 0.13, 24.0, 1.0, 0.0])
INJECTED_COMPANION = np.array([-3.0, np.log10(1.15), 68.0])
BASE_BOUNDS = np.array([[30.0, 42.0], [0.04, 0.5], [12.0, 45.0], [0.7, 1.3], [-0.2, 0.2]])
INJECTED_CARTESIAN = np.array([
    INJECTED_COMPANION[0],
    1.15 * np.cos(np.deg2rad(INJECTED_COMPANION[2])),
    1.15 * np.sin(np.deg2rad(INJECTED_COMPANION[2])),
])
COMPANION_BOUNDS = np.array([[-4.0, -2.0], [-1.5, 1.5], [-1.5, 1.5]])
MODEL_PRIOR_ODDS = 0.5


def _mulens_model_module():
    try:
        import MulensModel
    except ImportError as error:
        raise SystemExit("Install the optional inference dependencies with `pip install -e '.[inference]'`.") from error
    return MulensModel


def _predict_flux(mulens_model, time: np.ndarray, parameters: np.ndarray, binary: bool) -> np.ndarray:
    lens_parameters = {
        "t_0": float(INJECTED_BASE[0]),
        "u_0": float(INJECTED_BASE[1]),
        "t_E": float(INJECTED_BASE[2]),
    }
    if binary:
        separation = float(np.hypot(parameters[3], parameters[4]))
        if not 0.5 <= separation <= 1.5:
            return np.full(time.shape, np.nan)
        lens_parameters.update(
            q=10.0 ** float(parameters[2]),
            s=separation,
            alpha=float(np.rad2deg(np.arctan2(parameters[4], parameters[3])) % 360.0),
        )
    magnification = mulens_model.Model(lens_parameters).get_magnification(time)
    return float(parameters[0]) * magnification + float(parameters[1])


def _log_likelihood(parameters: np.ndarray, state: dict[str, object]) -> float:
    mulens_model = _mulens_model_module()
    values = np.asarray(parameters, dtype=float)
    binary = bool(state["binary"])
    try:
        model = _predict_flux(mulens_model, state["time"], values, binary)
    except (ArithmeticError, ValueError, RuntimeError):
        return -np.inf
    if not np.isfinite(model).all():
        return -np.inf
    residual = (state["flux"] - model) / state["uncertainty"]
    return float(-0.5 * np.sum(residual**2))


def _simulate_data(mulens_model, time: np.ndarray) -> dict[str, np.ndarray]:
    random = np.random.default_rng(814)
    noise = random.normal(0.0, PHOTOMETRIC_UNCERTAINTY, time.size)
    binary_parameters = np.concatenate((INJECTED_BASE[3:], INJECTED_CARTESIAN))
    single_parameters = INJECTED_BASE[3:].copy()
    return {
        "binary injection": _predict_flux(mulens_model, time, binary_parameters, True) + noise,
        "single-lens null": _predict_flux(mulens_model, time, single_parameters, False) + noise,
    }


def _display_output_path(path: Path) -> str:
    try:
        return str(path.relative_to(EXAMPLE_DIRECTORY.parent))
    except ValueError:
        return path.name


def _fit_model(
    time: np.ndarray,
    flux: np.ndarray,
    uncertainty: np.ndarray,
    *,
    binary: bool,
    chains: int,
    samples: int,
    burn: int,
    evidence_samples: int,
    seed: int,
) -> dict[str, object]:
    names = ("source_flux", "blend_flux")
    flux_bounds = BASE_BOUNDS[3:]
    minima = flux_bounds[:, 0]
    maxima = flux_bounds[:, 1]

    initial = INJECTED_BASE[3:].copy()
    if binary:
        names += ("log10_q", "x_companion", "y_companion")
        minima = np.concatenate((minima, COMPANION_BOUNDS[:, 0]))
        maxima = np.concatenate((maxima, COMPANION_BOUNDS[:, 1]))
        initial = np.concatenate((initial, INJECTED_CARTESIAN))
    proposal_scales = np.array([0.08, 0.08] + ([0.05, 0.05, 0.05] if binary else []))
    proposal_correlation = np.eye(len(names))
    proposal_correlation[0, 1] = proposal_correlation[1, 0] = -0.7
    state = {
        "time": time,
        "flux": flux,
        "uncertainty": uncertainty,
        "binary": binary,
    }
    chain, log_probability, evidence, sampler_state = sample_fixed_chains(
        state,
        _log_likelihood,
        None,
        names,
        ["self"] * len(names),
        minima,
        maxima,
        np.zeros(len(names)),
        np.ones(len(names)),
        np.repeat(initial[None, :], chains, axis=0),
        chains,
        samples,
        burn,
        typeverb=0,
        estimate_log_evidence=True,
        evidence_samples=evidence_samples,
        seed=seed,
        return_state=True,
        typeseed=seed,
        proposal_scales=proposal_scales,
        proposal_correlation=proposal_correlation,
        probpropblock=0.2,
        probdemc=0.25,
        booladaptstdp=True,
        boolcheckconv=False,
        boolmakeplot=False,
        boolmakeplotinit=False,
        booldiag=False,
    )
    convergence = posterior_convergence(sampler_state)
    return {
        "chain": chain,
        "log_probability": log_probability,
        "evidence": evidence,
        "convergence": {
            "converged": bool(convergence["converged"]),
            "max_rhat": float(convergence["max_rhat"]),
            "min_effective_sample_size": float(convergence["min_effective_sample_size"]),
        },
    }


def _compare_models(fits: dict[str, dict[str, object]]) -> dict[str, float]:
    single_log_evidence = float(fits["single"]["evidence"]["log_evidence"])
    binary_log_evidence = float(fits["binary"]["evidence"]["log_evidence"])
    log_odds = binary_log_evidence - single_log_evidence + np.log(MODEL_PRIOR_ODDS)
    probability_binary = float(expit(log_odds))
    return {
        "posterior_probability_binary": probability_binary,
        "posterior_probability_single": 1.0 - probability_binary,
        "log_bayes_factor_binary_to_single": binary_log_evidence - single_log_evidence,
    }


def _plot_results(
    mulens_model,
    time: np.ndarray,
    data: dict[str, np.ndarray],
    uncertainty: np.ndarray,
    fits: dict[str, dict[str, dict[str, object]]],
    comparisons: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), layout="constrained")
    cases = ("binary injection", "single-lens null")
    for axis, case, color in zip(axes[0], cases, ("#167D8D", "#C14A36")):
        axis.errorbar(
            time,
            data[case],
            yerr=uncertainty,
            fmt=".",
            ms=2,
            lw=0.4,
            color="#39454C",
            alpha=0.5,
            label="Synthetic photometry",
            rasterized=True,
        )
        for model_name, linestyle, model_color in (
            ("single", "--", "#75858B"),
            ("binary", "-", color),
        ):
            chain = fits[case][model_name]["chain"]
            parameters = np.median(chain.reshape(-1, chain.shape[-1]), axis=0)
            model = _predict_flux(mulens_model, time, parameters, model_name == "binary")
            axis.plot(time, model, color=model_color, ls=linestyle, lw=1.2, label=f"PCAT {model_name} lens")
        axis.set(xlim=(0.0, SURVEY_DURATION), xlabel="Time [day]", ylabel="Relative flux", title=case)
        axis.grid(False)
        axis.legend(frameon=True, facecolor="white", framealpha=1.0, fontsize=8)

    width = 0.34
    positions = np.arange(2)
    for index, (key, color, label) in enumerate((
        ("posterior_probability_single", "#75858B", "Single lens"),
        ("posterior_probability_binary", "#167D8D", "Binary lens"),
    )):
        values = [comparisons[case][key] for case in cases]
        axes[1, 0].bar(positions + (index - 0.5) * width, values, width=width, color=color, label=label)
    axes[1, 0].set_xticks(positions, ("Injected binary", "Single-lens null"))
    axes[1, 0].set(ylim=(0.0, 1.0), ylabel="Posterior model probability", title="PCAT evidence comparison")
    axes[1, 0].legend(frameon=True, facecolor="white", framealpha=1.0, fontsize=8)
    axes[1, 0].grid(False)

    for case, color in zip(cases, ("#167D8D", "#C14A36")):
        log_mass_ratio = fits[case]["binary"]["chain"].reshape(-1, 5)[:, 2]
        axes[1, 1].hist(log_mass_ratio, bins=28, histtype="step", lw=1.3, color=color, label=case)
    axes[1, 1].axvline(INJECTED_COMPANION[0], color="#D29B3D", ls="--", lw=1.2, label="Injected q")
    axes[1, 1].set_xlabel(r"$\log_{10} q$")
    axes[1, 1].set_ylabel("PCAT posterior samples")
    axes[1, 1].set_title("Binary-lens fit mass ratio")
    axes[1, 1].legend(frameon=True, facecolor="white", framealpha=1.0, fontsize=8)
    axes[1, 1].grid(False)

    figure.suptitle("Synthetic Roman-like binary microlensing: PCAT model comparison")
    figure.text(
        0.5,
        -0.012,
        "Simulated 72-day light curves; companion detection does not identify a primordial origin.",
        ha="center",
        fontsize=9,
    )
    print(f"Writing to {_display_output_path(output_path)}...")
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Use a short cadence grid and short chains.")
    parser.add_argument("--chains", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--burn", type=int)
    parser.add_argument("--evidence-samples", type=int)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT_DIRECTORY)
    arguments = parser.parse_args()

    chains = arguments.chains if arguments.chains is not None else (2 if arguments.smoke else 4)
    samples = arguments.samples if arguments.samples is not None else (400 if arguments.smoke else 4000)
    burn = arguments.burn if arguments.burn is not None else (200 if arguments.smoke else 2000)
    evidence_samples = (
        arguments.evidence_samples
        if arguments.evidence_samples is not None
        else (500 if arguments.smoke else 10000)
    )
    if chains < 2 or samples < 2 or not 0 <= burn < samples or evidence_samples < 2:
        parser.error("Require chains >= 2 and samples > burn >= 0.")

    mulens_model = _mulens_model_module()
    time = (
        np.linspace(0.0, SURVEY_DURATION, 121)
        if arguments.smoke
        else np.arange(0.0, SURVEY_DURATION, ROMAN_CADENCE)
    )
    data = _simulate_data(mulens_model, time)
    uncertainty = np.full(time.size, PHOTOMETRIC_UNCERTAINTY)
    fits: dict[str, dict[str, dict[str, object]]] = {}
    comparisons: dict[str, dict[str, float]] = {}
    for case_index, (case, flux) in enumerate(data.items()):
        fits[case] = {
            model_name: _fit_model(
                time,
                flux,
                uncertainty,
                binary=model_name == "binary",
                chains=chains,
                samples=samples,
                burn=burn,
                evidence_samples=evidence_samples,
                seed=814 + case_index * 10 + model_index,
            )
            for model_index, model_name in enumerate(("single", "binary"))
        }
        comparisons[case] = _compare_models(fits[case])

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    figure_path = arguments.output_directory / "paper_microlensing_pcat.png"
    data_path = arguments.output_directory / "paper_microlensing_pcat.npz"
    summary_path = arguments.output_directory / "paper_microlensing_pcat.json"
    _plot_results(mulens_model, time, data, uncertainty, fits, comparisons, figure_path)

    arrays: dict[str, np.ndarray] = {"time_day": time, "uncertainty": uncertainty}
    for case in data:
        key = case.replace(" ", "_")
        arrays[f"{key}_flux"] = data[case]
        for model_name in ("single", "binary"):
            fit = fits[case][model_name]
            arrays[f"{key}_{model_name}_chain"] = fit["chain"]
            arrays[f"{key}_{model_name}_log_probability"] = fit["log_probability"]
    print(f"Writing to {_display_output_path(data_path)}...")
    np.savez_compressed(data_path, **arrays)

    summary = {
        "analysis": "synthetic binary-microlensing model comparison",
        "photometry": "relative flux with independent Gaussian errors",
        "survey_duration_day": SURVEY_DURATION,
        "cadence_minutes": 15.0 if time.size > 1000 else None,
        "number_epochs": int(time.size),
        "photometric_uncertainty": PHOTOMETRIC_UNCERTAINTY,
        "fixed_event_trajectory": {
            "t_0_day": float(INJECTED_BASE[0]),
            "u_0": float(INJECTED_BASE[1]),
            "t_E_day": float(INJECTED_BASE[2]),
        },
        "model_comparison_conditioned_on_trajectory": True,
        "injected_mass_ratio": float(10.0 ** INJECTED_COMPANION[0]),
        "binary_prior": "log10(q) uniform [-4,-2], projected companion coordinates uniform [-1.5,1.5] with 0.5 <= s <= 1.5",
        "model_prior_odds_binary_to_single": MODEL_PRIOR_ODDS,
        "MulensModel_version": version("MulensModel"),
        "PCAT_sampler": "pcat.fixed.sample_fixed_chains",
        "PCAT_evidence": "pcat.diagnostics.estimate_evidence",
        "results": {
            case: {
                "posterior_model_probability": comparisons[case],
                "single_lens_log_evidence": fits[case]["single"]["evidence"],
                "binary_lens_log_evidence": fits[case]["binary"]["evidence"],
                "single_lens_convergence": fits[case]["single"]["convergence"],
                "binary_lens_convergence": fits[case]["binary"]["convergence"],
            }
            for case in data
        },
    }
    print(f"Writing to {_display_output_path(summary_path)}...")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for case in data:
        probability = comparisons[case]["posterior_probability_binary"]
        print(f"{case}: PCAT posterior P(binary lens)={probability:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())