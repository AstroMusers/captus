import math
import sys
from types import ModuleType

import numpy as np
import pytest
import rebound
from astropy import constants as const
from astropy import units as u

from captus.three_body_capture.planetary_binary import (
    PlanetBody,
    ThreeBodyEncounter,
    _find_escaped_binary,
    _poisson_count_log_likelihood,
    compute_binary_planet_formation_rate,
    infer_binary_planet_rate_normalization_pcat,
    sample_equal_mass_unbound_encounter,
    simulate_three_body_capture,
)


@pytest.mark.parametrize("scale", [0.1, 2.0])
@pytest.mark.parametrize("observed, base", [(0, 0.1), (1, 0.5), (12, 7.0)])
def test_capture_count_likelihood_preserves_factorial_normalization(scale, observed, base):
    state = {"observed_count": observed, "base_expected_count": base}
    expected_count = scale * base
    expected = observed * math.log(expected_count) - expected_count - math.lgamma(observed + 1)
    assert _poisson_count_log_likelihood([scale], state) == pytest.approx(expected, rel=1e-13, abs=1e-14)


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf])
def test_capture_count_likelihood_retains_invalid_scale_rejection(scale):
    assert _poisson_count_log_likelihood([scale], {}) == -np.inf


def test_capture_count_likelihood_preserves_zero_rate_limits_after_underflow():
    state = {"observed_count": 0, "base_expected_count": 1e-200}
    assert _poisson_count_log_likelihood([1e-200], state) == 0.0
    state["observed_count"] = 1
    assert np.isneginf(_poisson_count_log_likelihood([1e-200], state))


def test_equal_mass_hard_binary_rate_matches_published_scaling():
    mass = 1.0 * u.M_jup
    density = 1.0e6 / u.pc**3
    dispersion = 0.3 * u.km / u.s
    result = compute_binary_planet_formation_rate(mass, density, dispersion)

    expected_hard_axis = const.G * mass / (3.0 * dispersion**2)
    expected_rate = (
        0.081
        * const.G**5
        * mass**5
        * density**3
        / dispersion**9
        * np.sqrt(10.0)
    )

    assert result["hard_semimajor_axis"].to(u.au).value == pytest.approx(
        expected_hard_axis.to_value(u.au)
    )
    assert result["formation_rate_density"].to_value(1 / (u.pc**3 * u.yr)) == pytest.approx(
        expected_rate.to_value(1 / (u.pc**3 * u.yr))
    )
    assert result["interaction_radius"].to_value(u.m) == pytest.approx(
        (20.0 * expected_hard_axis).to_value(u.m)
    )


def test_formation_rate_scalings_and_unit_equivalence():
    mass = 1.0 * u.M_jup
    density = 1.0e6 / u.pc**3
    dispersion = 0.3 * u.km / u.s
    base = compute_binary_planet_formation_rate(mass, density, dispersion)
    doubled_density = compute_binary_planet_formation_rate(mass, 2 * density, dispersion)
    doubled_dispersion = compute_binary_planet_formation_rate(mass, density, 2 * dispersion)
    half_axis = compute_binary_planet_formation_rate(
        mass, density, dispersion, maximum_semimajor_axis=base["hard_semimajor_axis"] / 2
    )
    equivalent_units = compute_binary_planet_formation_rate(
        mass.to(u.kg), density.to(1 / u.m**3), dispersion.to(u.m / u.s)
    )

    rate_unit = 1 / (u.pc**3 * u.yr)
    base_rate = base["formation_rate_density"].to_value(rate_unit)
    assert doubled_density["formation_rate_density"].to_value(rate_unit) == pytest.approx(
        8 * base_rate
    )
    assert doubled_dispersion["formation_rate_density"].to_value(rate_unit) == pytest.approx(
        base_rate / 2**9
    )
    assert half_axis["formation_rate_density"].to_value(rate_unit) == pytest.approx(
        base_rate / 2**3
    )
    assert equivalent_units["formation_rate_density"].to_value(rate_unit) == pytest.approx(
        base_rate
    )


def test_formation_rate_rejects_unsupported_inputs():
    mass = 1.0 * u.M_jup
    density = 1.0e6 / u.pc**3
    dispersion = 0.3 * u.km / u.s

    with pytest.raises(ValueError, match="chi1"):
        compute_binary_planet_formation_rate(mass, density, dispersion, chi1=5.0)
    with pytest.raises(ValueError, match="hard-binary scale"):
        compute_binary_planet_formation_rate(
            mass,
            density,
            dispersion,
            maximum_semimajor_axis=100 * u.au,
        )
    with pytest.raises(ValueError, match="positive scalar"):
        compute_binary_planet_formation_rate(0 * u.M_jup, density, dispersion)


def test_unbound_encounter_sampling_is_reproducible_and_rejects_bound_pairs():
    first = sample_equal_mass_unbound_encounter(
        1 * u.M_jup, 0.3 * u.km / u.s, seed=314, checkpoints=4
    )
    second = sample_equal_mass_unbound_encounter(
        1 * u.M_jup, 0.3 * u.km / u.s, seed=314, checkpoints=4
    )

    np.testing.assert_array_equal(
        first.positions.to_value(u.m), second.positions.to_value(u.m)
    )
    np.testing.assert_array_equal(
        first.velocities.to_value(u.m / u.s), second.velocities.to_value(u.m / u.s)
    )
    assert first.integration_time == second.integration_time
    positions = first.positions.to_value(u.m)
    velocities = first.velocities.to_value(u.m / u.s)
    mass = (1 * u.M_jup).to_value(u.kg)
    for first_index in range(3):
        for second_index in range(first_index + 1, 3):
            separation = np.linalg.norm(positions[second_index] - positions[first_index])
            relative_speed = np.linalg.norm(velocities[second_index] - velocities[first_index])
            specific_energy = (
                0.5 * relative_speed**2
                - const.G.value * 2 * mass / separation
            )
            assert specific_energy >= 0.0


def test_capture_classifier_requires_a_bound_pair_and_escaping_single():
    simulation = rebound.Simulation()
    simulation.units = ("AU", "yr", "Msun")
    binary_speed = np.sqrt(simulation.G * 2.0)
    simulation.add(m=1.0, x=-0.5, vy=-binary_speed / 2.0)
    simulation.add(m=1.0, x=0.5, vy=binary_speed / 2.0)
    simulation.add(m=0.1, x=100.0, vx=5.0)

    result = _find_escaped_binary(simulation, np.array([1.0, 1.0, 0.1]), 20.0)

    assert result is not None
    assert result[:3] == ((0, 1), 2, pytest.approx(1.0))
    assert result[3] == pytest.approx(0.0)


def test_direct_three_body_integration_tracks_conservation():
    encounter = sample_equal_mass_unbound_encounter(
        1 * u.M_jup, 0.3 * u.km / u.s, seed=42, checkpoints=8
    )

    result = simulate_three_body_capture(encounter)

    assert result.status in {"binary_capture", "unresolved"}
    assert result.captured is (result.status == "binary_capture")
    assert result.energy_error < 1e-8
    assert result.angular_momentum_error < 1e-8


def test_seeded_three_planet_encounter_forms_an_escaping_binary():
    encounter = sample_equal_mass_unbound_encounter(
        1 * u.M_jup,
        0.3 * u.km / u.s,
        seed=29,
        checkpoints=20,
        escape_separation_factor=5.0,
    )

    result = simulate_three_body_capture(encounter)

    assert result.captured
    assert result.binary_indices == (0, 1)
    assert result.single_index == 2
    assert result.semi_major_axis.to_value(u.au) == pytest.approx(280.4, rel=1e-3)
    assert result.eccentricity == pytest.approx(0.9173, rel=1e-3)


def test_direct_three_body_integration_reports_finite_size_collision():
    bodies = tuple(
        PlanetBody(f"planet_{index}", 1 * u.M_jup, 0.001 * u.au)
        for index in range(3)
    )
    encounter = ThreeBodyEncounter(
        bodies=bodies,
        positions=np.array([[-0.0011, 0.0, 0.0], [0.0011, 0.0, 0.0], [100.0, 0.0, 0.0]]) * u.au,
        velocities=np.array([[20.0, 0.0, 0.0], [-20.0, 0.0, 0.0], [0.0, 5.0, 0.0]]) * u.au / u.yr,
        integration_time=0.1 * u.yr,
        checkpoints=4,
    )

    result = simulate_three_body_capture(encounter)

    assert result.status == "collision"
    assert not result.captured


def test_rate_normalization_inference_delegates_to_pcat(monkeypatch):
    fake_pcat = ModuleType("pcat")
    fake_pcat.__path__ = []
    fake_fixed = ModuleType("pcat.fixed")
    calls = {}

    def sample_fixed_chains(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        calls["log_likelihood_value"] = args[1](np.array([1.0]), args[0])
        return np.array([[[0.8], [1.0]], [[1.2], [1.4]]]), np.zeros((2, 2))

    fake_fixed.sample_fixed_chains = sample_fixed_chains
    monkeypatch.setitem(sys.modules, "pcat", fake_pcat)
    monkeypatch.setitem(sys.modules, "pcat.fixed", fake_fixed)

    result = infer_binary_planet_rate_normalization_pcat(
        2 / (u.pc**3 * u.yr),
        3 * u.pc**3,
        4 * u.yr,
        observed_count=5,
        chain_count=2,
        sample_count=2,
        burn_count=1,
    )

    assert result["sampler"] == "PCAT"
    assert result["base_expected_count"] == pytest.approx(24.0)
    assert result["normalization_samples"].shape == (2, 2)
    assert result["expected_formation_count_samples"][0, 0] == pytest.approx(19.2)
    assert calls["args"][0]["observed_count"] == 5
    assert calls["args"][1] is not None
    assert calls["kwargs"]["boolcheckconv"] is False
    assert calls["log_likelihood_value"] == pytest.approx(
        5 * np.log(24.0) - 24.0 - math.lgamma(6)
    )