import numpy as np
import pytest
from astropy import units as u

import captus.three_body_capture.simulations.montecarlo as mc_mod
from captus.three_body_capture.simulations.montecarlo import MonteCarloSimulation
from captus.utils import calculations as calc


@pytest.mark.parametrize("angle", [0.0, np.pi / 2.0, -0.7, 2.0 * np.pi + 0.4])
def test_rodrigues_rotation_preserves_axis_angle_convention(angle):
    vector = np.array([2.0, -1.0, 4.0])
    axis = np.array([1.0, 2.0, -3.0])
    direction = axis / np.linalg.norm(axis)
    expected = (vector * np.cos(angle) + np.cross(direction, vector) * np.sin(angle)
                + direction * np.dot(direction, vector) * (1.0 - np.cos(angle)))
    actual = calc.rodrigues_rotate(vector, axis, angle)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(actual), np.linalg.norm(vector), rtol=1e-14)
    np.testing.assert_array_equal(vector, [2.0, -1.0, 4.0])
    np.testing.assert_array_equal(axis, [1.0, 2.0, -3.0])


def test_rodrigues_rotation_retains_zero_axis_rejection():
    with pytest.raises(ValueError, match="Zero vector"):
        calc.rodrigues_rotate([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.5)


def test_area_uniform_monte_carlo_cross_section_estimator():
    assert calc.annular_cross_section(1.0, 3.0) == pytest.approx(8.0 * np.pi)
    area = calc.annular_cross_section(1.0 * u.m, 0.003 * u.km)
    assert area.to_value(u.m**2) == pytest.approx(8.0 * np.pi)
    contribution, total = calc.capture_cross_section(b_mins=0.0, b_maxs=2e11, b_cap=1e11, n_sampled=10)
    expected = np.pi * (2e11 * u.m).to_value(u.au)**2 / 10.0
    assert contribution.isscalar
    assert contribution.to_value(u.au**2) == pytest.approx(expected)
    assert total.to_value(u.au**2) == pytest.approx(expected)


@pytest.mark.parametrize("confidence, z_score", [(0.68, 1.0), (0.95, 1.96), (0.9, None)])
def test_capture_estimate_and_error_preserve_trajectory_weights(confidence, z_score):
    from scipy.stats import norm

    minimum = np.array([0.2, 0.4, 0.6]) * 1e11  # [m]
    maximum = np.array([1.2, 1.7, 2.1]) * 1e11  # [m]
    captured = np.array([0.7, 0.9, 1.5]) * 1e11  # [m]
    sample_count = 20
    minimum_au = (minimum * u.m).to_value(u.au)  # [au]
    maximum_au = (maximum * u.m).to_value(u.au)  # [au]
    captured_au = (captured * u.m).to_value(u.au)  # [au]
    weights = np.pi * (maximum_au**2 - minimum_au**2)  # [au^2]
    expected = np.sum(weights) / sample_count  # [au^2]
    expected_error = np.sqrt((np.sum(weights**2) - np.sum(weights)**2 / sample_count)
                             / (sample_count * (sample_count - 1)))  # [au^2]
    contributions, total = calc.capture_cross_section(minimum, maximum, captured, sample_count)
    estimate, error, interval = calc.capture_cross_section_and_error(
        captured, minimum, maximum, sample_count, conf=confidence,
    )
    np.testing.assert_array_equal(contributions.to_value(u.au**2), weights / sample_count)
    assert total.to_value(u.au**2) == pytest.approx(expected, rel=1e-14)
    assert estimate == pytest.approx(expected, rel=1e-14)
    assert error == pytest.approx(expected_error, rel=1e-14)
    if z_score is None:
        z_score = norm.ppf(0.5 + confidence / 2.0)
    np.testing.assert_allclose(interval, [expected - z_score * expected_error, expected + z_score * expected_error], rtol=1e-14)


def test_capture_estimate_retains_empty_capture_result():
    contributions, total = calc.capture_cross_section([], [], [], 10)
    estimate, error, interval = calc.capture_cross_section_and_error([], [], [], 10)
    assert contributions.shape == (0,)
    assert total.to_value(u.au**2) == estimate == error == 0.0
    assert interval == (0.0, 0.0)


class DummyConfiguration:
    def __init__(self):
        self.system_params = {
            "mA": 1.0,
            "mB": 2.0,
            "mC": 3.0,
            "aB": 4.0,
            "rB": 5.0,
            "vB": 6.0,
            "eB": 0.1,
            "iB": 0.2,
            "epsilon": 0.3,
            "epsilon_adjust_coeff": 0.5,
            "seed_base": 42,
            "name": "dummy",
        }
        self.sim_params = {
            "importance_sampling": False,
            "sample_size": 1,
            "trials": 1,
            "max_trials": 1,
            "max_e": None,
            "max_execution_time": 60,
        }

    def get_system_param(self, all=False):
        return self.system_params if all else self.system_params["seed_base"]

    def get_simulation_param(self, key, all=False):
        return self.sim_params if all else self.sim_params[key]

    def get_save_dir_mc(self):
        return "/tmp/mc"


class DummyRNG:
    def uniform(self, low, high, size=None):
        if size is None:
            return float(low)
        return np.full(size, (low + high) / 2.0)


class ValueWrapper:
    def __init__(self, value):
        self.value = value


def test_initialization_sets_system(monkeypatch):
    monkeypatch.setattr(mc_mod.calcs, "standard_gravitational_parameter", lambda a, b, approx=False: 10.0)
    monkeypatch.setattr(mc_mod.calcs, "v_esc", lambda mu, r: 1.5)
    monkeypatch.setattr(mc_mod.calcs, "r_close", lambda epsilon, muA, muB, aB, approx=False: 0.25)
    monkeypatch.setattr(mc_mod.calcs, "hill_radius", lambda aB, mA, mB, eB: 1.0)
    monkeypatch.setattr(mc_mod.calcs, "schwarzchild_radius", lambda mC: 0.01)

    sim = MonteCarloSimulation(DummyConfiguration(), DummyRNG())

    assert sim.muA == 10.0
    assert sim.muB == 10.0
    assert sim.rClose == 0.25
    assert sim.aC == 3.75
    assert sim.vEscape == 1.5


def test_run_monte_carlo_simulation_builds_results(monkeypatch):
    monkeypatch.setattr(mc_mod.calcs, "standard_gravitational_parameter", lambda a, b, approx=False: 10.0)
    monkeypatch.setattr(mc_mod.calcs, "v_esc", lambda mu, r: 1.5)
    monkeypatch.setattr(mc_mod.calcs, "r_close", lambda epsilon, muA, muB, aB, approx=False: 0.25)
    monkeypatch.setattr(mc_mod.calcs, "hill_radius", lambda aB, mA, mB, eB: 1.0)
    monkeypatch.setattr(mc_mod.calcs, "schwarzchild_radius", lambda mC: 0.01)
    monkeypatch.setattr(mc_mod.calcs, "v_1_mag", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(mc_mod.calcs, "v_1_vec", lambda v, beta: np.array([1.0, 0.0, 0.0]))
    monkeypatch.setattr(mc_mod.calcs, "v_B_vec", lambda v, lam: np.array([0.0, 1.0, 0.0]))
    monkeypatch.setattr(mc_mod.calcs, "v_1_prime_vec", lambda v1, vB: np.array([1.0, 1.0, 0.0]))
    monkeypatch.setattr(mc_mod.calcs, "potential_energy", lambda *args, **kwargs: -1.0)
    monkeypatch.setattr(mc_mod.calcs, "b_min", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(mc_mod.calcs, "compute_b_max", lambda *args, **kwargs: 2.0)
    monkeypatch.setattr(mc_mod.calcs, "v_2_prime_vec", lambda *args, **kwargs: np.array([0.5, 0.5, 0.0]))
    monkeypatch.setattr(mc_mod.calcs, "v_2_vec", lambda *args, **kwargs: np.array([0.25, 0.25, 0.0]))
    monkeypatch.setattr(mc_mod.calcs, "v_2_mag", lambda v: 0.5)
    monkeypatch.setattr(mc_mod.calcs, "r_AB_vec", lambda *args, **kwargs: np.array([1.0, 0.0, 0.0]))
    monkeypatch.setattr(mc_mod.calcs, "b_unit_vector", lambda *args, **kwargs: np.array([1.0, 0.0, 0.0]))
    monkeypatch.setattr(mc_mod.calcs, "exit_point_from_scatter", lambda *args, **kwargs: np.array([0.0, 0.0, 0.0]))
    monkeypatch.setattr(mc_mod.calcs, "specific_L2", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(mc_mod.calcs, "a_e", lambda *args, **kwargs: (1.0, 0.5))
    monkeypatch.setattr(mc_mod.calcs, "capture_cross_section", lambda *args, **kwargs: np.array([ValueWrapper(3.0)], dtype=object))
    monkeypatch.setattr(mc_mod.calcs, "collision_cross_section", lambda *args, **kwargs: 4.0)
    monkeypatch.setattr(mc_mod.calcs, "capture_cross_section_MC", lambda *args, **kwargs: ValueWrapper(5.0))
    monkeypatch.setattr(mc_mod.MonteCarloSimulation, "_save_mc_results", lambda *args, **kwargs: None)

    sim = MonteCarloSimulation(DummyConfiguration(), DummyRNG())
    sim.run_monte_carlo_simulation(1000.0)

    assert hasattr(sim, "mc_results")
    assert sim.mc_results["v_inf"] == 1000.0
    assert sim.mc_results["sample_number"] == 1
    assert sim.mc_results["n_captured"] == 1
    assert sim.mc_results["cap_a_au"].shape == (1,)
