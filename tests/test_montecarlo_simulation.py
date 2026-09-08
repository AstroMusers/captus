import numpy as np

import captus.three_body_capture.simulations.montecarlo as mc_mod
from captus.three_body_capture.simulations.montecarlo import MonteCarloSimulation


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
