import numpy as np

import captus.three_body_capture.simulations.rebound as rb_mod
from captus.three_body_capture.simulations.rebound import OrbitalSimulation


class DummyConfiguration:
    def __init__(self, save_dir_rebound="/tmp/rebound", save_dir_plots="/tmp/plots"):
        self._save_dir_rebound = save_dir_rebound
        self._save_dir_plots = save_dir_plots
        self.system_params = {
            "mA": 1.0,
            "mB": 2.0,
            "mC": 3.0,
            "aB": 4.0,
            "rA": 5.0,
            "rB": 6.0,
            "rC": 7.0,
            "eB": 0.1,
            "iB": 0.2,
            "epsilon": 0.3,
            "name": "dummy",
            "seed_base": 123,
        }
        self.sim_params = {"max_execution_time": 60}

    def get_system_param(self, all=False):
        return self.system_params if all else self.system_params["seed_base"]

    def get_simulation_param(self, all=False):
        return self.sim_params if all else self.sim_params["max_execution_time"]

    def get_save_dir_rebound(self):
        return self._save_dir_rebound

    def get_save_dir_plots(self):
        return self._save_dir_plots


class DummyRNG:
    pass


class DummyParticles:
    def __init__(self):
        self._orbit = type("Orbit", (), {"a": 1.0, "e": 0.1, "P": 2.0})()
        self.m = 1.0
        self.vx = self.vy = self.vz = 0.0
        self.x = self.y = self.z = 0.0

    def orbit(self, primary=None):
        return self._orbit

    def __pow__(self, other):
        return 1.0


class DummySimulation:
    def __init__(self):
        self.units = None
        self.integrator = None
        self.G = 1.0
        self.dt = 0.1
        self.t = 0.0
        self.particles = [DummyParticles(), DummyParticles(), DummyParticles()]
        self.collision = None
        self.collision_resolve = None

    def add(self, **kwargs):
        return None

    def move_to_com(self):
        return None

    def energy(self):
        return -1.0

    def integrate(self, time, exact_finish_time=0):
        self.t = time


def test_initialization_sets_system(monkeypatch):
    monkeypatch.setattr(rb_mod.calcs, "r_close", lambda epsilon, mA, mB, aB: 0.25)

    sim = OrbitalSimulation(DummyConfiguration(), DummyRNG())

    assert sim.name == "dummy"
    assert sim.seed_base == 123
    assert sim.rClose == 0.25
    assert sim.max_execution_time == 60


def test_save_and_check_result_helpers(monkeypatch, tmp_path):
    config = DummyConfiguration(save_dir_rebound=str(tmp_path / "rebound"), save_dir_plots=str(tmp_path / "plots"))
    monkeypatch.setattr(rb_mod.calcs, "r_close", lambda epsilon, mA, mB, aB: 0.25)
    sim = OrbitalSimulation(config, DummyRNG())

    saved = {}
    monkeypatch.setattr(rb_mod.np, "savez", lambda path, **kwargs: saved.update({"path": path, "data": kwargs}))

    sim._save_result(10.0, [0, 1000.0, {}, [], 1, 2, 3, "dist", 4, 5, 6, "flag", np.array([]), np.array([]), np.array([]), np.array([]), None, "version"])
    assert "path" in saved
    assert saved["path"].endswith("sim_0_123.npz")
    assert sim._check_result_exists(10.0, 0) is False


def test_run_orbital_integration_with_fake_rebound(monkeypatch):
    monkeypatch.setattr(rb_mod.calcs, "r_close", lambda epsilon, mA, mB, aB: 0.25)
    monkeypatch.setattr(rb_mod, "get_script_version", lambda: "version")
    monkeypatch.setattr(rb_mod.exc, "EscapeError", RuntimeError)
    monkeypatch.setattr(rb_mod.exc, "EnergyError", RuntimeError)
    monkeypatch.setattr(rb_mod.exc, "MaxIntegrationTimeError", RuntimeError)
    monkeypatch.setattr(rb_mod.exc, "CollisionManualError", RuntimeError)

    class FakeReboundModule:
        class Collision(Exception):
            pass

        class OrbitPlotSetError(Exception):
            pass

        class Simulation:
            def __init__(self):
                self.units = None
                self.integrator = None
                self.G = 1.0
                self.dt = 0.1
                self.t = 0.0
                self.collision = None
                self.collision_resolve = None
                self.particles = [DummyParticles(), DummyParticles(), DummyParticles()]

            def add(self, **kwargs):
                return None

            def move_to_com(self):
                return None

            def energy(self):
                return -1.0

            def integrate(self, time, exact_finish_time=0):
                self.t = 1e7

    monkeypatch.setattr(rb_mod, "rebound", FakeReboundModule)
    monkeypatch.setattr(rb_mod.np, "savez", lambda *args, **kwargs: None)

    sim = OrbitalSimulation(DummyConfiguration(), DummyRNG())
    sim._save_result = lambda *args, **kwargs: None
    result = sim.run_orbital_integration(0, 1000.0, 0.1, 0.2, 0.3, 1.0, [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0])

    assert result is None
