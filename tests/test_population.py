import numpy as np

import captus.three_body_capture.simulations.population as pop_mod
from captus.three_body_capture.simulations.population import PBHPopulation


class DummyConfiguration:
    def __init__(self, name, seed, importance_sampling=True):
        self.name = name
        self.seed = seed
        self.importance_sampling = importance_sampling
        self.system_params = {"mC": None}
        self.simulation_params = {"sample_size": None, "v_inf_grid": [10.0, 20.0, 30.0], "max_v": 30.0}
        self.system_calls = []
        self.sim_calls = []

    def set_system_param(self, key, value):
        self.system_params[key] = value
        self.system_calls.append((key, value))

    def set_simulation_param(self, key, value):
        self.simulation_params[key] = value
        self.sim_calls.append((key, value))

    def get_simulation_param(self, all=False):
        return self.simulation_params if all else self.simulation_params["sample_size"]

    def print_configuration(self):
        return None

    def get_system_param(self, all=False):
        if all:
            return {"seed_base": self.seed, **self.system_params}
        return self.seed

    def get_save_dir_mc(self):
        return "/tmp/mc"

    def get_save_dir_rebound(self):
        return "/tmp/rebound"


class DummyAnalysis:
    def __init__(self, *args, **kwargs):
        self._mc = {"mc": True}
        self._sampled = {"sampled": True}
        self._combined = {"combined": True}

    def get_mc_results(self):
        return self._mc

    def get_sampled_mc_results(self):
        return self._sampled

    def get_combined_dictionary(self, *args, **kwargs):
        return self._combined


def test_preprocess_population_builds_configuration(monkeypatch):
    monkeypatch.setattr(pop_mod.conf, "Configuration", DummyConfiguration)

    pop = PBHPopulation(
        system_name="TestSystem",
        mPBH_min=1e-10,
        mPBH_max=1e-9,
        mPBH_num=2,
        n_per_mPBH=4,
        seed_start=10,
        identifier="id",
        limit_max_v=True,
        system_param_overrides={"mA": 99},
        simulation_param_overrides={"max_execution_time": 999},
    )

    assert len(pop.population_dict) == 2
    first = next(iter(pop.population_dict.values()))
    assert first["seed"] == 10
    assert first["configuration"].system_params["mC"] is not None
    assert first["configuration"].system_params["mA"] == 99
    assert first["configuration"].simulation_params["sample_size"] == 4
    assert first["configuration"].simulation_params["max_execution_time"] == 999
    assert first["configuration"].simulation_params["max_v"] == 22.5


def test_sample_mc_attaches_sampled_results(monkeypatch):
    monkeypatch.setattr(pop_mod.conf, "Configuration", DummyConfiguration)
    monkeypatch.setattr(pop_mod.anl, "Analysis", DummyAnalysis)

    pop = PBHPopulation(
        system_name="TestSystem",
        mPBH_min=1e-10,
        mPBH_max=1e-10,
        mPBH_num=1,
        n_per_mPBH=4,
        seed_start=10,
        identifier=None,
    )

    pop.sample_mc()
    entry = next(iter(pop.population_dict.values()))
    assert entry["sampled_mc"] == {"sampled": True}


def test_analyze_population_attaches_analysis_results(monkeypatch):
    monkeypatch.setattr(pop_mod.conf, "Configuration", DummyConfiguration)
    monkeypatch.setattr(pop_mod.anl, "Analysis", DummyAnalysis)

    pop = PBHPopulation(
        system_name="TestSystem",
        mPBH_min=1e-10,
        mPBH_max=1e-10,
        mPBH_num=1,
        n_per_mPBH=4,
        seed_start=10,
        identifier=None,
    )

    pop.analyze_population(r=8, time_averages=True, update=True, use_cached_data=False)
    entry = next(iter(pop.population_dict.values()))
    assert entry["analysis_results"] == {"combined": True}
    assert entry["analysis"]._combined == {"combined": True}
