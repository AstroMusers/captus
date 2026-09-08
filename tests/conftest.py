import pytest


class DummyPool:
    def __init__(self, processes=None):
        self.processes = processes
        self.imap_calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def imap_unordered(self, func, tasks, chunksize=1):
        self.imap_calls.append((func, list(tasks), chunksize))
        for task in tasks:
            yield func(task)

    def terminate(self):
        pass


@pytest.fixture
def dummy_pool():
    return DummyPool


@pytest.fixture
def dummy_configuration_class():
    class DummyConfiguration:
        def __init__(self, seed_base=1234, v_inf_grid=(10.0, 20.0, 30.0), save_dir="/tmp/rebound"):
            self._system_params = {"seed_base": seed_base}
            self._simulation_params = {"v_inf_grid": list(v_inf_grid)}
            self._save_dir = save_dir

        def get_system_param(self, all=False):
            if all:
                return self._system_params
            return self._system_params["seed_base"]

        def get_simulation_param(self, key):
            return self._simulation_params[key]

        def get_save_dir_rebound(self):
            return self._save_dir

    return DummyConfiguration


@pytest.fixture
def sampled_mc_block():
    def _sampled_mc_block(v_inf=1000.0, n=2):
        return {
            "block_1": {
                "v_inf": v_inf,
                "idx": list(range(n)),
                "cap_lambda": [0.1, 0.2],
                "cap_beta": [0.3, 0.4],
                "cap_b": [1.0, 2.0],
                "cap_phi": [0.5, 0.6],
                "cap_C_pos": ["cpos0", "cpos1"],
                "cap_C_v2": ["cv20", "cv21"],
                "cap_B_pos": ["bpos0", "bpos1"],
                "cap_B_v": ["bv0", "bv1"],
            }
        }

    return _sampled_mc_block