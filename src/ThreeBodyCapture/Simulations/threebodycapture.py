import multiprocessing
from typing import Any, Iterable

from numpy.random import Generator, PCG64, SeedSequence

import src.ThreeBodyCapture.Simulations.montecarlo as MCi


def _mc_worker(pars: tuple[Any, SeedSequence, float]):
    """Run one Monte Carlo simulation for a single seed / v_inf pair."""
    configuration, child_ss, v_inf = pars
    rng = Generator(PCG64(child_ss))
    simulation = MCi.MonteCarloSimulation(configuration, rng)
    return simulation.run_monte_carlo_simulation(v_inf)


class ThreeBodyCapture:
    def __init__(self, population_dict):
        self.prerun_dict = population_dict

    def _iter_runs(self) -> list[Any]:
        """Return the population entries as a list regardless of input container."""
        if isinstance(self.prerun_dict, dict):
            return list(self.prerun_dict.values())
        return list(self.prerun_dict)

    def _extract_configuration(self, run: Any):
        """Support both dict-style runs and direct configuration objects."""
        if isinstance(run, dict) and "configuration" in run:
            return run["configuration"]
        return run

    def _extract_seed(self, run: Any, configuration: Any):
        if isinstance(run, dict) and "seed" in run:
            return run["seed"]
        if hasattr(configuration, "get_system_param"):
            params = configuration.get_system_param(all=True)
            return params.get("seed_base")
        raise ValueError("No seed available for the run configuration.")

    def _extract_v_inf_grid(self, run: Any, configuration: Any):
        if isinstance(run, dict) and "v_inf_grid" in run:
            return run["v_inf_grid"]
        if hasattr(configuration, "get_simulation_param"):
            return configuration.get_simulation_param("v_inf_grid")
        raise ValueError("No v_inf grid available for the run configuration.")

    def _build_tasks(self) -> list[tuple[Any, SeedSequence, float]]:
        tasks = []
        for run in self._iter_runs():
            configuration = self._extract_configuration(run)
            seed = self._extract_seed(run, configuration)
            v_inf_grid = self._extract_v_inf_grid(run, configuration)

            base_ss = SeedSequence(seed)
            child_sss = base_ss.spawn(len(v_inf_grid))
            tasks.extend(
                (configuration, child_sss[i], v_inf_grid[i])
                for i in range(len(v_inf_grid))
            )

        return tasks

    def run_simulation(self, n_processes=None):
        """Run the full population of simulations and return the collected results."""
        tasks = self._build_tasks()
        if not tasks:
            return []

        if n_processes is None:
            n_processes = multiprocessing.cpu_count()

        with multiprocessing.Pool(processes=n_processes) as pool:
            results = pool.map(_mc_worker, tasks)

        return results

