import multiprocessing
from typing import Any
import time

from numpy.random import Generator, PCG64, SeedSequence
from captus.utils.misc import format_duration

import captus.three_body_capture.simulations.montecarlo as MCi


def _mc_worker(pars: tuple[Any, SeedSequence, float]):
    """Run one Monte Carlo simulation for a single seed / v_inf pair."""
    configuration, child_ss, v_inf = pars
    rng = Generator(PCG64(child_ss))
    simulation = MCi.MonteCarloSimulation(configuration, rng, verbose=False)
    return simulation.run_monte_carlo_simulation(v_inf)


class ThreeBodyCapture:
    def __init__(self, population_dict, verbose=False):
        self.prerun_dict = population_dict
        self.verbose = verbose

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

    def run_simulation(self, num_cores=None):
        """Run the full population of simulations and return the collected results."""
        overall_start_ts = time.time()
        tasks = self._build_tasks()
        if not tasks:
            return []

        if num_cores is None:
            num_cores = multiprocessing.cpu_count()

        total_to_run = len(tasks)
        processes = min(num_cores, total_to_run)
        print(f"\n{'='*70}")
        print(f"Running {total_to_run} Monte Carlo simulations")
        print(f"Using {processes} cores")
        print(f"{'='*70}\n")

        results = []
        completed = 0
        chunksize = 1

        with multiprocessing.Pool(processes=processes) as pool:
            try:
                for res in pool.imap_unordered(_mc_worker, tasks, chunksize=chunksize):
                    results.append(res)
                    completed += 1

                    elapsed_sec = time.time() - overall_start_ts
                    avg_sec = elapsed_sec / completed
                    remaining_sec = avg_sec * (total_to_run - completed)
                    pct = (completed / total_to_run) * 100

                    print(
                        f"\rProgress: {completed}/{total_to_run} ({pct:.1f}%), "
                        f"elapsed {format_duration(elapsed_sec)}, "
                        f"estimated remaining time: {format_duration(remaining_sec)}",
                        end='', flush=True,
                    )

            except KeyboardInterrupt:
                pool.terminate()
                print("\nInterrupted by user – terminating workers.")
                raise

        # Ensure final newline after progress bar
        print()
        elapsed = time.time() - overall_start_ts
        print(f"\n{'='*70}")
        print(f"✓ ALL MONTE CARLO SIMULATIONS COMPLETE!")
        print(f"Total time: {elapsed/60:.2f} min")
        if total_to_run > 0:
            print(f"Average: {elapsed/total_to_run:.2f} sec/simulation")
        print(f"{'='*70}\n")


