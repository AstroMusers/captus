import multiprocessing
from typing import Any, Iterable
import time
import datetime

from numpy.random import Generator, PCG64, SeedSequence

import src.ThreeBodyCapture.Simulations.montecarlo as MCi


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
        print(f"\n{'='*70}")
        print(f"Running {total_to_run} Monte Carlo simulations")
        print(f"Using {num_cores} cores")
        print(f"{'='*70}\n")

        # Helper to format seconds to H:MM:SS
        def _fmt(sec):
            sec = int(sec)
            m, s = divmod(sec, 60)
            h, m = divmod(m, 60)
            return f"{h:d}:{m:02d}:{s:02d}"

        results = []
        completed = 0
        processes = min(num_cores, total_to_run)
        chunksize = 1

        with multiprocessing.Pool(processes=num_cores) as pool:
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
                        f"elapsed {_fmt(elapsed_sec)}, estimated remaining time: {_fmt(remaining_sec)}",
                        end='', flush=True,
                    )

            except KeyboardInterrupt:
                pool.terminate()
                print("\nInterrupted by user — terminating workers.")
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


