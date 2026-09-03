import datetime
import multiprocessing
import os

import numpy as np
from numpy.random import Generator, PCG64, SeedSequence

from src.ThreeBodyCapture.Simulations.rebound import OrbitalSimulation


def check_result_exists(configuration, v_inf_kms, i, seed):
    """Check if a result file already exists."""
    v_str = f"{v_inf_kms:.0f}"
    save_dir = configuration.get_save_dir_rebound()
    dir_npz = os.path.join(save_dir, f"v{v_str}")
    out_npz = os.path.join(dir_npz, f"sim_{i}_{seed}.npz")
    return os.path.isfile(out_npz)


def rebound_worker(pars):
    """Top-level worker for multiprocessing."""
    configuration, child_ss, i, v_inf, lambda1, beta, phi, b, pos_C, v_C, pos_B, v_B = pars
    rng = Generator(PCG64(child_ss))
    simulation = OrbitalSimulation(configuration=configuration, rng=rng)
    return simulation.run_orbital_integration(i, v_inf, lambda1, beta, phi, b, pos_C, v_C, pos_B, v_B)


class ThreeBodyEvolution:
    def __init__(self, population_dict):
        self.population_dict = population_dict
        self.results = None

    def _iter_runs(self):
        if isinstance(self.population_dict, dict):
            return list(self.population_dict.values())
        return list(self.population_dict)

    def _build_tasks(self):
        """Flatten all runs into worker tasks, skipping existing results."""
        all_pars = []
        run_info = []

        for run in self._iter_runs():
            print(f"Preparing {run['name']}...")
            conf_ = run['configuration']
            seed_ = run['seed']
            sampled_mc_ = run['sampled_mc']

            base_ss = SeedSequence(seed_)
            total_possible = np.sum([len(vdata['idx']) for vdata in sampled_mc_.values()])
            all_child_sss = base_ss.spawn(int(total_possible))

            run_pars = []
            global_idx = 0
            skipped = 0

            for _, vdata in sampled_mc_.items():
                v_inf = vdata['v_inf']
                v_inf_kms = v_inf / 1e3
                lambda1 = vdata['cap_lambda']
                beta = vdata['cap_beta']
                b = vdata['cap_b']
                phi = vdata['cap_phi']
                pos_C = vdata['cap_C_pos']
                v_C = vdata['cap_C_v2']
                pos_B = vdata['cap_B_pos']
                v_B = vdata['cap_B_v']

                for i in vdata['idx']:
                    if not check_result_exists(conf_, v_inf_kms, i, seed_):
                        run_pars.append((
                            conf_,
                            all_child_sss[global_idx],
                            i, v_inf, lambda1[i], beta[i], phi[i], b[i],
                            pos_C[i], v_C[i], pos_B[i], v_B[i],
                        ))
                    else:
                        skipped += 1
                    global_idx += 1

            all_pars.extend(run_pars)
            run_info.append({
                'name': run['name'],
                'n_sims': len(run_pars),
                'skipped': skipped,
            })
            print(f"  Added {len(run_pars)} simulations (skipped {skipped})")

        return all_pars, run_info

    def run_simulation(self, num_cores=50):
        """Run all prepared evolution simulations in a multiprocessing pool."""
        overall_start = datetime.datetime.now()
        all_pars, run_info = self._build_tasks()

        total_to_run = len(all_pars)
        print(f"\n{'='*70}")
        print(f"Running {total_to_run} total simulations across {len(run_info)} runs")
        print(f"Using {num_cores} cores")
        print(f"{'='*70}\n")

        if total_to_run == 0:
            print('✓ All simulations already complete!')
            self.results = []
            return self.results

        with multiprocessing.Pool(processes=min(num_cores, total_to_run)) as pool:
            results = pool.map(rebound_worker, all_pars)

        elapsed = datetime.datetime.now() - overall_start
        print(f"\n{'='*70}")
        print(f"✓ ALL RUNS COMPLETE!")
        print(f"Total time: {elapsed} ({elapsed.total_seconds()/60:.2f} min)")
        print(f"Average: {elapsed.total_seconds()/total_to_run:.2f} sec/simulation")
        print(f"{'='*70}")

        self.results = results
        return results