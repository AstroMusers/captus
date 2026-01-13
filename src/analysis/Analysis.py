import os
import glob
import numpy as np
import src.utils.calculations as calc
from astropy import units as u
import astropy.constants as const
import src.configurations.Configuration as config
import src.utils.misc as misc
import pandas as pd

REPO_ROOT = misc._resolve_repo_root()
print(f'Using REPO_ROOT: {REPO_ROOT}')

class Analysis:
    def __init__(self, name, configuration, rng=None, results_dir_mc=None, results_dir_rebound=None):

        self.name = name
        self.system_param_dict = configuration.get_system_param(all=True)
        self.simulation_param_dict = configuration.get_simulation_param(all=True)
        self.mc_sample_size = self.simulation_param_dict['sample_size']

        if rng is None:
            seed = self.system_param_dict['seed_base']
            self.rng = np.random.default_rng(seed)

        if self.simulation_param_dict['importance_sampling']:
            trials = self.simulation_param_dict['max_trials']
        else:
            trials = self.simulation_param_dict['trials']

        trials_str = f"{int(trials):.0e}"  # e.g., 100000 -> "1e5"

        if results_dir_mc is not None:
            self.mc_dir = os.path.join(results_dir_mc)
        else:
            self.mc_dir = os.path.join(REPO_ROOT, f'runs/{name}/Monte_Carlo_Results/')

        if results_dir_rebound is not None:
            self.rebound_dir = os.path.join(results_dir_rebound)
        else:
            self.rebound_dir = os.path.join(REPO_ROOT, f'runs/{name}/Rebound_Simulation_Results/')

        print(f'Loading MC results from: {self.mc_dir}')
        print(f'Loading Rebound results from: {self.rebound_dir}')

        try:
            self.mc_results = self._load_mc_results()
            self.sampled_mc_results = self._get_sampled_mc_results()
            self.rebound_results = self._load_rebound_results()
            self.results_dictionary = self.get_combined_dictionary()
        except Exception as e:
            print(f"Error loading results: {e}")

    def _load_mc_results(self):

        mc_results = {}
        for mc_path in glob.glob(os.path.join(self.mc_dir, "monte_carlo_results_*.npz")):
            try:
                with np.load(mc_path, allow_pickle=True) as mc_probe:
                    v_inf = float(mc_probe["v_inf"])
                vk = self._vkey_from_vinf(v_inf)
                data = np.load(mc_path, allow_pickle=True)
                mc_results[vk] = data
        
            except Exception as e:
                print(f"Failed to index MC {mc_path}: {e}")

        return mc_results
    
    def _load_rebound_results(self):

        rebound_results = {}
        for v_folder in sorted(os.listdir(self.rebound_dir)):

            if not (v_folder.startswith("v") and os.path.isdir(os.path.join(self.rebound_dir, v_folder))):
                continue

            v_path = os.path.join(self.rebound_dir, v_folder)
            entries = []

            for f in sorted(glob.glob(os.path.join(v_path, "*.npz"))):
                try:
                    data = np.load(f, allow_pickle=True)

                    entries.append(data)
                except Exception as e:
                    print(f"Failed to load Rebound {f}: {e}")
            rebound_results[v_folder] = entries
        return rebound_results

    def _vkey_from_vinf(self, v_inf_mps: float) -> str:
    # km/s rounded to nearest integer, prefixed with 'v'
        return f"v{int(np.rint(v_inf_mps/1e3))}"
    
    def get_combined_dictionary(self):
        catalog = {}
        sorted_keys = sorted(self.mc_results.keys(), key=lambda x: int(x[1:]))  # sort by integer value after 'v'
        for v_key in sorted_keys:
            mc_npz = self.mc_results.get(v_key, None)
            rebound_npz_list = self.rebound_results.get(v_key, [])
            sampled_mc_results = self.sampled_mc_results.get(v_key, None) if hasattr(self, 'sampled_mc_results') else None

            if mc_npz is None or sampled_mc_results is None:
                print(f"Warning: No MC data for {v_key}, skipping.")
                continue
            sampled = int(mc_npz['sample_number']) if mc_npz is not None and 'sample_number' in mc_npz else 0
            n_captured = int(mc_npz['n_captured']) if mc_npz is not None and 'n_captured' in mc_npz else 0
            

            masks = self._get_masks(rebound_npz_list)

            catalog[v_key] = {
                #Metadata
                "v_inf_m_s": float(mc_npz['v_inf']) if mc_npz is not None else None,
                "v_inf_au_yr": (float(mc_npz['v_inf']) * u.m / u.s).to(u.au / u.yr).value if mc_npz is not None else None,
                'sample_count': sampled,
                'capture_count': n_captured,
                'sampled_capture_count': len(sampled_mc_results['idx']) if sampled_mc_results is not None else 0,

                # Raw data
                "mc": mc_npz,
                "rebound": rebound_npz_list,

                # Sampled data
                "sampled_mc": sampled_mc_results,

                # Compute derived data per-entry
                "termination_counts": self._get_termination_counts(rebound_npz_list),
                "masks": masks,
                "occurrences": self._get_occurrences(mc_npz, rebound_npz_list, sampled_mc_results, masks),
                
                # Legacy fields for backward compatibility
                "ej_mask": masks["ejection"].tolist(),
                "coll_mask": masks["collision"].tolist(),
            }

        # Compute overall statistics
        v_keys = [k for k in catalog.keys() if isinstance(k, str) and k.startswith('v')]
        catalog['total_sample_count'] = sum(catalog[k]['sample_count'] for k in v_keys)
        catalog['total_capture_count'] = sum(catalog[k]['capture_count'] for k in v_keys)
        catalog['total_sampled_capture_count'] = sum(catalog[k]['sampled_capture_count'] for k in v_keys)
        # Total occurrences computed once after all entries
        catalog['total_occurrences_trapz'] = self._compute_total_occurrences(catalog)
        catalog['total_occurrences_gl'] = self._integrate_total_occurrences(catalog)
        return catalog

    def get_mc_results(self):
        return self.mc_results

    def _get_sampled_mc_results(self):
        sampled_mc = {}
        for v in self.mc_results.keys():
            data = self.mc_results[v]  # npz file handle
            n_captured = int(data['n_captured'])
            print(f'captured objects for v_inf = {v} km/s: {n_captured}')
            if n_captured == 0:
                continue

            # Arrays of per-capture parameters saved by MC
            a_au = data['a_au']            # shape (n_captured,)
            e_arr = data['e']              # shape (n_captured,)
            cap_lambda = data['cap_lambda']# shape (n_captured,)
            cap_beta = data['cap_beta']    # shape (n_captured,)
            cap_b = data['cap_b']          # shape (n_captured,) in meters
            cap_phi = data['cap_phi']      # shape (n_captured,)
            try:
                epsilon = data['epsilon']
                sampled = data['sample_number']
            except:
                epsilon = 0.1
                sampled = 0

            # Sanity: all arrays must have length n_captured
            assert len(a_au) == len(e_arr) == len(cap_lambda) == len(cap_beta) == len(cap_b) == len(cap_phi) == n_captured


            if n_captured <= self.mc_sample_size:
                idx = np.arange(n_captured)
            else:
                idx = self.rng.choice(n_captured, size=self.mc_sample_size, replace=False)

            # Slice consistently
            sampled_mc[v] = {
                'v_inf': float(data['v_inf']),             # m/s
                'a_au': a_au[idx],
                'e': e_arr[idx],
                'lambda1': cap_lambda[idx],
                'beta': cap_beta[idx],
                'b': cap_b[idx],                         # meters (convert in OrbitalSimulation)
                'phi': cap_phi[idx],
                'epsilon': epsilon,
                'idx': idx,
                'total_capture_count': n_captured,
                'total_sample_count': sampled,
                'sampled_capture_count': len(idx),
            }
            print(f'v_inf = {v} km/s: sampled {len(idx)} captured objects')
        return sampled_mc
    

    def get_sampled_mc_results(self):
        if not hasattr(self, 'sampled_mc_results'):
            self._get_sampled_mc_results()
        return self.sampled_mc_results
    
    def sample_subpopulation(self, subpop_size):

        sampled_catalog = {}
        catalog = self.results_dictionary

        for v_key in catalog.keys():
            # Skip non-velocity keys
            if not (isinstance(v_key, str) and v_key.startswith('v')):
                continue
                
            # Get the rebound results list for this velocity
            rebound_list = catalog[v_key].get('rebound', [])
            mc_list = catalog[v_key].get('sampled_mc', [])
            n_available = catalog[v_key]['sampled_capture_count']
            print(f'available simulations for {v_key}: {n_available}')
            if n_available == 0:
                print(f"Warning: No rebound results for {v_key}, skipping.")
                continue
                
            # Sample indices (or use all if fewer than subpop_size)
            if n_available <= subpop_size:
                sampled_indices = np.arange(n_available)
                print(f"{v_key}: Using all {n_available} simulations (fewer than {subpop_size})")
            else:
                sampled_indices = self.rng.choice(n_available, size=subpop_size, replace=False)
                print(f"{v_key}: Sampled {subpop_size} out of {n_available} simulations")
            
            # Create sampled rebound list
            sampled_rebound_list = [rebound_list[i] for i in sampled_indices]
            sampled_mc_dict = self._get_sampled_mc_subpopulation(v_key, sampled_indices)
            
            # Recompute all derived quantities for the sampled subset
            masks = self._get_masks(sampled_rebound_list)
            
            sampled_catalog[v_key] = {
                # Copy metadata (unchanged)
                "v_inf_m_s": catalog[v_key]["v_inf_m_s"],
                "v_inf_au_yr": catalog[v_key]["v_inf_au_yr"],
                'sample_count': catalog[v_key]['sample_count'],  # Original MC sample count
                'capture_count': catalog[v_key]['capture_count'],  # Original MC capture count
                'sampled_capture_count': len(sampled_indices),  # New: sampled rebound count
                
                # Copy original MC data (unchanged)
                "mc": catalog[v_key]["mc"],

                
                
                # Sampled rebound data
                "rebound": sampled_rebound_list,
                "sampled_mc": sampled_mc_dict,

                # Recompute derived quantities for sampled subset
                "termination_counts": self._get_termination_counts(sampled_rebound_list),
                "masks": masks,
                "occurrences": self._get_occurrences(
                    catalog[v_key]["mc"], 
                    sampled_rebound_list, 
                    sampled_mc_dict, 
                    masks
                ),
                
                # Legacy fields
                "ej_mask": masks["ejection"].tolist(),
                "coll_mask": masks["collision"].tolist(),
            }
        
        # Compute overall statistics for sampled catalog
        v_keys = [k for k in sampled_catalog.keys() if isinstance(k, str) and k.startswith('v')]
        sampled_catalog['total_sample_count'] = sum(sampled_catalog[k]['sample_count'] for k in v_keys)
        sampled_catalog['total_capture_count'] = sum(sampled_catalog[k]['capture_count'] for k in v_keys)
        sampled_catalog['total_sampled_capture_count'] = sum(sampled_catalog[k]['sampled_capture_count'] for k in v_keys)
        sampled_catalog['total_occurrences_trapz'] = self._compute_total_occurrences(sampled_catalog)
        sampled_catalog['total_occurrences_gl'] = self._integrate_total_occurrences(sampled_catalog)
        
        print(f"\nSampled catalog summary:")
        print(f"  Total velocity bins: {len(v_keys)}")
        print(f"  Total sampled simulations: {sampled_catalog['total_sampled_capture_count']}")
        print(f"  Original total simulations: {catalog['total_sampled_capture_count']}")
        
        return sampled_catalog
    
    def _get_sampled_mc_subpopulation(self, vkey, sampled_indices):
        data = self.sampled_mc_results[vkey]
        return {
            'v_inf': data['v_inf'],
            'a_au': data['a_au'][sampled_indices],
            'e': data['e'][sampled_indices],
            'lambda1': data['lambda1'][sampled_indices],
            'beta': data['beta'][sampled_indices],
            'b': data['b'][sampled_indices],
            'phi': data['phi'][sampled_indices],
            'epsilon': data['epsilon'],
            'idx': sampled_indices,
            'total_capture_count': data['total_capture_count'],
            'total_sample_count': data['total_sample_count'],
            'sampled_capture_count': len(sampled_indices),
        }

    def get_rebound_results(self):
        return self.rebound_results

    def _get_termination_counts(self, rebound_list):
        """
        Compute termination counts for a single v bin.
        Pure function: takes rebound entries, returns counts dict.
        """
        counts = {
            "b_err_count": 0,
            "P_err_count": 0,
            "time_err_count": 0,
            "escape_C_count": 0,
            "escape_B_count": 0,
            "energy_err_count": 0,
            "preprocess_err_count": 0,
            "time_exceeded_count": 0,
            "collision_count_rebound": 0,
            "collision_count": 0,
            "completed_count": 0,
        }
        
        for entry in rebound_list:
            if "errors" not in entry.files or "termination_flag" not in entry.files:
                continue
                
            errors = entry["errors"]
            flag = self._as_scalar(entry["termination_flag"])
            
            if len(errors) > 0:
                err_str = str(errors[0])
                if 'impact parameter b' in err_str:
                    counts["b_err_count"] += 1
                if 'period' in err_str:
                    counts["P_err_count"] += 1
                if 'time' in err_str:
                    counts["time_err_count"] += 1
                if 'preprocess' in err_str:
                    counts["preprocess_err_count"] += 1

            if flag is None:
                counts["collision_count"] += 1

            elif isinstance(flag, str):
                if 'escape_C' in flag:
                    counts["escape_C_count"] += 1
                if 'escape_B' in flag:
                    counts["escape_B_count"] += 1
                if 'energy' in flag:
                    counts["energy_err_count"] += 1
                if 'time_exceeded' in flag:
                    counts["time_exceeded_count"] += 1
                if 'collision' in flag:
                    counts["collision_count_rebound"] += 1
                if 'completed' in flag:
                    counts["completed_count"] += 1
                    
        return counts
    def _get_masks(self, rebound_list):
        """
        Compute ejection/collision/termination masks for a single v bin.
        Pure function: takes rebound entries, returns masks dict.
        """
        ej_mask = []
        coll_mask = []
        
        for entry in rebound_list:
            if "termination_flag" not in entry.files:
                ej_mask.append(0)
                coll_mask.append(0)
                continue
                
            flag = self._as_scalar(entry["termination_flag"])
            
            if flag is None:
                coll_mask.append(1)
                ej_mask.append(0)
            elif isinstance(flag, str) and 'escape_C' in flag:
                ej_mask.append(1)
                coll_mask.append(0)
            else:
                ej_mask.append(0)
                coll_mask.append(0)
                
        ej_arr = np.array(ej_mask, dtype=int)
        coll_arr = np.array(coll_mask, dtype=int)
        
        return {
            "ejection": ej_arr,
            "collision": coll_arr,
            "termination": ej_arr + coll_arr,
        }
    
    def _get_occurrences(self, mc_npz, rebound_list, sampled_mc, masks):
        """
        Compute occurrence metrics for a single v bin.
        Pure function: takes inputs, returns occurrences dict.
        """
        ej_mask = masks["ejection"]
        coll_mask = masks["collision"]
        term_mask = masks["termination"]
        
        mA = self.system_param_dict["mA"]
        mB = self.system_param_dict["mB"]   
        aB = self.system_param_dict["aB"]
        vdm = self.simulation_param_dict['vDM']
        muA = mA * const.G.value
        muB = mB * const.G.value
        epsilon = sampled_mc["epsilon"] 
        rClose = calc.r_close(epsilon, mA, mB, aB)
        v1Mag = calc.v_1_mag(mc_npz["v_inf"], muA, muB, aB, rClose)
        v1Mag_au = v1Mag * (u.m / u.s).to(u.au/u.yr)
        vdm = (vdm*u.m/u.s).to(u.au/u.yr).value
        b_max = (rClose * u.m).to(u.au).value
        vinfinity = (mc_npz["v_inf"]*u.m/u.s).to(u.au/u.yr).value
        
        n_captured = int(sampled_mc["total_capture_count"])
        try:
            n_sampled = int(sampled_mc["total_sample_count"])
        except KeyError:
            n_sampled = 0
        try:
            n_sampled_captures = int(sampled_mc["sampled_capture_count"])
        except KeyError:
            n_sampled_captures = 0

        n_ejected = int(np.sum(ej_mask))
        n_collided = int(np.sum(coll_mask))
        n_terminated = int(np.sum(term_mask))
        
        capture_cross_section_total = (mc_npz["sigma_MC_m2"] * u.m**2).to(u.au**2).value
        capture_cross_section_ejected = ((n_ejected / n_sampled_captures) * capture_cross_section_total) if n_sampled > 0 else 0.0
        capture_cross_section_collided = ((n_collided / n_sampled_captures) * capture_cross_section_total) if n_sampled > 0 else 0.0
        capture_cross_section_terminated = ((n_terminated / n_sampled_captures) * capture_cross_section_total) if n_sampled > 0 else 0.0

        lifetimes = [e["lifetime"] for e in rebound_list if "lifetime" in e.files]
        ejected_lifetimes = [l for l, ej in zip(lifetimes, ej_mask) if ej == 1]
        collided_lifetimes = [l for l, coll in zip(lifetimes, coll_mask) if coll == 1]
        terminated_lifetimes = [l for l, term in zip(lifetimes, term_mask) if term == 1]

        total_rate = (len([l for l in lifetimes if l is not None]) / 
                      np.sum([l for l in lifetimes if l is not None])) if len(lifetimes) > 0 else 0.0
        ejection_rate = (len([l for l in ejected_lifetimes if l is not None]) /
                        np.sum([l for l in ejected_lifetimes if l is not None])) if len(ejected_lifetimes) > 0 else 0.0
        collision_rate = (len([l for l in collided_lifetimes if l is not None]) / 
                         np.sum([l for l in collided_lifetimes if l is not None])) if len(collided_lifetimes) > 0 else 0.0
        termination_rate = (len([l for l in terminated_lifetimes if l is not None]) / 
                           np.sum([l for l in terminated_lifetimes if l is not None])) if len(terminated_lifetimes) > 0 else 0.0
        
        f_approx = (np.sqrt(2/np.pi)) * (vinfinity**2 / (vdm**3)) * np.exp(- (vinfinity)**2/(2*vdm**2))
        # f_approx = (vinfinity**2 / (vdm**3))

        neq_ejected = f_approx * capture_cross_section_ejected * v1Mag_au / ejection_rate if ejection_rate > 0 else 0.0
        neq_collided = f_approx * capture_cross_section_collided * v1Mag_au / collision_rate if collision_rate > 0 else 0.0
        neq_terminated = f_approx * capture_cross_section_terminated * v1Mag_au / termination_rate if termination_rate > 0 else 0.0
        neq_total = f_approx * capture_cross_section_total * v1Mag_au / total_rate if total_rate > 0 else 0.0
        
        return {
            "n_captured": n_captured,
            "n_sampled": n_sampled,
            "n_sampled_captures": n_sampled_captures,
            "n_ejected": n_ejected,
            "n_collided": n_collided,
            "n_terminated": n_terminated,
            "capture_cross_section_total": capture_cross_section_total,
            "capture_cross_section_ejected": capture_cross_section_ejected,
            "capture_cross_section_collided": capture_cross_section_collided,
            "capture_cross_section_terminated": capture_cross_section_terminated,
            "capture_cross_section_total_areaB": mc_npz["sigma_MC_areaB"],
            "ejection_rate": ejection_rate,
            "collision_rate": collision_rate,
            "termination_rate": termination_rate,
            "total_rate": total_rate,
            "terminated_systems_neq": neq_terminated,
            "ejected_systems_neq": neq_ejected,
            "collided_systems_neq": neq_collided,
            "total_systems_neq": neq_total,
            "average_lifetime_ejected": np.mean(ejected_lifetimes) if len(ejected_lifetimes) > 0 else None,
            "average_lifetime_collided": np.mean(collided_lifetimes) if len(collided_lifetimes) > 0 else None,
            "average_lifetime_terminated": np.mean(terminated_lifetimes) if len(terminated_lifetimes) > 0 else None,
            "average_lifetime_total": np.mean(lifetimes) if len(lifetimes) > 0 else None,
        }
    
    def _compute_total_occurrences(self, catalog):
        """
        Compute aggregate occurrences across all v bins.
        Uses canonical v_inf_au_yr from catalog entries.
        """
        v_keys = [k for k in catalog.keys() if isinstance(k, str) and k.startswith('v')]
        
        if len(v_keys) == 0:
            return {
                "total_captured": 0,
                "total_sampled": 0,
                "total_ejected": 0,
                "total_collided": 0,
                "total_terminated": 0,
                "terminated_systems_neq": 0.0,
                "ejected_systems_neq": 0.0,
                "collided_systems_neq": 0.0,
                "total_systems_neq": 0.0,
            }
        
        total_captured = sum([catalog[v]['occurrences']["n_captured"] for v in v_keys if 'occurrences' in catalog[v]])
        total_sampled = sum([catalog[v]['occurrences']["n_sampled"] for v in v_keys if 'occurrences' in catalog[v]])
        total_ejected = sum([catalog[v]['occurrences']["n_ejected"] for v in v_keys if 'occurrences' in catalog[v]])
        total_collided = sum([catalog[v]['occurrences']["n_collided"] for v in v_keys if 'occurrences' in catalog[v]])
        total_terminated = sum([catalog[v]['occurrences']["n_terminated"] for v in v_keys if 'occurrences' in catalog[v]])
        
        # Use canonical v_inf_au_yr from catalog entries (no nested lookup needed)
        x_vals = [catalog[v]['v_inf_au_yr'] for v in v_keys if 'v_inf_au_yr' in catalog[v] and 'occurrences' in catalog[v]]
        y_total = [catalog[v]['occurrences']["total_systems_neq"] for v in v_keys if 'occurrences' in catalog[v]]
        y_collided = [catalog[v]['occurrences']["collided_systems_neq"] for v in v_keys if 'occurrences' in catalog[v]]
        y_ejected = [catalog[v]['occurrences']["ejected_systems_neq"] for v in v_keys if 'occurrences' in catalog[v]]
        y_terminated = [catalog[v]['occurrences']["terminated_systems_neq"] for v in v_keys if 'occurrences' in catalog[v]]
        
        total_systems_neq = calc.integrate_trapezoidal(x=x_vals, y=y_total)
        collided_systems_neq = calc.integrate_trapezoidal(x=x_vals, y=y_collided)
        ejected_systems_neq = calc.integrate_trapezoidal(x=x_vals, y=y_ejected)
        terminated_systems_neq = calc.integrate_trapezoidal(x=x_vals, y=y_terminated)
        
        return {
            "total_captured": total_captured,
            "total_sampled": total_sampled,
            "total_ejected": total_ejected,
            "total_collided": total_collided,
            "total_terminated": total_terminated,
            "terminated_systems_neq": terminated_systems_neq,
            "ejected_systems_neq": ejected_systems_neq,
            "collided_systems_neq": collided_systems_neq,
            "total_systems_neq": total_systems_neq,
        }

    def _integrate_total_occurrences(self, catalog):
        """
        Build per-v arrays and integrate using Gauss-Legendre quadrature to make totals
        independent of sampling density. Returns a dict of integrated totals.

        - x: velocity array built from v_inf_au_yr (float)
        - y fields integrated: total_systems_neq, collided_systems_neq,
          ejected_systems_neq, terminated_systems_neq
        - Sorting: by x ascending, dropping NaNs
        """
        x_vals = []
        y_total = []
        y_collided = []
        y_ejected = []
        y_terminated = []

        for v_key, entry in catalog.items():
            if not isinstance(v_key, str) or not v_key.startswith('v'):
                continue
            occ = entry.get('occurrences')
            if occ is None:
                continue
            
            x = entry.get('v_inf_au_yr')
            if x is None:
                continue
                
            try:
                x = float(np.asarray(x).item())
            except Exception:
                try:
                    x = float(x)
                except Exception:
                    continue

            total_neq = occ.get('total_systems_neq')
            collided_neq = occ.get('collided_systems_neq')
            ejected_neq = occ.get('ejected_systems_neq')
            terminated_neq = occ.get('terminated_systems_neq')

            if any(v is None for v in (total_neq, collided_neq, ejected_neq, terminated_neq)):
                continue

            def _f(val):
                try:
                    return float(np.asarray(val).item())
                except Exception:
                    return float(val)

            t = _f(total_neq)
            c = _f(collided_neq)
            e = _f(ejected_neq)
            r = _f(terminated_neq)

            if not (np.isfinite(x) and np.isfinite(t) and np.isfinite(c) and np.isfinite(e) and np.isfinite(r)):
                continue

            x_vals.append(x)
            y_total.append(t)
            y_collided.append(c)
            y_ejected.append(e)
            y_terminated.append(r)

        if len(x_vals) == 0:
            return {
                'total_systems_neq_gl': 0.0,
                'collided_systems_neq_gl': 0.0,
                'ejected_systems_neq_gl': 0.0,
                'terminated_systems_neq_gl': 0.0,
                'v_bins_used': 0,
            }

        order = np.argsort(x_vals)
        x = np.asarray(x_vals)[order]
        y_total = np.asarray(y_total)[order]
        y_collided = np.asarray(y_collided)[order]
        y_ejected = np.asarray(y_ejected)[order]
        y_terminated = np.asarray(y_terminated)[order]

        total_gl = calc.integrate_gauss_legendre(x, y_total, n=16)
        collided_gl = calc.integrate_gauss_legendre(x, y_collided, n=16)
        ejected_gl = calc.integrate_gauss_legendre(x, y_ejected, n=16)
        terminated_gl = calc.integrate_gauss_legendre(x, y_terminated, n=16)

        return {
            'total_systems_neq_gl': total_gl,
            'collided_systems_neq_gl': collided_gl,
            'ejected_systems_neq_gl': ejected_gl,
            'terminated_systems_neq_gl': terminated_gl,
            'v_bins_used': int(len(x)),
        }

    def print_total_occurrences_summary_trapz(self):
        """User-facing summary printer based on Gauss-Legendre integration."""
        res = self.results_dictionary.get('total_occurrences_trapz', {})
        print("Total Occurrences (Gauss-Legendre over v) summary:")
        print(f"    v bins used: {res['v_bins_used']}")
        print(f"    total_systems_neq_gl: {res['total_systems_neq_gl']}")
        print(f"    collided_systems_neq_gl: {res['collided_systems_neq_gl']}")
        print(f"    ejected_systems_neq_gl: {res['ejected_systems_neq_gl']}")
        print(f"    terminated_systems_neq_gl: {res['terminated_systems_neq_gl']}")
        print("--------------------------------")

    def print_catalog_summary(self):

        catalog = self.results_dictionary

        print("Catalog summary:")
        for k in catalog.keys():
            if 'v' in k:
                mc_ok = "yes" if catalog[k]["mc"] else "no"
                n_rb = len(catalog[k]["rebound"])
                print(f"  {k}: MC={mc_ok}, Rebound files={n_rb}")


    
    def print_detailed_catalog_summary(self):

        catalog = self.results_dictionary
        v_keys = [k for k in catalog.keys() if isinstance(k, str) and k.startswith('v')]

        for k in v_keys:
            sampled = catalog[k].get("sampled_mc")
            total_capture_count = sampled.get("total_capture_count") if sampled else 0
            total_sample_count = sampled.get("total_sample_count") if sampled else 0
            sampled_capture_count = sampled.get("sampled_capture_count") if sampled else 0

            print(
                f'Velocity bin: {k}, sampled capture count {sampled_capture_count} '
                f'total capture count: {total_capture_count} out of {total_sample_count} samples'
            )

            term_counts = catalog[k].get('termination_counts', {})
            nonzero = {key: val for key, val in term_counts.items() if val > 0}
            if nonzero:
                print(f"{k} non-zero termination counts:")
                for k, val in nonzero.items():
                    # handle arrays vs scalars
                    out = int(np.asarray(val).sum()) if isinstance(val, (list, tuple, np.ndarray)) else val
                    print(f"  {k}: {out}")
            print("--------------------------------")
   
 
    
    def get_name(self):
        return self.name

    def print_occurrences_summary(self):

        catalog = self.results_dictionary
        print("Occurrences summary:")
        for k in [k for k in catalog.keys() if isinstance(k, str) and k.startswith('v')]:
            if 'occurrences' in catalog[k]:
                print(f'Velocity bin: {k}, captured count: {catalog[k]["occurrences"]["n_captured"]}')
                for flag, count in catalog[k]['occurrences'].items():
                    if flag != 'n_captured':
                        print(f"    {flag}: {count}")
                print("--------------------------------")

    def print_total_occurrences_summary(self):

        catalog = self.results_dictionary
        if 'total_occurrences' in catalog:
            print("Total Occurrences summary:")
            for flag, count in catalog['total_occurrences'].items():
                print(f"    {flag}: {count}")
            print("--------------------------------")

    def _rebound_entries(self, v_key):
        """Return the list of npz entries for given velocity key."""
        return self.results_dictionary.get(v_key, {}).get("rebound", []) or []

    @staticmethod
    def _as_scalar(x):
        """Best-effort convert 0-d numpy arrays to Python scalars."""
        try:
            return x.item()
        except Exception:
            return x

    def build_mask(self, v_key, predicate, as_int=False):
        """
        Build a mask over catalog[v_key]['rebound'] given a predicate(entry) -> bool.
        - v_key: e.g., 'v17'
        - predicate: callable that receives an npz entry and returns True/False
        - as_int: return 1/0 instead of True/False
        """
        entries = self._rebound_entries(v_key)
        out = []
        for e in entries:
            try:
                out.append(bool(predicate(e)))
            except Exception:
                out.append(False)
        arr = np.asarray(out, dtype=bool)
        return arr.astype(int) if as_int else arr

    def mask_flag_contains(self, v_key, substr, as_int=False):
        """Mask entries whose termination_flag contains substr."""
        def _pred(e):
            if "termination_flag" not in e.files:
                return False
            flag = self._as_scalar(e["termination_flag"])
            return isinstance(flag, str) and (substr in flag)
        return self.build_mask(v_key, _pred, as_int=as_int)

    def mask_error_contains(self, v_key, substr, as_int=False):
        """Mask entries whose errors contain substr."""
        def _pred(e):
            if "errors" not in e.files:
                return False
            errs = np.atleast_1d(e["errors"]).ravel().tolist()
            errs = [str(self._as_scalar(x)) for x in errs]
            return any(substr in msg for msg in errs)
        return self.build_mask(v_key, _pred, as_int=as_int)

    def mask_collision(self, v_key, as_int=False):
        """Mask collisions (we treat termination_flag == None as collision)."""
        def _pred(e):
            if "termination_flag" not in e.files:
                return False
            flag = self._as_scalar(e["termination_flag"])
            return flag is None
        return self.build_mask(v_key, _pred, as_int=as_int)

    def mask_where(self, v_key, key, predicate, as_int=False):
        """
        Generic field-based mask: predicate receives the field value.
        Example: analysis.mask_where('v17','lifetime', lambda t: np.isfinite(t) and t>1e6)
        """
        def _pred(e):
            if key not in e.files:
                return False
            val = self._as_scalar(e[key])
            return bool(predicate(val))
        return self.build_mask(v_key, _pred, as_int=as_int)



############ OLD VERSIONS OF STABILITY ANALYSIS FUNCTIONS ##############
    # def _get_flags_errors(self, catalog):

    #     for v in catalog.keys():

    #         mc_data = catalog[v]["mc"] if catalog[v]["mc"] is not None else None
    #         orbsim_data = catalog[v]["rebound"] if catalog[v]["rebound"] is not None else None
    #         sampled_mc_data = catalog[v]["sampled_mc"] if catalog[v]["sampled_mc"] is not None else None
    #         # Skip if no MC data
    #         if sampled_mc_data is None:
    #             continue

    #         start_info = [o["start_info"] for o in orbsim_data if "start_info" in o.files]
    #         start_distances = [o["start_distances"] for o in orbsim_data if "start_distances" in o.files]
    #         errors = [o["errors"] for o in orbsim_data if "errors" in o.files]
    #         flags = [o["termination_flag"] for o in orbsim_data if "termination_flag" in o.files]
    #         lifetimes = [o["lifetime"] for o in orbsim_data if "lifetime" in o.files]
    #         b_err_count = 0
    #         P_err_count = 0
    #         time_err_count = 0
    #         escape_C_count = 0
    #         escape_B_count = 0
    #         energy_err_count = 0
    #         preprocess_err_count = 0
    #         time_exceeded_count = 0
    #         collision_count = 0
    #         ej_mask = []
    #         coll_mask = []
    #         for idx, (sinfo, sdist, err, flag, lifetime) in enumerate(zip(start_info, start_distances, errors, flags, lifetimes)):
    #             # print(f"  Simulation {idx}: start_info={sinfo}")
    #             # print(f"  Simulation {idx}: start_distances={sdist}")

    #             if len(err) > 0:

    #                 if 'impact parameter b' in err[0]:
    #                     b_err_count += 1
                        
    #                 if 'period' in err[0]:
    #                     P_err_count += 1
    #                     # print(f"  Simulation {idx}: errors={err}")
    #                     # print(f"  Simulation {idx}: start_info={sinfo}")
    #                 if 'time' in err[0]:
    #                     time_err_count += 1

    #                 if 'preprocess' in err[0]:
    #                     preprocess_err_count += 1

    #                 if flag == None:
    #                     preprocess_err_count += 1

                
    #             if flag == None:
    #                 collision_count += 1
    #                 coll_mask.append(1)
    #             else:
    #                 coll_mask.append(0)
        
    #             if 'escape_C' in flag:
    #                 escape_C_count += 1
    #                 ej_mask.append(1)
    #             else:
    #                 ej_mask.append(0)
                    
    #             if 'escape_B' in flag:
    #                 escape_B_count += 1

    #             if 'energy' in flag:
    #                 energy_err_count += 1

    #             if 'time_exceeded' in flag:
    #                 time_exceeded_count += 1

    #         catalog[v]['termination_counts'] = {
    #             "b_err_count": b_err_count,
    #             "P_err_count": P_err_count,
    #             "time_err_count": time_err_count,
    #             "escape_C_count": escape_C_count,
    #             "escape_B_count": escape_B_count,
    #             "energy_err_count": energy_err_count,
    #             "preprocess_err_count": preprocess_err_count,
    #             "time_exceeded_count": time_exceeded_count,
    #             "collision_count": collision_count,
    #         }
    #         catalog[v]['ej_mask'] = ej_mask
    #         catalog[v]['coll_mask'] = coll_mask

    #     return catalog

    #   def _add_occurrences_per_v(self, catalog):

    #     for v in catalog.keys():
    #         mc_data = catalog[v]["mc"]
    #         orbsim_data = catalog[v]["rebound"]
    #         sampled_mc_data = catalog[v]["sampled_mc"] if catalog[v]["sampled_mc"] is not None else None

    #         # Skip if no MC data
    #         if sampled_mc_data is None:
    #             continue

    #         ej_mask = catalog[v]['ej_mask']
    #         coll_mask = catalog[v]['coll_mask']
    #         term_mask = np.array(ej_mask) + np.array(coll_mask)
    #         preprocess_err_count = catalog[v]['termination_counts']['preprocess_err_count']
    #         time_exceeded_count = catalog[v]['termination_counts']['time_exceeded_count']

    #         mA = self.system_param_dict["mA"]
    #         mB = self.system_param_dict["mB"]   
    #         aB = self.system_param_dict["aB"]
    #         eB = self.system_param_dict["eB"]
    #         vdm = self.simulation_param_dict['vDM']
    #         muA = mA * const.G.value
    #         muB = mB * const.G.value
    #         epsilon = mc_data["epsilon"]
    #         rClose = calc.r_close(epsilon, mA, mB, aB)
    #         v1Mag = calc.v_1_mag(mc_data["v_inf"], muA, muB, aB, rClose)
    #         v1Mag_au = v1Mag * (u.m / u.s).to(u.au/u.yr) # convert to au/yr
    #         vdm = (vdm*u.m/u.s).to(u.au/u.yr).value
    #         b_max = (rClose * u.m).to(u.au).value
    #         vinfinity = (mc_data["v_inf"]*u.m/u.s).to(u.au/u.yr).value
    #         n_captured = int(sampled_mc_data["total_capture_count"])
    #         n_sampled = int(sampled_mc_data["total_sample_count"])
    #         n_sampled_captures = int(sampled_mc_data["sampled_capture_count"])
    #         n_ejected = np.sum(ej_mask)
    #         n_collided = np.sum(coll_mask)
    #         n_terminated = np.sum(term_mask)

    #         capture_cross_section_total = (mc_data["sigma_MC_m2"] * u.m**2).to(u.au**2).value
    #         capture_cross_section_ejected = ((n_ejected / n_sampled) * float(np.pi) * b_max**2)
    #         capture_cross_section_collided = ((n_collided / n_sampled) * float(np.pi) * b_max**2)
    #         capture_cross_section_terminated = ((n_terminated / n_sampled) * float(np.pi) * b_max**2)

    #         lifetimes = [e["lifetime"] for e in orbsim_data if "lifetime" in e.files]
    #         ejected_lifetimes = [l for l, ej in zip(lifetimes, ej_mask) if ej == 1]
    #         collided_lifetimes = [l for l, coll in zip(lifetimes, coll_mask) if coll == 1]
    #         terminated_lifetimes = [l for l, term in zip(lifetimes, term_mask) if term == 1]


    #         total_rate = n_captured / np.sum(lifetimes )
    #         ejection_rate = len([l for l in ejected_lifetimes if l is not None]) / np.sum([l for l in ejected_lifetimes if l is not None])
    #         collision_rate = len([l for l in collided_lifetimes if l is not None]) / np.sum([l for l in collided_lifetimes if l is not None])
    #         termination_rate = len([l for l in terminated_lifetimes if l is not None]) / np.sum([l for l in terminated_lifetimes if l is not None])

    #         f_approx = (np.sqrt(2/np.pi)) * (vinfinity**2 / (vdm**3)) * np.exp(- (vinfinity)**2/(2*vdm**2))

    #         neq_ejected = f_approx * capture_cross_section_ejected * v1Mag_au / ejection_rate  # captures per au
    #         neq_collided = f_approx * capture_cross_section_collided * v1Mag_au / collision_rate  # captures per au
    #         neq_terminated = f_approx * capture_cross_section_terminated * v1Mag_au / termination_rate  # captures per au
    #         neq_total = f_approx * capture_cross_section_total * v1Mag_au / total_rate  # captures per au

    #         catalog[v]['occurrences'] = {
    #             "n_captured": n_captured,
    #             "n_sampled": n_sampled,
    #             "n_sampled_captures": n_sampled_captures,
    #             "n_ejected": n_ejected,
    #             "n_collided": n_collided,
    #             "n_terminated": n_terminated,
    #             "capture_cross_section_total": capture_cross_section_total,
    #             "capture_cross_section_ejected": capture_cross_section_ejected,
    #             "capture_cross_section_collided": capture_cross_section_collided,
    #             "capture_cross_section_terminated": capture_cross_section_terminated,
    #             "capture_cross_section_total_areaB": mc_data["sigma_MC_areaB"],
    #             "ejection_rate": ejection_rate,
    #             "collision_rate": collision_rate,
    #             "termination_rate": termination_rate,
    #             "total_rate": total_rate,
    #             "terminated_systems_neq": neq_terminated,
    #             "ejected_systems_neq": neq_ejected,
    #             "collided_systems_neq": neq_collided,
    #             "total_systems_neq": neq_total,
    #             "average_lifetime_ejected": np.mean(ejected_lifetimes) if len(ejected_lifetimes) > 0 else None,
    #             "average_lifetime_collided": np.mean(collided_lifetimes) if len(collided_lifetimes) > 0 else None,
    #             "average_lifetime_terminated": np.mean(terminated_lifetimes) if len(terminated_lifetimes) > 0 else None,
    #             "average_lifetime_total": np.mean(lifetimes) if len(lifetimes) > 0 else None,
    #         }

    #     return catalog

    # def _add_total_occurrences(self, catalog):


    #     total_captured = sum([catalog[v]['occurrences']["n_captured"] for v in catalog.keys() if 'occurrences' in catalog[v]])
    #     total_sampled = sum([catalog[v]['occurrences']["n_sampled"] for v in catalog.keys() if 'occurrences' in catalog[v]])
    #     total_ejected = sum([catalog[v]['occurrences']["n_ejected"] for v in catalog.keys() if 'occurrences' in catalog[v]])
    #     total_collided = sum([catalog[v]['occurrences']["n_collided"] for v in catalog.keys() if 'occurrences' in catalog[v]])
    #     total_terminated = sum([catalog[v]['occurrences']["n_terminated"] for v in catalog.keys() if 'occurrences' in catalog[v]])
    #     # terminated_systems_neq = sum([catalog[v]['occurrences']["terminated_systems_neq"] for v in catalog.keys() if 'occurrences' in catalog[v]])
    #     # ejected_systems_neq = sum([catalog[v]['occurrences']["ejected_systems_neq"] for v in catalog.keys() if 'occurrences' in catalog[v]])
    #     # collided_systems_neq = sum([catalog[v]['occurrences']["collided_systems_neq"] for v in catalog.keys() if 'occurrences' in catalog[v]])
    #     # total_systems_neq = sum([catalog[v]['occurrences']["total_systems_neq"] for v in catalog.keys() if 'occurrences' in catalog[v]])
    #     v_grid, F_v = calc.dict_to_sorted_arrays([catalog[v]['occurrences']["total_systems_neq"] for v in catalog.keys() if 'occurrences' in catalog[v]])
        
    #     total_systems_neq = calc.integrate_trapezoidal(x=[catalog[v]['sampled_mc']['v_inf_au_yr'] for v in catalog.keys() if catalog[v]['sampled_mc'] is not None],
    #                                                 y=[catalog[v]['occurrences']["total_systems_neq"] for v in catalog.keys() if 'occurrences' in catalog[v]],
    #                                                 )
    #     collided_systems_neq = calc.integrate_trapezoidal(x=[catalog[v]['sampled_mc']['v_inf_au_yr'] for v in catalog.keys() if catalog[v]['sampled_mc'] is not None],
    #                                                        y=[catalog[v]['occurrences']["collided_systems_neq"] for v in catalog.keys() if 'occurrences' in catalog[v]],
    #                                                        )
    #     ejected_systems_neq = calc.integrate_trapezoidal(x=[catalog[v]['sampled_mc']['v_inf_au_yr'] for v in catalog.keys() if catalog[v]['sampled_mc'] is not None],
    #                                                       y=[catalog[v]['occurrences']["ejected_systems_neq"] for v in catalog.keys() if 'occurrences' in catalog[v]],
    #                                                       )
    #     terminated_systems_neq = calc.integrate_trapezoidal(x=[catalog[v]['sampled_mc']['v_inf_au_yr'] for v in catalog.keys() if catalog[v]['sampled_mc'] is not None],
    #                                                         y=[catalog[v]['occurrences']["terminated_systems_neq"] for v in catalog.keys() if 'occurrences' in catalog[v]],
    #                                                         )
    #     catalog['total_occurrences'] = {
    #         "total_captured": total_captured,
    #         "total_sampled": total_sampled,
    #         "total_ejected": total_ejected,
    #         "total_collided": total_collided,
    #         "total_terminated": total_terminated,
    #         "terminated_systems_neq": terminated_systems_neq,
    #         "ejected_systems_neq": ejected_systems_neq,
    #         "collided_systems_neq": collided_systems_neq,
    #         "total_systems_neq": total_systems_neq,
    #     }

    #     return catalog