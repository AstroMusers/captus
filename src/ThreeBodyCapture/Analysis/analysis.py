import os
import glob
import numpy as np
from rebound import data
import src.Utils.calculations as calc
from astropy import units as u
import astropy.constants as const
import src.ThreeBodyCapture.Configurations.configuration as config
import src.Utils.misc as misc
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
import copy
import pickle
import json

REPO_ROOT = misc._resolve_repo_root()
print(f'Using REPO_ROOT: {REPO_ROOT}')

class Analysis:
    def __init__(self, name, configuration, load='All', load_rebound=True, rng=None, results_dir_mc=None, results_dir_rebound=None):

        self.name = name
        self.system_param_dict = configuration.get_system_param(all=True)
        self.simulation_param_dict = configuration.get_simulation_param(all=True)
        self.mc_sample_size = self.simulation_param_dict['sample_size']
        self.load = load
        self.load_rebound = load_rebound

        if self.load != 'All':
            self.mc_sample_size = min(self.mc_sample_size, self.load)

        if rng is None:
            seed = self.system_param_dict['seed_base']
            self.rng = np.random.default_rng(seed)

        if self.simulation_param_dict['importance_sampling']:
            trials = self.simulation_param_dict['max_trials']
        else:
            trials = self.simulation_param_dict['trials']


        if results_dir_mc is not None:
            if isinstance(results_dir_mc, list):
                self.mc_dir = [os.path.join(REPO_ROOT, dir) for dir in results_dir_mc]
            else:
                self.mc_dir = os.path.join(REPO_ROOT, results_dir_mc)
        
        else:
            self.mc_dir = os.path.join(REPO_ROOT, f'runs/{name}/Monte_Carlo_Results/')

        if results_dir_rebound is not None:
            if isinstance(results_dir_rebound, list):
                self.rebound_dir = [os.path.join(REPO_ROOT, dir) for dir in results_dir_rebound]
            else:
                self.rebound_dir = os.path.join(REPO_ROOT, results_dir_rebound)
        else:
            self.rebound_dir = os.path.join(REPO_ROOT, f'runs/{name}/Rebound_Simulation_Results/')

        print(f'Loading MC results from: {self.mc_dir}')
        print(f'Loading Rebound results from: {self.rebound_dir}')

        try:
            self.mc_results = self._load_mc_results()
        except Exception as e:
            print(f"Error loading MC results: {e}")
        try:
            self.sampled_mc_results = self._get_sampled_mc_results()
        except Exception as e:
            print(f"Error loading sampled MC results: {e}")

        if self.load_rebound is True:
            try:
                self.rebound_results = self._load_rebound_results()
            except Exception as e:           
                print(f"Error loading Rebound results: {e}")
            try:
                self.results_dictionary = self.get_combined_dictionary()
            except Exception as e:
                print(f"Error loading results: {e}")
        else:
            self.rebound_results = {}
            self.results_dictionary = {}
            print("Rebound results loading is disabled. Only MC results will be processed. Results dictionary not generated.")

    def get_system_parameters(self):
        return self.system_param_dict
    
    def _load_mc_results(self):

        mc_results = {}
        if self.load == 'All':
            if isinstance(self.mc_dir, list):
                mc_files = []
                for dir in self.mc_dir:
                    mc_files.extend(glob.glob(os.path.join(dir, "monte_carlo_results_*.npz")))


                for mc_path in mc_files:
                    with np.load(mc_path, allow_pickle=True) as mc_probe:
                        v_inf = float(mc_probe["v_inf"])

                    vk = self._vkey_from_vinf(v_inf)
                    data = np.load(mc_path, allow_pickle=True)
                    if vk in mc_results:
                        print(f"Extending existing MC results for {vk} with data from {mc_path}"
                                f" (existing count: {len(mc_results[vk])}, new count: {len(data)+len(mc_results[vk])})")
                        mc_results[vk]['data_2'] = data

                    else:
                        mc_results[vk] = {}
                        mc_results[vk]['data_1'] = data
                    
                return mc_results

            else:
                mc_files = glob.glob(os.path.join(self.mc_dir, "monte_carlo_results_*.npz"))
        else:
            mc_files = glob.glob(os.path.join(self.mc_dir, f"monte_carlo_results_*.npz"))[:self.load]
        
        for mc_path in mc_files:
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
        
        # Get all v folders
        if isinstance(self.rebound_dir, list):
            all_v_folders = []
            for dir in self.rebound_dir:
                all_v_folders.extend(sorted([
                    f for f in os.listdir(dir) 
                    if f.startswith("v") and os.path.isdir(os.path.join(dir, f))
                ]))
            
        else:
            all_v_folders = sorted([
                f for f in os.listdir(self.rebound_dir) 
                if f.startswith("v") and os.path.isdir(os.path.join(self.rebound_dir, f))
            ])
        
        # Limit folders if needed
        v_folders = all_v_folders if self.load == 'All' else all_v_folders[:self.load]
        for v_folder in v_folders:
            if isinstance(self.rebound_dir, list):
                all_files = []
                for dir in self.rebound_dir:
                    v_path = os.path.join(dir, v_folder)
                    all_files.extend(sorted(glob.glob(os.path.join(v_path, "*.npz"))))
            else:
                v_path = os.path.join(self.rebound_dir, v_folder)
                all_files = sorted(glob.glob(os.path.join(v_path, "*.npz")))
            
            # ✅ Limit files per folder based on load parameter
            if self.load == 'All':
                files_to_load = all_files
            else:
                files_to_load = all_files[:self.load]  # Load only 'load' files per folder

            if self.load_rebound is False:
                files_to_load = []  # Don't load any files if load_rebound is False
                print(f"Skipping Rebound files for {v_folder}.")

            entries = []
            if len(files_to_load) == 0:
                files_to_load = []  # Ensure no files are loaded if load_rebound is False
                print(f"Warning: No Rebound files found in {v_path}.")
            else:
                for f in files_to_load:
                    try:
                        data = np.load(f, allow_pickle=True)
                        entries.append(data)
                    except Exception as e:
                        print(f"Failed to load Rebound {f}: {e}")
            
            v_key = self._vkey_from_vinf(float(v_folder[1:])*1e3)
            rebound_results[v_key] = entries
            # print(f"✓ Found in rebound: {len(entries)} entries for {v_key}")  # Debug info

        return rebound_results
    

    def _vkey_from_vinf(self, v_inf_mps: float) -> str:
    # km/s rounded to nearest integer, prefixed with 'V'
        return f"V{int(np.rint(v_inf_mps/1e3))}"
    
    def get_combined_dictionary(self, r=8, time_averages=False, update=False, use_cached_data=True):

        if hasattr(self, 'results_dictionary') and use_cached_data:
            if not update and not time_averages:
                print("Using cached results dictionary.")
                return self.results_dictionary
            
            if update:
                print("Using cached data in results dictionary as base for update, (MC and Rebound data not relaoded).")
                catalog = self.results_dictionary  # start with existing catalog and update derived metrics
                for v_key in catalog['v_keys']:
                    rebound_npz_list = self.rebound_results.get(v_key, None)
                    masks = self._get_masks(rebound_npz_list)
                    sampled_mc_results = self.sampled_mc_results.get(v_key, None) if hasattr(self, 'sampled_mc_results') else None
                    catalog[v_key]['occurrences'] = self._get_occurrences(catalog[v_key]['mc'], rebound_npz_list, sampled_mc_results, masks, r)
                catalog['total_occurrences_trapz'] = self._total_occurrences_trapz(catalog)
                catalog['total_occurrences_gl'] = self._total_occurrences_gl(catalog)
                catalog['total_pbh_neq_n'] = self._get_Neq_at_r_f(catalog, rf=r)
                catalog['total_termination_counts'] = self._total_termination_counts(catalog)
                catalog['errors'] = self._get_errors(catalog)

                return catalog
            
            if time_averages:
                if hasattr(self.results_dictionary, 'semi_major_axis_time_avg'):
                    print("Time averges requested exist in results dictionary, using cached values.")
                    return self.results_dictionary
                else:
                    print("Computing time-averaged parameters for cached data. This may take some time...")
                    catalog = self.results_dictionary  # start with existing catalog and update derived metrics
                    for v_key in catalog['v_keys']:
                        rebound_npz_list = catalog[v_key]['rebound']
                        masks = catalog[v_key]['masks']
                        sampled_mc_results = self.sampled_mc_results.get(v_key, None) if hasattr(self, 'sampled_mc_results') else None
                        catalog[v_key]["semi_major_axis_time_avg"] = self._get_time_averaged_parameter(rebound_npz_list, parameter_key="semi_major_axes")
                        catalog[v_key]["eccentricity_time_avg"] = self._get_time_averaged_parameter(rebound_npz_list, parameter_key="eccentricities")
                        catalog[v_key]["orbital_period_time_avg"] = self._get_time_averaged_parameter(rebound_npz_list, parameter_key="orbital_periods")

                    return catalog

        else:
            catalog = {**self.system_param_dict, **self.simulation_param_dict}  # start with system and simulation params at top level
            sorted_keys = sorted(self.mc_results.keys(), key=lambda x: int(x[1:]))  # sort by integer value after 'V'
        
            # print(f"Combining results for v bins: {sorted_keys}")
            for v_key in sorted_keys:
                # if isinstance(self.mc_dir, list):
                #     # print(f"MC data for {v_key} has data_1 with n_captured = {self.mc_results[v_key]['data_1']['n_captured']}")
                #     mc_npz = self.mc_results[v_key]['data_1']
                # else:
                #     mc_npz = self.mc_results.get(v_key, None)
                mc_npz = self.mc_results.get(v_key, None)
                sampled_mc_results = self.sampled_mc_results.get(v_key, None) if hasattr(self, 'sampled_mc_results') else None
                if mc_npz is None or sampled_mc_results is None:
                    print(f"Warning: No MC data for {v_key}.")
                    continue
                
                rebound_npz_list = self.rebound_results.get(v_key, None)
                # print(f"Combining {v_key}: MC data found, {len(rebound_npz_list)} Rebound entries found.")
                if rebound_npz_list is None or len(rebound_npz_list) == 0:
                    print(f"Warning: No Rebound data for {v_key}.")

                masks = self._get_masks(rebound_npz_list)
                
                catalog[v_key] = {

                    **sampled_mc_results,  # include sampled MC metadata at top level for easy access
                    "v_inf_au_yr": (float(mc_npz['v_inf']) * u.m / u.s).to(u.au / u.yr).value if mc_npz is not None else None,
                    "sampled_capture_count": len(sampled_mc_results['idx']) if sampled_mc_results is not None else 0,


                    # Raw data
                    "mc": mc_npz,
                    "rebound": rebound_npz_list,

                    # Compute derived data per-entry
                    "termination_counts": self._get_termination_counts(rebound_npz_list),
                    "masks": masks,
                    "occurrences": self._get_occurrences(mc_npz, rebound_npz_list, sampled_mc_results, masks, r),

                    
                    # # Time averaged data
                    "semi_major_axis_time_avg": None if not time_averages else self._get_time_averaged_parameter(rebound_npz_list, parameter_key="semi_major_axes"),
                    "eccentricity_time_avg": None if not time_averages else self._get_time_averaged_parameter(rebound_npz_list, parameter_key="eccentricities"),
                    "orbital_period_time_avg": None if not time_averages else self._get_time_averaged_parameter(rebound_npz_list, parameter_key="orbital_periods"),


                }

            # Compute overall statistics
            v_keys = sorted_keys
            catalog['total_sample_count'] = sum(catalog[k]['sample_number'] for k in v_keys)
            catalog['v_keys'] = v_keys
            catalog['total_capture_count'] = sum(catalog[k]['n_captured'] for k in v_keys)
            # catalog['total_sampled_capture_count'] = sum(catalog[k]['sampled_capture_count'] for k in v_keys)
            # Total occurrences computed once after all entries
            catalog['total_occurrences_trapz'] = self._total_occurrences_trapz(catalog)
            catalog['total_occurrences_gl'] = self._total_occurrences_gl(catalog)
            catalog['total_pbh_neq_n'] = self._get_Neq_at_r_f(catalog, rf=r)
            catalog['total_termination_counts'] = self._total_termination_counts(catalog)
            catalog['errors'] = self._get_errors(catalog)

            return catalog

    def get_mc_results(self):
        return self.mc_results

    def _get_sampled_mc_results(self):
        sampled_mc = {}
        sorted_keys = sorted(self.mc_results.keys(), key=lambda x: int(x[1:]))

        for v in sorted_keys:
            data = self.mc_results[v]

            keys = data.files if hasattr(data, "files") else data.keys()

            n_captured = int(np.asarray(data["n_captured"]).item())

            if n_captured == 0:
                continue

            if n_captured <= self.mc_sample_size:
                idx = np.arange(n_captured)
            else:
                idx = self.rng.choice(n_captured, size=self.mc_sample_size, replace=False)

            sampled_dict = {}

            for key in keys:
                val = data[key]

                if key.startswith("cap"):
                    sampled_dict[key] = val[idx]
                else:
                    sampled_dict[key] = val

            sampled_dict["idx"] = idx
            sampled_mc[v] = sampled_dict

            # print(f"v_inf = {v} km/s: sampled {len(idx)} captured objects")

        return sampled_mc
    

    def get_sampled_mc_results(self):
        if not hasattr(self, 'sampled_mc_results'):
            self._get_sampled_mc_results()
        return self.sampled_mc_results

    def _get_time_averaged_parameter(self, rebound_list, parameter_key="semi_major_axes"):
        """
        Compute time-averaged parameter across rebound entries.
        Optimized for speed using vectorization where possible.
        """
        if rebound_list is None or len(rebound_list) == 0:
            print("Warning: No Rebound data to compute time-averaged parameters.")
            return None
        
        # ✅ OPTIMIZATION 1: List comprehension (faster than loop + append)
        time_avg_list = [
            np.mean(entry[parameter_key]) 
            for entry in rebound_list 
            if hasattr(entry, 'files') and parameter_key in entry.files
        ]
        
        if len(time_avg_list) == 0:
            print(f"Warning: No {parameter_key} found in Rebound entries.")
            return None
        
        # ✅ OPTIMIZATION 2: Return as numpy array directly (no intermediate list)
        return np.asarray(time_avg_list, dtype=np.float64)

    def update_results_dictionary(self, r=8):
        self.results_dictionary = self.get_combined_dictionary(r, update=True)

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
        if rebound_list is None:
            print("Warning: No Rebound data to compute termination counts.")
            return counts
        
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
        ✅ OPTIMIZED: Pre-allocate numpy arrays instead of list appends (2-3x faster)
        """
        if rebound_list is None:
            print("Warning: No Rebound data to compute masks.")
            return {
                "ejection": np.array([], dtype=int),
                "collision": np.array([], dtype=int),
                "termination": np.array([], dtype=int),
                "completed": np.array([], dtype=int),
            }
        
        # ✅ OPTIMIZATION: Pre-allocate numpy arrays instead of list + append + convert
        n = len(rebound_list)
        ej_mask = np.zeros(n, dtype=int)
        coll_mask = np.zeros(n, dtype=int)
        complete_with_termination_mask = np.zeros(n, dtype=int)
        complete_mask = np.zeros(n, dtype=int)
        
        for i, entry in enumerate(rebound_list):
            if "termination_flag" not in entry.files:
                continue
                
            flag = self._as_scalar(entry["termination_flag"])
            
            if isinstance(flag, str):
                if 'escape_C' in flag:
                    ej_mask[i] = 1
                    complete_with_termination_mask[i] = 1
                elif 'collision' in flag:
                    coll_mask[i] = 1
                    complete_with_termination_mask[i] = 1
                elif 'completed' in flag:
                    complete_mask[i] = 1

        return {
            "ejection": ej_mask,
            "collision": coll_mask,
            "termination": complete_with_termination_mask,
            "completed": complete_mask,
        }

    

    def _get_occurrences(self, mc_npz, rebound_list, sampled_mc, masks, r):
        """
        Compute occurrence metrics for a single v bin.
        Pure function: takes inputs, returns occurrences dict.
        """
        mA = self.system_param_dict["mA"]
        mB = self.system_param_dict["mB"]   
        mC = self.system_param_dict["mC"]
        aB = self.system_param_dict["aB"]
        vdm = self.simulation_param_dict['vDM']
        muA = calc.standard_gravitational_parameter(mA, mC)
        muB = calc.standard_gravitational_parameter(mB, mC)

        n_sampled = int(sampled_mc["sample_number"])
        n_sampled_captures = int(sampled_mc["n_captured"])
        check2 = int(sampled_mc['checks'][2])

        epsilon = sampled_mc["epsilon"] 
        b = sampled_mc["cap_b"]  # in meters
        # b_max = np.max(sampled_mc["cap_bmax"])  # in meters
        # b_min = np.min(sampled_mc["cap_bmin"])  # in meters
        v_inf = sampled_mc["v_inf"]  # in m/s
        v_inf_km_s = (v_inf*u.m/u.s).to(u.km/u.s).value  # in km/s
        v_inf_au_yr = (v_inf*u.m/u.s).to(u.au/u.yr).value
        vdm_au_yr = (vdm*u.m/u.s).to(u.au/u.yr).value
        b_au = (b * u.m).to(u.au).value

        rClose = calc.r_close(epsilon, muA, muB, aB)
        aC = aB - rClose
        v1Mag = calc.v_1_mag(v_inf, muA, muB, aC, rClose)
        v1Mag_au = (v1Mag*u.m / u.s).to(u.au/u.yr).value

        # b_max_sampled = np.max(b_au)
        # b_min_sampled = np.min(b_au)
        b_min_sampled = sampled_mc['cap_bmin']
        b_max_sampled = sampled_mc['cap_bmax']

        sigma_au2 = np.array(2*np.pi*b_au**2)

        # f_approx = (np.sqrt(2/np.pi)) * (v_inf_au_yr**2 / (vdm_au_yr**3)) * np.exp(- (v_inf_au_yr)**2/(2*vdm_au_yr**2))
        f_approx_local = calc.v_pbh_pdf(v_inf_km_s, r = r) # local PDF value at this v_inf, in units of 1/(au/yr)
        f_approx_local = f_approx_local.value
        # f_approx_local = f_approx
        # sigma_cap = sampled_mc["sigma_MC_dsigma_au2"] * n_sampled / check2
        sigma_cap_array, sigma_cap = calc.capture_cross_section(b_mins=b_min_sampled, b_maxs=b_max_sampled, b_cap=b, n_sampled=check2)  # in au^2
        # sigma_cap_recalc = calc.capture_cross_section_MC(b, n_sampled, b_min, b_max)
        # sigma_cap = (sampled_mc["sigma_MC_dsigma_captured_m2"] * u.m**2).to(u.au**2).value
        # sigma_cap = np.sum((sampled_mc["capture_cross_sections_captured"]* u.m**2).to(u.au**2).value)  # already in au^2
        n_captured = len(sampled_mc["idx"])

        # b_max = np.max(b_max_sampled*u.m).to(u.au).value
        # b_min = np.min(b_min_sampled*u.m).to(u.au).value
        # sigma_cap_array = 2 * np.pi * (b_max - b_min)  * (b_au / check2)  # in au^2

        capture_cross_section_total = sigma_cap.value
        capture_cross_section_total_array = sigma_cap_array.value

        capture_rate = f_approx_local * capture_cross_section_total * v1Mag_au
        # capture_rate_r = f_approx * capture_cross_section_total * v1Mag_au
        capture_rate_array = f_approx_local * capture_cross_section_total_array * v1Mag_au

        if rebound_list is None:
            print("Warning: No Rebound data to compute termination rate and Neq occurrences and no masks.")
            return {
                "n_captured": n_captured,
                "n_sampled": n_sampled,
                "n_sampled_captures": n_sampled_captures,
                "n_ejected": None,
                "n_collided": None,
                "n_terminated": None,
                "capture_cross_section_total": capture_cross_section_total,
                "capture_cross_section_total_array": capture_cross_section_total_array,
                "capture_cross_section_ejected": None,
                "capture_cross_section_collided": None,
                "capture_cross_section_terminated": None,
                "capture_rate": capture_rate,
                "ejection_rate": None,
                "collision_rate": None,
                "termination_rate": None,
                "total_rate": None,
                "neq_ejected": None,
                "neq_collided": None,
                "neq_terminated": None,
                "neq_total": None,
                "neq_total_array": None,
                "dsigma_au2": sigma_au2,
                "capture_rate_array": capture_rate_array,

            }

        ej_mask = masks["ejection"]
        coll_mask = masks["collision"]
        term_mask = masks["termination"]
        complete_mask = masks["completed"]

        n_ejected = int(np.sum(ej_mask))
        n_collided = int(np.sum(coll_mask))
        n_terminated = int(np.sum(term_mask))
        n_completed = int(np.sum(complete_mask))
        frac_ejected = n_ejected / n_sampled if n_sampled > 0 else 0.0
        frac_collided = n_collided / n_sampled if n_sampled > 0 else 0.0
        frac_terminated = n_terminated / n_sampled if n_sampled > 0 else 0.0
        frac_completed = n_completed / n_sampled if n_sampled > 0 else 0.0
        

        # # capture_cross_section_exclude_coll = capture_cross_section_total - ((capture_cross_section_total/b_max**2) * b_min**2)  # in au^2
        # capture_cross_section_ejected = (frac_ejected * capture_cross_section_total) if n_sampled > 0 else 0.0
        # capture_cross_section_collided = (frac_collided * capture_cross_section_total) if n_sampled > 0 else 0.0
        # capture_cross_section_terminated = (frac_terminated * capture_cross_section_total) if n_sampled > 0 else 0.0
        # capture_cross_section_completed = (frac_completed * capture_cross_section_total) if n_sampled > 0 else 0.0
        # capture_cross_section_total = ((n_completed + n_terminated) / n_sampled_captures) * capture_cross_section_total
        # ✅ OPTIMIZATION: Use numpy boolean indexing (faster than list comprehensions with zip)
        lifetimes = [e["lifetime"] for e in rebound_list if "lifetime" in e.files] if rebound_list is not None else []
        terminations = [1/l if l is not None else 0 for l in lifetimes]
        
        # Convert to numpy array for vectorized boolean masking
        lifetimes_arr = np.asarray(lifetimes)
        ejected_lifetimes = lifetimes_arr[ej_mask == 1] if len(lifetimes_arr) > 0 else np.array([])
        collided_lifetimes = lifetimes_arr[coll_mask == 1] if len(lifetimes_arr) > 0 else np.array([])
        terminated_lifetimes = lifetimes_arr[term_mask == 1] if len(lifetimes_arr) > 0 else np.array([])
        
        lifetimes_average = np.mean(lifetimes) if len(lifetimes) > 0 else None
        total_rate = 1 / lifetimes_average if lifetimes_average is not None and lifetimes_average > 0 else 0.0

        lifetimes_ejected_average = np.mean(ejected_lifetimes) if len(ejected_lifetimes) > 0 else None
        ejection_rate = 1 / lifetimes_ejected_average if lifetimes_ejected_average is not None and lifetimes_ejected_average > 0 else 0.0   
        lifetimes_collided_average = np.mean(collided_lifetimes) if len(collided_lifetimes) > 0 else None
        collision_rate = 1 / lifetimes_collided_average if lifetimes_collided_average is not None and lifetimes_collided_average > 0 else 0.0
        lifetimes_terminated_average = np.mean(terminated_lifetimes) if len(terminated_lifetimes) > 0 else None
        termination_rate = 1 / lifetimes_terminated_average if lifetimes_terminated_average is not None and lifetimes_terminated_average > 0 else 0.0   

        # # total_rate = (len([l for l in lifetimes if l is not None]) / 
        # #               np.sum([l for l in lifetimes if l is not None])) if len(lifetimes) > 0 else 0.0
        # ejection_rate = (len([l for l in ejected_lifetimes if l is not None]) /
        #                 np.sum([l for l in ejected_lifetimes if l is not None])) if len(ejected_lifetimes) > 0 else 0.0
        # collision_rate = (len([l for l in collided_lifetimes if l is not None]) / 
        #                  np.sum([l for l in collided_lifetimes if l is not None])) if len(collided_lifetimes) > 0 else 0.0
        # termination_rate = (len([l for l in terminated_lifetimes if l is not None]) / 
        #                    np.sum([l for l in terminated_lifetimes if l is not None])) if len(terminated_lifetimes) > 0 else 0.0

        neq_ejected = frac_ejected * capture_rate / ejection_rate if ejection_rate > 0 else 0.0
        neq_collided = frac_collided * capture_rate / collision_rate if collision_rate > 0 else 0.0
        neq_terminated = capture_rate / termination_rate if termination_rate > 0 else 0.0
        neq_total = capture_rate / total_rate if total_rate > 0 else 0.0
        # neq_total_r = capture_rate_r / total_rate if total_rate > 0 else 0.0
        neq_total_array = capture_rate_array / total_rate if total_rate > 0 else 0.0

        return {
            "n_captured": n_captured,
            "n_sampled": n_sampled,
            "n_sampled_captures": n_sampled_captures,
            "n_ejected": n_ejected,
            "n_collided": n_collided,
            "n_terminated": n_terminated,
            "capture_cross_section_total": capture_cross_section_total,
            "capture_cross_section_total_array": capture_cross_section_total_array,
            # "capture_cross_section_ejected": capture_cross_section_ejected,
            # "capture_cross_section_collided": capture_cross_section_collided,
            # "capture_cross_section_terminated": capture_cross_section_terminated,
            "capture_rate": capture_rate,
            "ejection_rate": ejection_rate,
            "collision_rate": collision_rate,
            "termination_rate": termination_rate,
            "total_rate": total_rate,
            "terminated_systems_neq": neq_terminated,
            "ejected_systems_neq": neq_ejected,
            "collided_systems_neq": neq_collided,
            "total_systems_neq": neq_total,
            "total_systems_neq_array": neq_total_array,
            # "total_systems_neq_r": neq_total_r,
            "average_lifetime_collided": np.mean(collided_lifetimes) if len(collided_lifetimes) > 0 else None,
            "average_lifetime_terminated": np.mean(terminated_lifetimes) if len(terminated_lifetimes) > 0 else None,
            "average_lifetime_total": np.mean(lifetimes) if len(lifetimes) > 0 else None,
            "terminations" : terminations,
            "dsigma_au2": sigma_au2,
            "capture_rate_array": capture_rate_array,
            "lifetimes_array": np.array(lifetimes),
        }
    def _total_occurrences_trapz(self, catalog):
        """
        Compute aggregate occurrences across all v bins.
        Uses canonical v_inf_au_yr from catalog entries.
        """
        v_keys = [k for k in catalog.keys() if isinstance(k, str) and k.startswith('V')]

        x_vals = [catalog[v]['v_inf_au_yr'] for v in v_keys if 'v_inf_au_yr' in catalog[v] and 'occurrences' in catalog[v]]
        total_captured = sum([catalog[v]['occurrences']["n_captured"] for v in v_keys if 'occurrences' in catalog[v]])
        y_capture_rate = [catalog[v]['occurrences']["capture_rate"] for v in v_keys if 'occurrences' in catalog[v]]
        total_cap_rate = calc.integrate_trapezoidal(x=x_vals, y=y_capture_rate)

        if len(self.rebound_results) == 0 or self.load_rebound is False:
            print("Warning: No Rebound results found. Only calculating the capture rate totals without occurrence integration.")
            return {
                "total_captured": total_captured,
                "total_sampled": 0,
                "total_ejected": 0,
                "total_collided": 0,
                "total_terminated": 0,
                "terminated_systems_neq": 0.0,
                "ejected_systems_neq": 0.0,
                "collided_systems_neq": 0.0,
                "total_systems_neq": 0.0,
                "total_systems_neq_array": 0.0,
                "total_capture_rate": total_cap_rate,
            }
        
        total_sampled = sum([catalog[v]['occurrences']["n_sampled"] for v in v_keys if 'occurrences' in catalog[v]])
        total_ejected = sum([catalog[v]['occurrences']["n_ejected"] for v in v_keys if 'occurrences' in catalog[v]])
        total_collided = sum([catalog[v]['occurrences']["n_collided"] for v in v_keys if 'occurrences' in catalog[v]])
        total_terminated = sum([catalog[v]['occurrences']["n_terminated"] for v in v_keys if 'occurrences' in catalog[v]])
        # Use canonical v_inf_au_yr from catalog entries (no nested lookup needed)
        y_total = [catalog[v]['occurrences']["total_systems_neq"] for v in v_keys if 'occurrences' in catalog[v]]
        # y_total_2 = [catalog[v]['occurrences']["total_systems_neq_2"] for v in v_keys if 'occurrences' in catalog[v]]
        y_collided = [catalog[v]['occurrences']["collided_systems_neq"] for v in v_keys if 'occurrences' in catalog[v]]
        y_ejected = [catalog[v]['occurrences']["ejected_systems_neq"] for v in v_keys if 'occurrences' in catalog[v]]
        y_terminated = [catalog[v]['occurrences']["terminated_systems_neq"] for v in v_keys if 'occurrences' in catalog[v]]
        total_systems_neq = calc.integrate_trapezoidal(x=x_vals, y=y_total)
        # total_systems_neq_2 = calc.integrate_trapezoidal(x=x_vals, y=y_total_2)
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
            # "total_systems_neq_2": total_systems_neq_2,
            "total_capture_rate": total_cap_rate,
        }

    def _total_occurrences_gl(self, catalog):
        """
        Build per-v arrays and integrate using Gauss-Legendre quadrature to make totals
        independent of sampling density. Returns a dict of integrated totals.

        - x: velocity array built from v_inf_au_yr (float)
        - y fields integrated: total_systems_neq, collided_systems_neq,
          ejected_systems_neq, terminated_systems_neq
        - Sorting: by x ascending, dropping NaNs
        """
        def _f(val):
            try:
                return float(np.asarray(val).item())
            except Exception:
                return float(val)
            
        x_vals = []    
        y_capture_rate = []
        for v_key, entry in catalog.items():
            if not isinstance(v_key, str) or not v_key.startswith('V'):
                continue

            occ = entry.get('occurrences')    
            x = entry.get('v_inf_au_yr')
            r_cap = occ.get('capture_rate')
            cap_rate = _f(r_cap)
            y_capture_rate.append(cap_rate)
            x_vals.append(x)

        order = np.argsort(x_vals)
        x = np.asarray(x_vals)[order]
        y_capture_rate = np.asarray(y_capture_rate)[order]
        captured_rate_gl = calc.integrate_gauss_legendre(x, y_capture_rate, n=100)


        if len(self.rebound_results) == 0 or self.load_rebound is False:
            print("Warning: No Rebound results found. Only calculating the capture rate totals without occurrence integration.")
            return {
                "terminated_systems_neq": 0.0,
                "ejected_systems_neq": 0.0,
                "collided_systems_neq": 0.0,
                "total_systems_neq": 0.0,
                "total_systems_neq_2": 0.0,
                "capture_rate_gl": captured_rate_gl,
                "termination_rate_gl": 0.0,
            }
        
        y_total = []
        y_total_r = []
        y_collided = []
        y_ejected = []
        y_terminated = []
        y_termination_rate = []

        for v_key, entry in catalog.items():
            if not isinstance(v_key, str) or not v_key.startswith('V'):
                continue
            occ = entry.get('occurrences')
            if occ is None:
                continue
            
            v = entry.get('v_inf_au_yr')
            if v is None:
                continue
                
            try:
                v = float(np.asarray(v).item())
            except Exception:
                try:
                    v = float(v)
                except Exception:
                    continue

            
            total_neq = occ.get('total_systems_neq')
            # total_neq_2 = occ.get('total_systems_neq_2')
            total_neq_r = occ.get('total_systems_neq_r')
            collided_neq = occ.get('collided_systems_neq')
            ejected_neq = occ.get('ejected_systems_neq')
            terminated_neq = occ.get('terminated_systems_neq')
            r_term = occ.get('total_rate')

            # if any(v is None for v in (total_neq, collided_neq, ejected_neq, terminated_neq)):
            #     continue

            t = _f(total_neq)
            # t2 = _f(total_neq_2)
            # t_r = _f(total_neq_r)
            c = _f(collided_neq)
            e = _f(ejected_neq)
            r = _f(terminated_neq)
            term_rate = _f(r_term)

            # if not (np.isfinite(x) and np.isfinite(t)  and np.isfinite(c) and np.isfinite(e) and np.isfinite(r)):
            #     continue

            y_total.append(t)
            # y_total_2.append(t2)
            # y_total_r.append(t_r)
            y_collided.append(c)
            y_ejected.append(e)
            y_terminated.append(r)
            y_termination_rate.append(term_rate)
            # x_vals.append(x)


        y_total = np.asarray(y_total)[order]
        # y_total_2 = np.asarray(y_total_2)[order]
        # y_total_r = np.asarray(y_total_r)[order]
        y_collided = np.asarray(y_collided)[order]
        y_ejected = np.asarray(y_ejected)[order]
        y_terminated = np.asarray(y_terminated)[order]
        y_termination_rate = np.asarray(y_termination_rate)[order]
        
        total_gl = calc.integrate_gauss_legendre(x, y_total, n=100)
        # total_2_gl = calc.integrate_gauss_legendre(x, y_total_2, n=100)
        # total_r_gl = calc.integrate_gauss_legendre(x, y_total_r, n=100)
        collided_gl = calc.integrate_gauss_legendre(x, y_collided, n=100)
        ejected_gl = calc.integrate_gauss_legendre(x, y_ejected, n=100)
        terminated_gl = calc.integrate_gauss_legendre(x, y_terminated, n=100)
        termination_rate_gl = calc.integrate_gauss_legendre(x, y_termination_rate, n=100)

        return {
            'total_systems_neq_gl': total_gl,
            # 'total_systems_neq_2_gl': total_2_gl,
            # 'total_systems_neq_r_gl': total_r_gl,
            'collided_systems_neq_gl': collided_gl,
            'ejected_systems_neq_gl': ejected_gl,
            'terminated_systems_neq_gl': terminated_gl,
            'v_bins_used': int(len(x)),
            'capture_rate_gl': captured_rate_gl,
            'termination_rate_gl': termination_rate_gl,
        }

    def _get_errors(self, catalog, percentile=0.95):
        v_keys = [v for v in catalog.keys() if 'V' in v]
        v_inf_au_yr = [catalog[v]['v_inf_au_yr'] for v in v_keys]
        Neq_std_list, Caprate_std_list, Capcrossec_std_list = [], [], []
        Neq_list, Caprate_list, Capcrossec_list = [], [], []
        Neq_tot = catalog['total_occurrences_gl']['total_systems_neq_gl'] if 'total_systems_neq_gl' in catalog['total_occurrences_gl'] else None
        Caprate_tot = catalog['total_occurrences_gl']['capture_rate_gl'] if 'capture_rate_gl' in catalog['total_occurrences_gl'] else None
        for v in v_keys:
            neq = catalog[v]['occurrences']['total_systems_neq'] if 'total_systems_neq' in catalog[v]['occurrences'] else None
            cap = catalog[v]['occurrences']['capture_rate'] if 'capture_rate' in catalog[v]['occurrences'] else None
            cap_tot = catalog[v]['occurrences']['capture_cross_section_total']
            b_min_sampled = catalog[v]['cap_bmin']
            b_max_sampled = catalog[v]['cap_bmax']
            bi = catalog[v]['cap_b'] if 'cap_b' in catalog[v] else None
            n_samp = catalog[v]['checks'][2]
            sigma, se, ci = calc.capture_cross_section_and_error(bi, b_min_sampled, b_max_sampled, n_sampled=n_samp, conf=percentile)
            # Check if cap_tot and sigma matches
            if not np.isclose(cap_tot, sigma, rtol=0.01):
                print(f"v={v}: Warning: capture cross-section total = {cap_tot:.2e} au^2, sigma from error calculation = {sigma:.2e} au^2 capture cross-section total and sigma do not match within 1% relative tolerance.")
            Capcrossec_std_list.append((sigma, se, ci)) 
            Caprate_list.append(cap)
            Neq_list.append(neq) 
        sigmas = np.array([s[0] for s in Capcrossec_std_list])
        ses = np.array([s[1] for s in Capcrossec_std_list])
        cis = np.array([s[2] for s in Capcrossec_std_list])
        cap_crossec = np.sum(sigmas)
        # print(f" Total capture crossection: {cap_crossec}")
        caprate, cap_se, cap_ci = calc.rate_and_error(Caprate_list, sigmas, ses, conf=percentile) 
        total_cap_rate, total_cap_err, ci68, ci95, alpha, qvals = calc.integrated_rate_and_error(v_inf_au_yr, caprate, cap_se, n_gl=100, q_vals=[0.025, 0.5, 0.975])
        if not np.isclose(total_cap_rate, Caprate_tot, rtol=0.01):
            print(f"Warning: Integrated capture rate {total_cap_rate:.2e} does not match catalog's capture_rate_gl {Caprate_tot:.2e} within 1% relative tolerance.")

        if self.rebound_results is None or len(self.rebound_results) == 0 or self.load_rebound is False:
            print("Warning: No Rebound results found. Only calculating the capture rate errors without Neq error calculation.")
            neq, neq_se, neq_ci = None, None, None
            total_neq, total_neq_err, neq_ci68, neq_ci95, alpha, neq_qvals = None, None, None, None, None, None
        else:
            neq, neq_se, neq_ci = calc.rate_and_error(Neq_list, sigmas, ses, conf=percentile) 
            total_neq, total_neq_err, neq_ci68, neq_ci95, alpha, neq_qvals = calc.integrated_rate_and_error(v_inf_au_yr, neq, neq_se, n_gl=100, q_vals=[0.025, 0.5, 0.975])
            if not np.isclose(total_neq, Neq_tot, rtol=0.01):
                print(f"Warning: Integrated Neq {total_neq:.2e} does not match catalog's total_systems_neq_gl {Neq_tot:.2e} within 1% relative tolerance.")
            fractional_error_neq = (total_neq_err / total_neq) if total_neq is not None and total_neq > 0 else None
            neq_full = catalog['total_pbh_neq_n']['Neq_pbh_f_full']
            neq_bound = catalog['total_pbh_neq_n']['Neq_pbh_f_bound']
            error_neq_full = fractional_error_neq * neq_full
            error_neq_bound = fractional_error_neq * neq_bound

        return {
            "percentile": percentile,
            "capcrossec_errors": Capcrossec_std_list,
            "total_cap_crossection": cap_crossec,
            "caprate": caprate,
            "caprate_se": cap_se,
            "caprate_ci": cap_ci,
            "total_cap_rate": total_cap_rate,
            "total_cap_err": total_cap_err,
            "total_cap_ci68": ci68,
            "total_cap_ci95": ci95,
            "total_cap_alpha": alpha,
            "total_cap_qvals": qvals,
            "neq": neq,
            "neq_se": neq_se,
            "neq_ci": neq_ci,
            "total_neq": total_neq,
            "total_neq_err": total_neq_err,
            "full_neq_err": error_neq_full,
            "bound_neq_err": error_neq_bound,
            "total_neq_ci68": neq_ci68,
            "total_neq_ci95": neq_ci95,
            "total_neq_alpha": alpha,
            "total_neq_qvals": neq_qvals,
        }
    
    def get_error_ci_summary(self, percentile=0.95, required_n=False):
        catalog = self.results_dictionary
        errors = catalog['errors']
        if errors['percentile'] != percentile:
            print(f"Warning: Catalog errors were calculated for percentile {catalog['errors']['percentile']}, but requested percentile is {percentile}. Recalculating errors with requested percentile.")
            self._get_errors(catalog, percentile=percentile)
        else:
            print(f"Using pre-calculated errors from catalog for percentile {percentile}.")

        # print(f"Error margin summary for percentile {percentile*100}% confidence interval:")
        v_keys = [v for v in catalog.keys() if 'V' in v]
        z_score = norm.ppf(0.5 + percentile / 2)
        # for v in v_keys:

        #     print(f'sigma for v={v}: sigma hat = {errors["capcrossec_errors"][0]}, sigma se = {errors["capcrossec_errors"][1]}, ci = {errors["capcrossec_errors"][2]}')
        
        # print(f'Capture rate: caprate hat = {errors["caprate"]}, caprate se = {errors["caprate_se"]}, caprate ci = {errors["caprate_ci"]}')
        print(f'Integrated capture rate: total_cap_rate hat = {errors["total_cap_rate"]:.2e}, total_cap_err = {errors["total_cap_err"]:.2e}, error marigin = {(errors["total_cap_err"]*z_score):.2e} total_cap_ci95 = {errors["total_cap_ci95"][0]:.2e} to {errors["total_cap_ci95"][1]:.2e}, q values = {errors["total_cap_qvals"]}')
        # print(f'Neq: neq hat = {errors["neq"]}, neq se = {errors["neq_se"]}, neq ci = {errors["neq_ci"]}')
        print(f'Integrated Neq: total_neq hat = {errors["total_neq"]:.2e}, total_neq_err = {errors["total_neq_err"]:.2e}, total_neq_ci95 = {errors["total_neq_ci95"][0]:.2e} to {errors["total_neq_ci95"][1]:.2e}, q values = {errors["total_neq_qvals"]}')

            # std_neq, std_cap = np.std(neq_array), np.std(caprate_array) if neq_array is not None and caprate_array is not None else (None, None)
            # error_marigin_neq = self._get_error_margin(std_neq, len(neq_array), percentile)
            # error_marigin_cap = self._get_error_margin(std_cap, len(caprate_array), percentile)
            # error_marigin_percent_neq = (error_marigin_neq / np.mean(neq_array) * 100) if error_marigin_neq is not None and Neq_tot is not None else None
            # error_marigin_percent_cap = (error_marigin_cap / np.mean(caprate_array) * 100) if error_marigin_cap is not None and Caprate_tot is not None else None
            # print(f"Neq : v={v}: std = {std_neq:.2e}, current error margin relative for 1000 samples ({percentile*100}% CI) = {error_marigin_percent_neq:.2f}%")
            # print(f"Cap : v={v}: std = {std_cap:.2e}, current error margin relative for 1000 samples ({percentile*100}% CI) = {error_marigin_percent_cap:.2f}%")
            # if required_n:
            #     pr = np.mean() * 0.05
            #     required_n = int((z_score * std/ (pr))**2)
            #     # print(f'sigma std for v={v}: {std} sigma sum for system: {np.sum(sigma)} vs capture cross-section: {cap_tot} ')
            #     print(f'Required samples for v={v}: {required_n} \n with std = {std} and pr = {pr}')
        #     Neq_std_list.append(std_neq)
        #     Caprate_std_list.append(std_cap)
        # Neq_std_propogated = np.sqrt(np.sum(np.array(Neq_std_list)**2))
        # Caprate_std_propogated = np.sqrt(np.sum(np.array(Caprate_std_list)**2))
        # Neq_error_marigin_propogated = Neq_std_propogated * z_score / np.sqrt(len(v_keys))
        # Caprate_error_marigin_propogated = Caprate_std_propogated * z_score / np.sqrt(len(v_keys))

        # Neq_error_marigin_propogated_percent = Neq_error_marigin_propogated / Neq_tot * 100 if Neq_error_marigin_propogated is not None and Neq_tot is not None else None
        # Caprate_error_marigin_propogated_percent = Caprate_error_marigin_propogated / Caprate_tot * 100 if Caprate_error_marigin_propogated is not None and Caprate_tot is not None else None

        # print(f"Propagated error margin across all v bins relative for 1000 samples ({percentile*100}% CI) = {Neq_error_marigin_propogated_percent:.2f}% ; metric with ci = {Neq_tot:.2e} +/- {Neq_error_marigin_propogated:.2e} \n")
        # print(f"Propagated error margin across all v bins relative for 1000 samples ({percentile*100}% CI) = {Caprate_error_marigin_propogated_percent:.2f}% ; metric with ci = {Caprate_tot:.2e} +/- {Caprate_error_marigin_propogated:.2e} \n")

    def _get_Neq_at_r_f(self, catalog, ri=None, rf=8, num_points=100):
        """
        Compute total Neq across all v bins for a range of PBH fractions.
        Returns a dict of PBH fraction to total Neq.
        """
        # if pbh_fractions is None:
        #     pbh_fractions = [1e-2, 1e-1, 1.0]
        
        # used_bounds_mode = False
        # if isinstance(pbh_fractions, str) and pbh_fractions == 'bounds':
        bound_ids = ['EGRB', '511keV-DeLaTorreLuque2024', 'CMBevap', 'EDGESevap-Mittal2021',
        'Voyager', 'INTEGRAL', 'LeoTevap', 'SuperK', 'Comptel', 'AMS', 'Xrayevap',
        'LyaEvap-Khan2025', 'OGLE-strict', 'OGLE-highcadence', 'SNe', 'M', 'HSC',
        'EROS', 'Icarus', 'Microlensing-LongDuration', 'Quasars-Xray',
        'FRB-Leung2022', 'Radio', 'CMB', 'EDGES', 'X-ray', 'LeoT', 'WideBinaries',
        'UFdwarfs', 'LIGO', 'LIGO-SGWB-O2', 'LIGO-SGWB-O3', 'LIGO-subsolar',
        'GW-Lensing', 'PBH-EMRIs', '3G-GW-1y', '3G-GW-10y', 'SIGWs', 'OGLE-hint',
        'FL-small', 'GRB-parallax', 'WhiteDwarfmicro', 'Xraylensing-eXTP-proj',
        'GECCO']
        bounds_result = self.get_pbh_fraction_bounds(catalog, bound_id=bound_ids, return_all_bounds=False)
        pbh_fractions = [bounds_result['min_fraction'], 1e0]
        # used_bounds_mode = True
        
        total_Neq = catalog.get('total_occurrences_gl', {}).get('total_systems_neq_gl', None)
        total_Neq_kpc3 = (total_Neq * u.au**3).to(u.kpc**3)  # convert total Neq to number per kpc^3
        range_kpc3, number_density_in_range_kpc3 = self._get_pbh_number_density_r(catalog, ri=ri, rf=rf, num_points=num_points)
        d = {'total_Neq_kpc3': total_Neq_kpc3.value, 'range_kpc3': range_kpc3, 'number_density_in_range_kpc3': number_density_in_range_kpc3.value}
        
        for i, f in enumerate(pbh_fractions):
            pbh_number_density = number_density_in_range_kpc3 * f  # in number/kpc^3
            neq_range = total_Neq_kpc3 * pbh_number_density  # total Neq scaled by PBH number density
            if i == 0:
                d[f'Neq_pbh_f_bound'] = neq_range.value
                d['bound_fraction'] = f
            else:
                d[f'Neq_pbh_f_full'] = neq_range.value
                d['full_fraction'] = f
        return d
    
    def update_Neq_at_r_f(self, pbh_fraction, ri=None, rf=8.0, num_points=100):
        """
        Update catalog with Neq at a specific PBH fraction.
        """
        catalog = self.results_dictionary.copy()
        catalog['total_pbh_neq_n'] = self._get_Neq_at_r_f(catalog, ri=ri, rf=rf, num_points=num_points, pbh_fractions=[pbh_fraction])
        self.results_dictionary = catalog


    def _get_pbh_number_density_r(self, catalog, ri, rf, num_points):
        """
        Compute PBH number density across a range of radii.
        Default DM density is in M_sun/au^3 for number density calculation.
        Returns a dict of radius to PBH number density.
        param ri: inner radius in kpc
        param rf: outer radius in kpc
        param num_points: number of points to sample across the radius range
        """
        if ri is not None:
            radii = np.linspace(ri, rf, num_points)
        else:
            radii = rf
        dm_densities = calc.dm_density_profile_milkyway(radii)  # in M_sun/kpc^3
        mpbh = (catalog['mC'] * u.kg).to(u.M_sun)  # default to 1e-16 M_sun if not specified
        n_pbh = dm_densities / mpbh  # in number/kpc^3
        return radii, n_pbh


    def _get_pbh_number_density_local(self, catalog, pbh_fraction, r=8.0):
        """
        Compute total PBH number densities across all v bins.
        Default DM density is in M_sun/au^3 for number density calculation.
        """
        dm_density_at_r = calc.density_profile_milkyway_nfw(r)  # in M_sun/kpc^3
        dm_density_at_r_au3 = dm_density_at_r.to(u.M_sun / u.au**3)  # convert to M_sun/au^3
        pbh_density = dm_density_at_r * pbh_fraction  # in M_sun/kpc^3
        for o in catalog.get('total_occurrences_gl', {}).keys():
            if o.startswith('total_systems_neq'):
                neq = catalog['total_occurrences_gl'][o]
                mpbh = (catalog['mC'] * u.kg).to(u.M_sun)  # default to 1e-16 M_sun if not specified
                n_pbh = pbh_density  / mpbh  # in number/au^3
                n_captured_pbh = neq * n_pbh  # total number of captured PBHs
                catalog['total_occurrences_gl'][f'Neq_pbh_local'] = n_captured_pbh

    def get_pbh_fraction_bounds(self, catalog, bound_id, return_all_bounds=False):
        """
        Get the PBH fraction bounds for the mC mass from PlotPBHbounds module.
        
        Parameters
        ----------
        catalog : dict
            The analysis catalog containing 'mC' (mass in kg)
        bound_id : str or list
            Single bound ID name(s) to query (e.g., 'LIGO', 'Microlensing', etc.)
            See PBHbounds/bounds/ for available bound files.
        return_all_bounds : bool
            If True and bound_id is a list, return all bounds. If False, return minimum.
            
        Returns
        -------
        dict : Contains 'mC_Msun' and bound values 'bound_{bound_id}' for each constraint
        """
        try:
            # Dynamically import tools from PBHbounds
            import sys
            pbhbounds_path = os.path.join(REPO_ROOT, 'PBHbounds')
            if pbhbounds_path not in sys.path:
                sys.path.insert(0, pbhbounds_path)
            import tools
        except ImportError as e:
            raise ImportError(f"Could not import tools from PBHbounds: {e}")
        
        # Convert mC from kg to solar masses
        mC_kg = catalog.get('mC')
        if mC_kg is None:
            raise ValueError("catalog must contain 'mC' key with mass in kg")
        
        mC_Msun = (mC_kg * u.kg).to(u.M_sun).value
        
        # Ensure bound_id is a list
        if isinstance(bound_id, str):
            bound_ids = [bound_id]
        else:
            bound_ids = list(bound_id)
        
        result = {'mC_kg': mC_kg, 'mC_Msun': mC_Msun}
        bounds_values = []
        
        for bid in bound_ids:
            try:
                # Load bound data from PlotPBHbounds
                m_arr, f_arr = tools.load_bound(bid)
                
                # Interpolate to find f at mC_Msun
                # Handle case where mC is outside the bounds
                if mC_Msun < m_arr.min() or mC_Msun > m_arr.max():
                    f_at_mC = np.nan  # Out of range
                    status = "out_of_range"
                else:
                    # Use log-log interpolation for better accuracy
                    f_at_mC = np.interp(np.log10(mC_Msun), np.log10(m_arr), np.log10(f_arr), left=np.nan, right=np.nan)
                    f_at_mC = 10**f_at_mC if not np.isnan(f_at_mC) else np.nan
                    status = "valid"
                
                result[f'bound_{bid}'] = f_at_mC
                result[f'status_{bid}'] = status
                bounds_values.append(f_at_mC)
                
            except Exception as e:
                # print(f"Warning: Could not load bound '{bid}': {e}")
                result[f'bound_{bid}'] = np.nan
                result[f'status_{bid}'] = "error"
        
        # If multiple bounds and not returning all, compute summary statistics
        if len(bounds_values) > 1 and not return_all_bounds:
            valid_bounds = [b for b in bounds_values if not np.isnan(b)]
            if valid_bounds:
                result['min_fraction'] = np.min(valid_bounds)
                result['max_fraction'] = np.max(valid_bounds)
                result['mean_fraction'] = np.mean(valid_bounds)
        
        return result

    def _get_time_series(self, mc_npz, rebound_list, sampled_mc, masks):
        """
        Compute averaged time series for a single v bin.
        Pure function: takes inputs, returns time series dict.
        """

        semi_major_axis_averages = [float(np.mean(m['semi_major_axes'])) for m in rebound_list if np.size(m['semi_major_axes']) > 0]
        eccentricity_averages = [float(np.mean(m['eccentricities'])) for m in rebound_list if np.size(m['eccentricities']) > 0]
        orbital_period_averages = [float(np.mean(m['orbital_periods'])) for m in rebound_list if np.size(m['orbital_periods']) > 0]
        lifetimes = [float(m['lifetime']) for m in rebound_list if 'lifetime' in m.files]

        return {
            "semi_major_axis_averages": semi_major_axis_averages,
            "eccentricity_averages": eccentricity_averages,
            "orbital_period_averages": orbital_period_averages,
            "lifetimes": lifetimes,
        }
    
    def _total_termination_counts(self, catalog):
        """
        Compute total termination counts across all v bins.
        """
        total_counts = {}
        for v_key, entry in catalog.items():
            if not isinstance(v_key, str) or not v_key.startswith('V'):
                continue
            term_counts = entry.get('termination_counts', {})
            for flag, count in term_counts.items():
                total_counts[flag] = total_counts.get(flag, 0) + count

        return total_counts

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
            if 'V' in k:
                mc_ok = "yes" if catalog[k]["mc"] else "no"
                n_rb = len(catalog[k]["rebound"])
                print(f"  {k}: MC={mc_ok}, Rebound files={n_rb}")


    
    def print_detailed_catalog_summary(self):

        catalog = self.results_dictionary
        v_keys = [k for k in catalog.keys() if isinstance(k, str) and k.startswith('V')]

        for k in v_keys:
            total_capture_count = catalog[k].get('n_captured', 0)
            total_sample_count = catalog[k].get('n_sampled', 0)
            try:
                cross_section = catalog[k].get('sigma_MC_dsigma_m2')
            except KeyError:
                cross_section = catalog[k].get('sigma_MC_dsigma_au2')
            sampled_capture_count = catalog[k].get('sampled_capture_count', 0)
            checks = catalog[k].get('checks', {})
            print(
                f'Velocity bin: {k}, sampled capture count {sampled_capture_count}'
                f'total capture count: {total_capture_count} out of {total_sample_count} samples'
                f',sigma_MC_dsigma_au2: {cross_section if cross_section is not None else "N/A"}'
                f', satisfied conditions- bmax>bmin, b>bmin, E_sys<0, E_cap<0: {checks[2:-1]}' #sampled, failed, check1, check2, check3, check4, n_captured
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
    
    def get_vkeys(self):
        catalog = self.results_dictionary
        return [k for k in catalog.keys() if isinstance(k, str) and k.startswith('V')]

    def print_occurrences_summary(self):

        catalog = self.results_dictionary
        print("Occurrences summary:")
        for k in self.get_vkeys():
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

    def print_termination_counts_summary(self, mode="per_v"):

        catalog = self.results_dictionary
        if mode == "total":
            if 'total_termination_counts' in catalog:
                print("Total Termination Counts summary:")
                for flag, count in catalog['total_termination_counts'].items():
                    print(f"    {flag}: {count}")
                print("--------------------------------")
            else:
                print("No total termination counts found in catalog.")
        print("Termination counts summary:")
        for k in [k for k in catalog.keys() if isinstance(k, str) and k.startswith('V')]:
            if 'termination_counts' in catalog[k]:
                print(f'Velocity bin: {k}')
                for flag, count in catalog[k]['termination_counts'].items():
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

    def build_mask(self, v_key, predicate, entry_type='Rebound', as_int=False):
        """
        Build a mask over catalog[v_key]['rebound'] given a predicate(entry) -> bool.
        - v_key: e.g., 'v17'
        - predicate: callable that receives an npz entry and returns True/False
        - as_int: return 1/0 instead of True/False
        """
        if entry_type != 'Rebound':
            entries = self.results_dictionary.get(v_key, {})
            print(f"Building mask for entry_type='{entry_type}' with {len(entries)} entries from catalog[{v_key}][{entry_type}]")
        else:
            entries = self._rebound_entries(v_key)
        out = []
        for e in entries:
            try:
                out.append(bool(predicate(e, entry_type)))
            except Exception:
                out.append(False)
        arr = np.asarray(out, dtype=bool)
        return arr.astype(int) if as_int else arr

    def mask_flag_contains(self, v_key, substr, mask_type="include", as_int=False):
        """Mask entries whose termination_flag contains substr."""
        def _pred(e):
            if "termination_flag" not in e.files:
                return False
            flag = self._as_scalar(e["termination_flag"])
            if mask_type == "include":
                return isinstance(flag, str) and (substr in flag)      # ✅ Include entries with substr
            else:  # mask_type == "exclude"
                return isinstance(flag, str) and (substr not in flag)  # ✅ Include entries without substr
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

    def mask_where(self, v_key, key, predicate, entry_type, as_int=False):
        """
        Generic field-based mask: predicate receives the field value.
        Example: analysis.mask_where('v17','lifetime', lambda t: np.isfinite(t) and t>1e6)
        """
        def _pred(e, entry_type):
            if entry_type == 'Rebound':
                if key not in e.files:
                    return False
                val = self._as_scalar(e[key])
                return bool(predicate(val))
            else:
                # For non-Rebound entries, look in the catalog's v_key dict directly
                catalog_entry = self.results_dictionary.get(v_key, {})
                if key not in catalog_entry:
                    print(f"Warning: key '{key}' not found in catalog entry for {v_key} when applying mask_where with entry_type='{entry_type}'.")
                    return False
                val = self._as_scalar(catalog_entry[key])
                print(f"Applying mask_where with entry_type='{entry_type}': key='{key}', value={val}")
                return bool(predicate(val))
        return self.build_mask(v_key, _pred, entry_type=entry_type, as_int=as_int)

    @staticmethod
    def _npz_to_dict(npz):
        """
        Convert np.load(...)-style NpzFile or dict-like object into a plain dict.
        Arrays are copied so we are not holding views into closed npz files.
        """
        if npz is None:
            return {}

        if hasattr(npz, "files"):
            return {k: np.array(npz[k], copy=True) for k in npz.files}

        if isinstance(npz, dict):
            return {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else copy.deepcopy(v)
                    for k, v in npz.items()}

        raise TypeError(f"Unsupported MC data type: {type(npz)}")


    @staticmethod
    def _merge_mc_dicts(d1, d2):
        import copy
        import numpy as np

        out = {}

        skip_keys = {
            "execution_time",
            "random_seed",
        }

        count_keys = {
            "n_captured",
            "sample_number",
        }

        derived_scalar_keys = {
            "sigma_MC_m2",
            "sigma_MC_au2",
            "sigma_MC_dsigma_m2",
            "sigma_MC_dsigma_au2",
            "sigma_MC_dsigma_captured_m2",
            "sigma_MC_dsigma_captured_au2",
        }

        keys = set(d1.keys()) | set(d2.keys())

        for key in keys:
            if key in skip_keys:
                # Skip metadata that should not be merged.
                continue
            if key in derived_scalar_keys:
                # Recompute later from raw samples.
                continue

            if key not in d1:
                out[key] = copy.deepcopy(d2[key])
                continue

            if key not in d2:
                out[key] = copy.deepcopy(d1[key])
                continue

            a = d1[key]
            b = d2[key]

            if key in count_keys:
                out[key] = np.asarray(a).item() + np.asarray(b).item()

            elif key == "checks":
                out[key] = np.asarray(a) + np.asarray(b)

            elif key.startswith("cap"):
                out[key] = np.concatenate([np.atleast_1d(a), np.atleast_1d(b)])

            elif key == "idx":
                continue

            elif key in {"v_inf", "epsilon"}:
                # Metadata that should match; keep one copy.
                if not np.allclose(a, b, equal_nan=True):
                    raise ValueError(f"Metadata mismatch for key={key}: {a} vs {b}")
                out[key] = copy.deepcopy(a)

            else:
                # Conservative fallback:
                # if arrays look event-level and compatible, concatenate;
                # otherwise require equality.
                if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.ndim > 0:
                    if a.shape[1:] == b.shape[1:]:
                        out[key] = np.concatenate([a, b])
                    else:
                        raise ValueError(f"Shape mismatch for key={key}: {a.shape} vs {b.shape}")
                else:
                    try:
                        if np.allclose(a, b, equal_nan=True):
                            out[key] = copy.deepcopy(a)
                        else:
                            raise ValueError(f"Metadata mismatch for key={key}: {a} vs {b}")
                    except TypeError:
                        if a == b:
                            out[key] = copy.deepcopy(a)
                        else:
                            raise ValueError(f"Metadata mismatch for key={key}: {a} vs {b}")

        if "n_captured" in out:
            out["idx"] = np.arange(int(out["n_captured"]))

        return out
    def _npz_to_plain_dict(data):
        """
        Convert npz or dict-like MC data into a plain dictionary.
        This avoids keeping npz file handles in the merged object.
        """
        if hasattr(data, "files"):
            return {k: np.array(data[k], copy=True) for k in data.files}

        if isinstance(data, dict):
            out = {}
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    out[k] = np.array(v, copy=True)
                else:
                    out[k] = copy.copy(v)
            return out

        raise TypeError(f"Unsupported data type: {type(data)}")


    def merged_copy_for_error_analysis(self, other):
        """
        Return a new Analysis-like object containing pooled MC and rebound data
        from self and other, without using deepcopy on npz file handles.
        """
        import copy

        combined = copy.copy(self)
        combined.name = self.name + "_plus_" + other.name

        # Do NOT clear combined.mc_results here.
        # merge_with_for_error_analysis needs self.mc_results from the copied object.

        combined.merge_with_for_error_analysis(other)

        return combined

    def merge_with_for_error_analysis(self, other):
        """
        Merge another Analysis object into this one at the raw-data level,
        then rebuild sampled_mc_results and results_dictionary.

        This is the method to use for M=1 + M=2.
        """
        if self.system_param_dict.get("mC") != other.system_param_dict.get("mC"):
            raise ValueError("Can only merge analyses with the same mC.")

        # Optional but recommended: require same velocity grid.
        v_self = set(self.mc_results.keys())
        v_other = set(other.mc_results.keys())
        if v_self != v_other:
            raise ValueError(f"Velocity grids differ: {v_self ^ v_other}")

        merged_mc_results = {}

        for v in sorted(v_self, key=lambda x: int(x[1:])):
            d1 = Analysis._npz_to_dict(self.mc_results[v])
            d2 = Analysis._npz_to_dict(other.mc_results[v])

            merged_mc_results[v] = Analysis._merge_mc_dicts(d1, d2)

        # Merge rebound lists.
        merged_rebound_results = {}
        for v in sorted(v_self, key=lambda x: int(x[1:])):
            rb1 = self.rebound_results.get(v, []) or []
            rb2 = other.rebound_results.get(v, []) or []
            merged_rebound_results[v] = list(rb1) + list(rb2)

        self.mc_results = merged_mc_results
        self.rebound_results = merged_rebound_results

        # Very important: do not downsample back to only 1000 captures.
        # Use all merged captured systems.
        self.mc_sample_size = 10**18

        self.sampled_mc_results = self._get_sampled_mc_results()
        self.results_dictionary = self.get_combined_dictionary(use_cached_data=False)

        return self
    def _get_metric_from_sources(self, analysis_dict, v_key, metric_name):
        """Search for metric in multiple sources."""
        sp = analysis_dict

        if sp is not None and metric_name in sp:
            value = sp[metric_name]
            print(f'✓ Found in analysis_dict')
            return value
        an_entry = sp[v_key]
        mc = an_entry.get('mc')
        rb = an_entry.get('rebound', [])
        oc = an_entry.get('occurrences')
        tc = an_entry.get('termination_counts')
        smc = an_entry.get('sampled_mc')

        # 0) Direct entry check
        if metric_name in an_entry:
            value = an_entry[metric_name]
            print(f'✓ Found in an_entry')
            return value

        # 1) Rebound entries (returns list of arrays)
        if rb:
            try:
                metric = [rb_entry[metric_name] for rb_entry in rb 
                        if hasattr(rb_entry, 'files') and metric_name in rb_entry.files]
                if metric:
                    print(f'✓ Found in rebound: {len(metric)} entries')
                    return metric
            except Exception as e:
                print(f'✗ Rebound error: {e}')

        # 2) MC npz file (returns single array)
        if mc is not None and hasattr(mc, 'files') and metric_name in mc.files:
            # ✅ FIX: Make a COPY of the array from the npz file
            value = np.array(mc[metric_name])  # Force copy, not a view
            print(f'✓ Found in mc: shape={value.shape}')
            return value

        # 3) Occurrences
        if oc is not None and metric_name in oc:
            value = oc[metric_name]
            print(f'✓ Found in occurrences')
            return value
        
        # 4) Termination counts
        if tc is not None and metric_name in tc:
            value = tc[metric_name]
            print(f'✓ Found in termination_counts')
            return value
        # 5) Sampled MC
        if smc is not None and metric_name in smc:
            value = smc[metric_name]
            print(f'✓ Found in sampled_mc')
            return value
        


        print(f'✗ Metric "{metric_name}" not found')
        return np.array([])  # Return empty array instead of []







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