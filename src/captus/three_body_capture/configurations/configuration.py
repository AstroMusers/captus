import os
from importlib.resources import files
from pathlib import Path

import numpy as np
import astropy.constants as const
from astropy import units as u
import yaml

import captus.utils.calculations as calc


_SYSTEM_CONFIG_FILES = {
    "Sun": "SunJupiter.yaml",
    "Cygnus": "CygnusX1.yaml",
    "Intermediate": "IntermediateBHStar.yaml",
    "OmegaCenturi": "OmegaCenturi.yaml",
    "BetaPictoris": "BetaPictoris.yaml",
}


class Configuration:
    def __init__(self, name, seed, importance_sampling=False, config_file=None):
        # Load configuration from file
        self.name = name
        self.seed = seed
        self.config_file = config_file
        self.importance_sampling = importance_sampling
        self._set_params()


    def _set_params(self):
        # Set system parameters based on the configuration file
        cfg = None
        sys_params = None
        if self.config_file:
            with Path(self.config_file).open("r") as config_file:
                cfg = yaml.safe_load(config_file)
        else:
            for system_name, filename in _SYSTEM_CONFIG_FILES.items():
                if system_name in self.name:
                    config_path = (
                        files("captus.three_body_capture.configurations") / filename
                    )
                    with config_path.open("r") as config_file:
                        cfg = yaml.safe_load(config_file)
                    if system_name == "Sun":
                        if "lightPBH" in self.name:
                            sys_params = cfg["system_param_dict_lightPBH"]
                        if "massivePBH" in self.name:
                            sys_params = cfg["system_param_dict_massivePBH"]

        if cfg is None:
            raise ValueError(
                f"No built-in configuration matches {self.name!r}; supply config_file."
            )

        if sys_params is None:
            sys_params = cfg['system_param_dict']

        sys_params['name'] = self.name
        sys_params['seed_base'] = self.seed
        sys_params['rC'] = calc.schwarzchild_radius(sys_params['mC'])
        self.system_param_dict = sys_params

        sim_params = cfg['simulation_param_dict']
        sim_params['v_inf_grid_params'] = cfg['v_inf_grid_params']
        sim_params['importance_sampling'] = self.importance_sampling
        sim_params['v_inf_grid'] = self._create_vinf_grid(cfg['v_inf_grid_params'])
        self.simulation_param_dict = sim_params

    def set_system_param(self, key, value):
        if key == 'mC':
            self.system_param_dict[key] = value
            self.system_param_dict['rC'] = calc.schwarzchild_radius(value)
            print(f"Updated system parameter '{key}' to {value} and recalculated rC.")
        else:
            self.system_param_dict[key] = value

    def set_simulation_param(self, key, value):
        if key in self.simulation_param_dict['v_inf_grid_params'].keys():
            self.simulation_param_dict['v_inf_grid_params'][key] = value
            self.simulation_param_dict['v_inf_grid'] = self._create_vinf_grid(self.simulation_param_dict['v_inf_grid_params'])
            print(f"Updated v_inf_grid_params '{key}' to {value} and regenerated v_inf_grid.")

        else:
            self.simulation_param_dict[key] = value

    def get_system_param(self, key=None, all=False):
        if all:
            return self.system_param_dict
        if key is None:
            raise ValueError("Key must be provided if 'all' is False.")
        return self.system_param_dict.get(key, None)

    def get_simulation_param(self, key=None, all=False):
        if all:
            return self.simulation_param_dict
        if key is None:
            raise ValueError("Key must be provided if 'all' is False.")
        return self.simulation_param_dict.get(key, None)

    def print_configuration(self):
        print("System Parameters:")
        for key, value in self.system_param_dict.items():
            print(f"  {key}: {value}")
        print("\nSimulation Parameters:")
        for key, value in self.simulation_param_dict.items():
            print(f"  {key}: {value}")

    def print_preview(self):
        print("Configuration Preview:")
        print(f"  Name: {self.name}")
        print(f"  Seed: {self.seed}")
        mA = self.system_param_dict['mA']
        mB = self.system_param_dict['mB']
        mC = self.system_param_dict['mC']
        aB = self.system_param_dict['aB']
        rB = self.system_param_dict['rB']
        rA = self.system_param_dict['rA']
        rC = self.system_param_dict['rC']
        eB = self.system_param_dict['eB']
        vBMag = self.system_param_dict['vB']
        epsilon = self.system_param_dict['epsilon']
        epsilon_adjust_coeff = self.system_param_dict['epsilon_adjust_coeff']
        G = const.G.value
        muA = mA * G
        muB = mB * G
        T = calc.orbital_period(aB, mA, mB)
        r_hill = calc.hill_radius(aB, mA, mB, eB)
        vesc = calc.v_esc(muA, aB)
        rclose = calc.r_close(epsilon, mA, mB, aB)
        vmax = vesc + 2*vBMag
        vinf_max = np.sqrt(vmax**2 - (2 * muA / aB) - (2 * muB / rclose))

        print(f'Orbital velocity of B: {vBMag} m/s')
        print(f'Escape velocity v_esc: {vesc} m/s')
        print(f'Maximum incoming velocity v_max: {vinf_max} m/s')
        print(f'Orbital period of B: {T/(3600*24)} days')
        print(f'Semimajor axis of B: {aB*u.m.to(u.au)} au')
        print(f'Radius of B: {rB*u.m.to(u.au)} au')
        print(f'rClose: {rclose*u.m.to(u.au)} au')
        print(f'rHill: {r_hill*u.m.to(u.au)} au')
        print(f'rClose/rHill: {rclose/r_hill}')
        print(f'Radius of C: {calc.schwarzchild_radius(mC)*u.m.to(u.au)} au')

    def _create_vinf_grid(self, vinf_params):
        v_inf_min = vinf_params['min_v']  # default 1 km/s
        v_inf_max = vinf_params['max_v']  # default 300 km/s
        samples = vinf_params['n_v']  # default 10
        v_inf_grid = np.linspace(v_inf_min, v_inf_max, samples)
        return v_inf_grid

    
    def _set_save_dir(self, mc_dir=None, rebound_dir=None, plots_dir=None):
        directories = (
            ("save_dir_mc", mc_dir or f"runs/{self.name}/Monte_Carlo_Results"),
            ("save_dir_rebound", rebound_dir or f"runs/{self.name}/Rebound_Simulation_Results"),
            ("save_dir_plots", plots_dir or f"plots/{self.name}"),
        )
        for attribute, directory in directories:
            path = os.path.join(Path.cwd(), "..", directory)
            os.makedirs(path, exist_ok=True)
            setattr(self, attribute, path)

    def get_save_dir_mc(self):
        if getattr(self, 'save_dir_mc', None) is None:
            self._set_save_dir()
        return self.save_dir_mc

    def get_save_dir_rebound(self):
        if getattr(self, 'save_dir_rebound', None) is None:
            self._set_save_dir()
        return self.save_dir_rebound
    
    def get_save_dir_plots(self):
        if getattr(self, 'save_dir_plots', None) is None:
            self._set_save_dir()
        return self.save_dir_plots
    
    def set_save_dir(self, mc_dir=None, rebound_dir=None, plots_dir=None):
        self._set_save_dir(mc_dir, rebound_dir, plots_dir)