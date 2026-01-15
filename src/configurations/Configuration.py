import yaml
import numpy as np
import src.utils.calculations as calc
import astropy.constants as const
from astropy import units as u
import src.utils.misc as misc
import os, sys

REPO_ROOT = misc._resolve_repo_root()

class Configuration:
    def __init__(self, name, seed, importance_sampling=False, config_file=None):
        # Load configuration from file
        self.name = name
        self.seed = seed
        self.config_file = os.path.join(REPO_ROOT, config_file) if config_file else None
        self.importance_sampling = importance_sampling
        self._set_params()


    def _set_params(self):
        # Set system parameters based on the configuration file
        sys_params = None
        sim_params = None
        if self.config_file:
            cfg = yaml.safe_load(open(self.config_file))
        else:
            if 'Sun' in self.name:
                cfg = yaml.safe_load(open(os.path.join(REPO_ROOT, 'src/configurations/SunJupiter.yaml')))
                if 'lightPBH' in self.name:
                    sys_params = cfg['system_param_dict_lightPBH']
                if 'massivePBH' in self.name:
                    sys_params = cfg['system_param_dict_massivePBH']

            if 'Cygnus' in self.name:
                cfg = yaml.safe_load(open(os.path.join(REPO_ROOT, 'src/configurations/CygnusX1.yaml')))

            if 'Intermediate' in self.name:
                cfg = yaml.safe_load(open(os.path.join(REPO_ROOT, 'src/configurations/IntermediateBHStar.yaml')))

        if sys_params is None:
            sys_params = cfg['system_param_dict']

        sys_params['name'] = self.name
        sys_params['seed_base'] = self.seed
        self.system_param_dict = sys_params

        sim_params = cfg['simulation_param_dict']
        sim_params['v_inf_grid_params'] = cfg['v_inf_grid_params']
        sim_params['importance_sampling'] = self.importance_sampling
        sim_params['v_inf_grid'] = self._create_vinf_grid(cfg['v_inf_grid_params'])
        self.simulation_param_dict = sim_params

    def set_system_param(self, key, value):
        
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
        eB = self.system_param_dict['eB']
        vBMag = self.system_param_dict['vB']
        epsilon = self.system_param_dict['epsilon']
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
        name = self.name
        dir_mc = mc_dir if mc_dir else f'runs/{name}/Monte_Carlo_Results'
        save_dir_mc = os.path.join(REPO_ROOT, dir_mc)
        os.makedirs(save_dir_mc, exist_ok=True)
        self.save_dir_mc = save_dir_mc

        dir_rebound = rebound_dir if rebound_dir else f'runs/{name}/Rebound_Simulation_Results'
        save_dir_rebound = os.path.join(REPO_ROOT, dir_rebound)
        os.makedirs(save_dir_rebound, exist_ok=True)
        self.save_dir_rebound = save_dir_rebound

        dir_plots = plots_dir if plots_dir else f'plots/{name}'
        save_dir_plots = os.path.join(REPO_ROOT, dir_plots)
        os.makedirs(save_dir_plots, exist_ok=True)
        self.save_dir_plots = save_dir_plots

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