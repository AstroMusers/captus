import yaml
import numpy as np
import src.utils.calculations as calc
import astropy.constants as const
from astropy import units as u
import os, sys

class Configuration:
    def __init__(self, name, seed, config_file=None):
        # Load configuration from file
        self.name = name
        self.seed = seed
        self.config_file = os.path.join(os.getcwd(), config_file) if config_file else None
        self._set_system_params()
        self._set_simulation_params()


    def _set_system_params(self):
        # Set system parameters based on the configuration file
        params = None
        if self.config_file:
            cfg = yaml.safe_load(open(self.config_file))
        else:
            if 'Sun' in self.name:
                cfg = yaml.safe_load(open(os.path.join(os.getcwd(), '../src/configurations/SunJupiter.yaml')))
                if 'lightPBH' in self.name:
                    params = cfg['system_param_dict_lightPBH']
                if 'massivePBH' in self.name:
                    params = cfg['system_param_dict_massivePBH']

            if 'Cygnus' in self.name:
                cfg = yaml.safe_load(open(os.path.join(os.getcwd(), '../src/configurations/CygnusX1.yaml')))
            if 'Intermediate' in self.name:
                cfg = yaml.safe_load(open(os.path.join(os.getcwd(), '../src/configurations/IntermediateBHStar.yaml')))

        if params is None:
            params = cfg['system_param_dict']

        self.system_param_dict = params
        self.system_param_dict['name'] = self.name
        self.system_param_dict['seed_base'] = self.seed
        self.vinf_params = cfg['v_inf_grid_params']  # default 300 km/s

    def set_system_param(self, key, value):
        self.system_param_dict[key] = value

    def _set_simulation_params(self):
        # Update system parameters with additional parameters
        self.simulation_param_dict = {}
        self.simulation_param_dict['trials'] = 10_000_000
        self.simulation_param_dict['sample_size'] = 200
        self.simulation_param_dict['e_lim'] = 0.95
        self.simulation_param_dict['v_inf_grid'] = self._create_vinf_grid()

    def set_simulation_param(self, key, value):
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

    def _create_vinf_grid(self):
        v_inf_min = self.vinf_params['min']  # default 1 km/s
        v_inf_max = self.vinf_params['max']  # default 300 km/s
        samples = self.vinf_params['n']  # default 10
        v_inf_grid = np.linspace(v_inf_min, v_inf_max, samples)
        return v_inf_grid