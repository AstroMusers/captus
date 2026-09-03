import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u
from src.ThreeBodyCapture.Simulations.rebound import OrbitalSimulation
from numpy.random import SeedSequence, Generator, PCG64
import src.Utils.calculations as calc
import multiprocessing
import datetime
from numpy.random import SeedSequence, Generator, PCG64
import src.ThreeBodyCapture.Simulations.montecarlo as MCi
import src.ThreeBodyCapture.Configurations.configuration as conf
import src.ThreeBodyCapture.Analysis.analysis as anl
import src.ThreeBodyCapture.Plotting.plotting as pl
import src.Utils.misc as misc 
import src.Utils.plotting_utils as pu
import datetime as dt
import os
import pandas as pd
from astropy.cosmology import FlatLambdaCDM

class PBHPopulation:
    def __init__(
            self,
            system_name,
            mPBH_min,
            mPBH_max,
            mPBH_num,
            n_per_mPBH,
            seed_start,
            identifier,
            limit_max_v=False,
            system_param_overrides=None,
            simulation_param_overrides=None,
            ):

        self.system_name = system_name
        self.mPBHs = np.logspace(np.log10(mPBH_min), np.log10(mPBH_max), mPBH_num)
        self.system_param_overrides = system_param_overrides or {}
        self.simulation_param_overrides = simulation_param_overrides or {}
        self.population_dict = self._preprocess_population(
            n_per_mPBH,
            seed_start,
            identifier,
            system_param_overrides=self.system_param_overrides,
            simulation_param_overrides=self.simulation_param_overrides,
            limit_max_v=limit_max_v,
        )

    def _preprocess_population(
            self,
            n_per_mPBH,
            seed_start=0,
            identifier=None,
            system_param_overrides=None,
            simulation_param_overrides=None,
            limit_max_v=False,):
        """
        Preprocess the population of PBHs for the simulation.

        Parameters:
        - n_per_mPBH: Number of PBHs per mass value.
        - mPBHs: Array of PBH masses.
        - seed_start: Starting seed for random number generation.
        - identifier: Optional identifier for the run.
        - system_param_overrides: Dictionary of system parameter overrides.
        - simulation_param_overrides: Dictionary of simulation parameter overrides.
        - limit_max_v: Boolean indicating whether to limit the maximum velocity.

        Returns:
        - seeds: Array of seeds for each PBH.
        - mPBH_array: Array of PBH masses corresponding to each seed.
        """
        system_param_overrides = system_param_overrides or {}
        simulation_param_overrides = simulation_param_overrides or {}

        pop_dict = {}
        for i, mPBH in enumerate(self.mPBHs):
            seed = i + seed_start  # Just a way to get different seeds
            name = f"{self.system_name}_s{seed}_Mpbh{mPBH/const.M_sun.value:.0e}{f'_{identifier}' if identifier else ''}"
            configuration = conf.Configuration(name=name, seed=seed, importance_sampling=True)
            configuration.set_system_param('mC', mPBH)  # Mass of light PBH
            configuration.set_simulation_param('sample_size', n_per_mPBH)  # Mass of light PBH

            if limit_max_v:
                max_v_default = configuration.get_simulation_param(all=True)['v_inf_grid'][-1]
                configuration.set_simulation_param('max_v', max_v_default*0.75) # From Dehnen 2022, for circular Jupiter 31.6 km/s
            # Extra parameter overrides if provided
            
            for key, value in system_param_overrides.items():
                if value is not None:
                    configuration.set_system_param(key, value)

            for key, value in simulation_param_overrides.items():
                if value is not None:
                    configuration.set_simulation_param(key, value)


            configuration.print_configuration()
            pop_dict[name] = {'name': name, 'seed': seed, 'configuration': configuration}

            print(f"Prepared configuration for mPBH = {mPBH / const.M_sun.value} solar masses.\n")

        return pop_dict