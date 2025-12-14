import os
import astropy.constants as const
import src.utils.calculations as calcs
import numpy as np
from astropy import units as u
import multiprocessing

class Simulation:

    def __init__(self, system_param_dict):
        # Initialization code here
        self.sys_par = system_param_dict
        self._set_system()

    def _set_system(self):

        G = const.G.value  # Gravitational constant in m^3 kg^-1 s^-2

        self.sys_par['muA'] = self.sys_par['mA'] * G
        self.sys_par['muB'] = self.sys_par['mB'] * G
        self.sys_par['muC'] = self.sys_par['mC'] * G

        self.sys_par['rC'] = calcs.schwarzchild_radius(self.sys_par['mC'])

        self.sys_par['areaB'] = np.pi * self.sys_par['rB']**2

        self.sys_par['vEscape'] = calcs.v_esc(self.sys_par['muA'], self.sys_par['rAB'])

        self.sys_par['potentialEnergy'] = calcs.potential_energy(self.sys_par['muA'], self.sys_par['rAB'], self.sys_par['muB'], self.sys_par['rClose'])

        self.sys_par['rABVec'] = np.array([0.0, self.sys_par['rAB'], 0.0])

    def _save_mc_results(self, v, N):
        # Code to save Monte Carlo results
        name = self.sys_par['name']
        seed = self.sys_par['seed']
        save_dir = f'../runs/{name}/Monte_Carlo_Results/N{N:.e0}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        results = self.mc_results
        np.savez(f'{save_dir}/monte_carlo_results_v{v/1e3:.1f}_s{seed}.npz',
                 v_inf=results['v_inf'],
                 n_captured=results['n_captured'],
                 sigma_MC_m2=results['sigma_MC_m2'],
                 sigma_MC_areaB=results['sigma_MC_areaB'],
                 a_au=results['a_au'],
                 e=results['e'],
                 cap_lambda=results['cap_lambda'],
                 cap_beta=results['cap_beta'],
                 cap_b=results['cap_b'],
                 cap_phi=results['cap_phi'])

    def run_monte_carlo_simulation(self, trials, sample_size, v_inf, lambda_samples, cosbeta_samples, phi_samples, b_samples):
        # def run_monte_carlo_simulation(self, seed, trials, v_inf, lambda_limits, cosbeta_limits, phi_limits):
        # rng = np.random.default_rng(seed)
        N = trials
        # # 1) sample direction of incoming PBH (λ, β)
        # lambda_samples = rng.uniform(lambda_limits[0], lambda_limits[1], size=N)
        # cosbeta_samples  = rng.uniform(cosbeta_limits[0], cosbeta_limits[1], size=N)    # cosβ
        beta_samples = np.arccos(cosbeta_samples)

        # # 2) sample impact parameter b with p(b) ∝ b
        # b_max = self.sys_par['rClose']                      # set maximum impact parameter (e.g. rclose)
        # u_b = rng.uniform(0.0, 1.0, size=N)
        # b_samples = b_max * np.sqrt(u_b)

        # # 3) sample scattering-plane angle φ
        # phi_samples = rng.uniform(phi_limits[0], phi_limits[1], size=N)

        # Shared constants
        muA = self.sys_par['muA']
        muB = self.sys_par['muB']
        rClose = self.sys_par['rClose']
        vBMag = self.sys_par['vB']
        rABVec = self.sys_par['rABVec']
        potentialEnergy = self.sys_par['potentialEnergy']
        b_max = rClose
        max_samples = sample_size
        n_captured = 0
        out_lambda, out_beta, out_b, out_phi = [], [], [], []
        out_a, out_e = [], []

        for lam, beta, b, phi in zip(lambda_samples, beta_samples, b_samples, phi_samples):
            v1Mag = calcs.v_1_mag(v_inf, muA, muB, self.sys_par['rAB'], rClose)
            v1Vec = calcs.v_1_vec(v1Mag, lam, beta)
            vBVec = calcs.v_B_vec(vBMag, lam)

            v1primeVec = calcs.v_1_prime_vec(v1Vec, vBVec, lam, beta)
            v1primeMag = np.linalg.norm(v1primeVec)

            if v1primeMag == 0:
                continue

            if b < self.sys_par['rB'] + self.sys_par['rC']:
                continue

            v2Vec = calcs.v_2_vec(v1primeVec, v1primeMag, vBVec, muB, b, phi)
            v2Mag = calcs.v_2_mag(v2Vec)

            E2_val = 0.5 * v2Mag**2 + potentialEnergy

            if E2_val + (muB / rClose) < 0:
                L2_val = calcs.L2(rABVec, v2Vec)
                a_val, e_val = calcs.a_e(muA, E2_val, L2_val)

                if a_val > 0 and 0 <= e_val < 1 and n_captured < max_samples:
                    out_a.append(a_val)
                    out_e.append(e_val)
                    out_lambda.append(lam)
                    out_beta.append(beta)
                    out_b.append(b)
                    out_phi.append(phi)
                    n_captured += 1

        # estimate capture cross-section
        sigma_MC = (n_captured / N) * float(np.pi) * b_max**2

        print(f"for v_inf={v_inf/1e3} km/s MC capture cross-section:", sigma_MC, "m^2")
        if 'areaB' in self.sys_par:
            print("In units of B area:", sigma_MC / self.sys_par['areaB'])

        # orbital element stats
        a_arr = np.array(out_a)
        e_arr = np.array(out_e)
        a_au = (a_arr * u.m).to(u.au).value if a_arr.size else np.array([])
        print("Number of captured orbits:", len(a_au))
        if a_au.size:
            print("a (au): mean =", np.mean(a_au), "std =", np.std(a_au))
        if e_arr.size:
            print("e: mean =", np.mean(e_arr), "std =", np.std(e_arr))

        self.mc_results = {
            'v_inf': v_inf,
            'n_captured': n_captured,
            'sigma_MC_m2': sigma_MC,
            'sigma_MC_areaB': (sigma_MC / self.sys_par['areaB']) if 'areaB' in self.sys_par else None,
            'a_au': a_au,
            'e': e_arr,
            'cap_lambda': np.array(out_lambda),
            'cap_beta': np.array(out_beta),
            'cap_b': np.array(out_b),
            'cap_phi': np.array(out_phi),
        }
        self._save_mc_results(v_inf, N)

        return self.get_mc_results()
        
    def get_mc_results(self):
        return self.mc_results
    
    def get_system_params(self):
        return self.sys_par
    
