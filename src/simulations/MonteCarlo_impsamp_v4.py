import os
import astropy.constants as const
import src.utils.calculations_v2 as calcs
import numpy as np
from astropy import units as u
import datetime as dt
import multiprocessing
from numpy.random import default_rng, SeedSequence, PCG64, Generator

class MonteCarloSimulation:

    def __init__(self, configuration, rng):
        # Initialization code here
        self.sys_par = configuration.get_system_param(all=True)
        self.configuration = configuration
        self.rng = rng
        self._set_system()

    def _set_system(self):

        G = const.G.value  # Gravitational constant in m^3 kg^-1 s^-2

        self.mA = self.sys_par['mA']
        self.mB = self.sys_par['mB']
        self.mC = self.sys_par['mC'] 

        # self.muA = self.mA * G
        # self.muB = self.mB * G
        # self.muC = self.mC * G
        self.muA = calcs.standard_gravitational_parameter(self.mA, self.mC, approx=False)
        self.muB = calcs.standard_gravitational_parameter(self.mB, self.mC, approx=False)

        self.aB = self.sys_par['aB']
        self.rB = self.sys_par['rB']
        self.eB = self.sys_par['eB']
        self.iB = self.sys_par['iB']

        self.vBMag = self.sys_par['vB']
        self.epsilon = self.sys_par['epsilon']

        self.rC = calcs.schwarzchild_radius(self.mC)

        self.areaB = np.pi * self.rB**2

        self.vEscape = calcs.v_esc(self.muA, self.aB)
        self.rClose = calcs.r_close(self.epsilon, self.muA, self.muB, self.aB, approx=False)
        self.rHill = calcs.hill_radius(self.aB, self.mA, self.mB, self.eB)

        while self.rClose > self.rHill:
            self.epsilon *= 0.5
            self.rClose = calcs.r_close(self.epsilon, self.muA, self.muB, self.aB, approx=False)
            print(f"Adjusted rClose to {self.rClose} to be within Hill radius {self.rHill}, epsilon={self.epsilon}")


        self._set_importance_sampling()

    def _save_mc_results(self,mc_results, N):
        # Code to save Monte Carlo results
        name = self.sys_par['name']
        seed = self.sys_par['seed_base']
        save_dir = self.configuration.get_save_dir_mc()
        results = mc_results

        np.savez(f'{save_dir}/monte_carlo_results_v{results["v_inf"]/1e3:.1f}_s{seed}.npz',
                 v_inf=results['v_inf'],
                 n_captured=results['n_captured'],
                 sigma_MC_m2=results['sigma_MC_m2'],
                 sigma_MC_areaB=results['sigma_MC_areaB'],
                 a_au=results['a_au'],
                 e=results['e'],
                 cap_lambda=results['cap_lambda'],
                 cap_beta=results['cap_beta'],
                 cap_b=results['cap_b'],
                 cap_phi=results['cap_phi'],
                 cap_C_pos=results['cap_C_pos'],
                 cap_C_v2=results['cap_C_v2'],
                 cap_B_pos=results['cap_B_pos'],
                 cap_B_v=results['cap_B_v'],
                 epsilon=results['epsilon'],
                 sample_number=results['sample_number'],
                 execution_time=results['execution_time'],
                 sigma_MC_dsigma_m2=results['sigma_MC_dsigma_m2'],
                 sigma_MC_dsigma_captured_m2=results['sigma_MC_dsigma_captured_m2'],
                 sigma_MC_bmax_m2=results['sigma_MC_bmax_m2'],
                 b_min=results['b_min'],
                 b_max=results['b_max'],
                 checks=results['checks'],
                 capture_cross_sections=results['capture_cross_sections'],
                 capture_cross_sections_captured=results['capture_cross_sections_captured'],
                 collision_cross_sections=results['collision_cross_sections'])

    def _set_importance_sampling(self):
        self.importance_sampling = self.configuration.get_simulation_param('importance_sampling', all=False)
        self.sample_size = self.configuration.get_simulation_param('sample_size', all=False)
        self.trials = self.configuration.get_simulation_param('trials', all=False)
        self.max_trials = self.configuration.get_simulation_param('max_trials', all=False)
        self.e_lim = self.configuration.get_simulation_param('max_e', all=False)
        self.max_execution_time = self.configuration.get_simulation_param('max_execution_time', all=False)

    def set_importance_sampling(self, sampling, sample_size=None, e_lim=None, max_execution_time=None, max_trials=1_000_000_000):
        if getattr(self, 'importance_sampling'):
            print("Importance sampling parameters already set. Overriding with new values.")
        self.importance_sampling = sampling
        self.sample_size = sample_size
        self.max_trials = max_trials
        self.e_lim = e_lim
        self.max_execution_time = max_execution_time # in seconds


    def run_monte_carlo_simulation(self, v_inf):


        if self.importance_sampling:
            N = self.max_trials
        else:
            N = self.trials


        quota_condition = False
        n_captured = 0
        out_lambda, out_beta, out_b, out_phi, out_C_pos, out_C_v2, out_B_pos, out_B_v = [], [], [], [], [], [], [], []
        capture_crossections_captured = []
        capture_crossections = []
        collision_crossections = []
        out_bmin, out_bmax = [], []
        out_a, out_e = [], []
        sampled, failed, check1, check2, check3 = 0, 0, 0, 0, 0
        now = dt.datetime.now()
        while sampled < N and not quota_condition and (dt.datetime.now() - now).total_seconds() < self.max_execution_time:
            remaining = N - sampled
            batch_size = min(remaining, 1_000_000)  # process in batches of 1,000,000

            # 1) sample direction of incoming PBH (λ, β)
            lambda_samples = self.rng.uniform(0, 2 * np.pi, size=batch_size)
            cosbeta_samples = self.rng.uniform(-1, 1, size=batch_size)    # cosβ
            beta_samples = np.arccos(cosbeta_samples)

            # 3) sample scattering-plane angle φ
            phi_samples = self.rng.uniform(0, 2 * np.pi, size=batch_size)

            # Shared constants
            for lam, beta, phi in zip(lambda_samples, beta_samples, phi_samples):
                sampled += 1

                v1Mag = calcs.v_1_mag(v_inf, self.muA, self.muB, self.aB, self.rClose)
                v1Vec = calcs.v_1_vec(v1Mag, lam, beta)
                vBVec = calcs.v_B_vec(self.vBMag, lam)

                v1primeVec = calcs.v_1_prime_vec(v1Vec, vBVec, lam, beta)
                v1primeMag = np.linalg.norm(v1primeVec)

                spec_UE1 = calcs.potential_energy(self.muA, self.aB, self.muB, self.rClose)

                b_min = calcs.b_min(self.muB, self.rB, v1primeMag)

                b_max = calcs.compute_b_max(self.rClose, b_min,
                                            v1primeVec, v1primeMag, vBVec,
                                            self.muB, phi,
                                            spec_UE1, self.rClose)
                
                if b_max <= b_min:
                    # print("Skipped due to b_max <= b_min:", b_max, "<=", b_min)
                    failed += 1
                    if failed > 1_000_000:
                        print("Too many consecutive failures in finding valid b range. Exiting early.")
                        quota_condition = True
                        break
                    continue
                failed = 0
                check1 += 1
                capture_crossections.append(calcs.capture_cross_section(b_min, b_max))
                v_escB = calcs.v_esc(self.muB, self.rB)
                collision_crossections.append(calcs.collision_cross_section(self.rB, v_escB, v1primeMag))
                u_b = self.rng.uniform(0.0, 1.0, size=1)
                b = b_max * np.sqrt(u_b)

                if b < b_min:
                    # print("Skipped due to b < bmin:", b, "<", bmin)
                    continue
                check2 += 1 
                if v1primeMag == 0:
                    continue

                v2Vec = calcs.v_2_vec(v1primeVec, v1primeMag, vBVec, self.muB, b, phi)
                v2Mag = calcs.v_2_mag(v2Vec)

                v2primeVec = calcs.v_2_prime_vec(v2Vec, vBVec)
                rABVec = calcs.r_AB_vec(self.aB, lam)

                exit_point_B_frame = calcs.exit_point_from_scatter(v1primeVec, v2primeVec, self.muB, self.rClose, b=b)
                exit_point = exit_point_B_frame + rABVec
                distance_A_to_exit = np.linalg.norm(exit_point)
                spec_UE2 = calcs.potential_energy(self.muA, distance_A_to_exit, self.muB, self.rClose)
                E2_val = 0.5 * v2Mag**2 + spec_UE2

                if E2_val + (self.muB / self.rClose) < 0:
                    check3 += 1

                    L2_val = calcs.specific_L2(exit_point, v2Vec)
                    a_val, e_val = calcs.a_e(self.muA, E2_val, L2_val)

                    if a_val > 0 and 0 <= e_val < (1 if self.e_lim is None else self.e_lim):
                        out_a.append(a_val)
                        out_e.append(e_val)
                        out_lambda.append(lam)
                        out_beta.append(beta)
                        out_b.append(b)
                        out_phi.append(phi)
                        out_bmin.append(b_min)
                        out_bmax.append(b_max)
                        out_C_pos.append(exit_point)
                        out_C_v2.append(v2Vec)
                        out_B_pos.append(rABVec)
                        out_B_v.append(vBVec)
                        capture_crossections_captured.append(calcs.capture_cross_section(b_min, b_max))
                        n_captured += 1
                

                if self.importance_sampling and n_captured >= self.sample_size:
                    quota_condition = True
                    break
                

        
        # estimate capture cross-section
        sigma_MC = (n_captured / sampled) * float(np.pi) * self.rClose**2
        sigma_MC_dsigma = np.sum(capture_crossections) / sampled if len(capture_crossections) > 0 else 0.0
        sigma_MC_bmax = (n_captured / sampled) * float(np.pi) * max(out_bmax)**2 if n_captured > 0 else 0.0
        sigma_MC_dsigma_captured = np.sum(capture_crossections_captured) / sampled if n_captured > 0 else 0.0
        print(f"for v_inf={v_inf/1e3} km/s and mC {self.mC/const.M_sun.value} M_sun MC capture cross-section:", sigma_MC, "m^2")
        print(f"for v_inf={v_inf/1e3} km/s and mC {self.mC/const.M_sun.value} M_sun MC capture cross-section (dσ avg):", sigma_MC_dsigma, "m^2")
        print(f"for v_inf={v_inf/1e3} km/s and mC {self.mC/const.M_sun.value} M_sun MC capture cross-section (b_max):", sigma_MC_bmax, "m^2")

        # orbital element stats
        a_arr = np.array(out_a)
        e_arr = np.array(out_e)
        a_au = (a_arr * u.m).to(u.au).value if a_arr.size else np.array([])
        print(f"Number of captured orbits: {n_captured}, e condition met: {check3}, out of {sampled} samples.")

        mc_results = {
            'v_inf': v_inf,
            'n_captured': n_captured,
            'sigma_MC_m2': sigma_MC,
            'sigma_MC_dsigma_m2': sigma_MC_dsigma,
            'sigma_MC_dsigma_captured_m2': sigma_MC_dsigma_captured,
            'sigma_MC_bmax_m2': sigma_MC_bmax,
            'sigma_MC_areaB': (sigma_MC / self.areaB),
            'a_au': a_au,
            'e': e_arr,
            'cap_lambda': np.array(out_lambda),
            'cap_beta': np.array(out_beta),
            'cap_b': np.array(out_b),
            'cap_phi': np.array(out_phi),
            'cap_C_pos': np.array(out_C_pos),
            'cap_C_v2': np.array(out_C_v2),
            'cap_B_pos': np.array(out_B_pos),
            'cap_B_v': np.array(out_B_v),
            'epsilon': self.epsilon,
            'sample_number': sampled,
            'b_min': np.array(out_bmin),
            'b_max': np.array(out_bmax),
            'checks': (sampled, failed, check1, check2, check3, n_captured),
            'capture_cross_sections': np.array(capture_crossections),
            'capture_cross_sections_captured': np.array(capture_crossections_captured),
            'collision_cross_sections': np.array(collision_crossections),
            'execution_time': (dt.datetime.now() - now).total_seconds()/60  # in minutes
        }
        self.mc_results = mc_results
        self._save_mc_results(mc_results, N)
        print(f'Results saved for v_inf={v_inf/1e3} km/s with N={sampled} trials.')
        return 
        
    def get_mc_results(self):
        return self.mc_results
    
    def get_system_params(self):
        return self.sys_par
    

        
    
