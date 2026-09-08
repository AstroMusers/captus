import os
import astropy.constants as const
import captus.utils.calculations as calcs
import numpy as np
from astropy import units as u
import datetime as dt
import multiprocessing
from numpy.random import default_rng, SeedSequence, PCG64, Generator

class MonteCarloSimulation:

    def __init__(self, configuration, rng, verbose=False):
        # Initialization code here
        self.sys_par = configuration.get_system_param(all=True)
        self.configuration = configuration
        self.rng = rng
        self.verbose = verbose
        self._set_system()

    def _vprint(self, message):
        if self.verbose:
            print(message)

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
        self.areaB = np.pi * self.rB**2
        self.vBMag = self.sys_par['vB']
        self.v_escB = calcs.v_esc(self.muB, self.rB)

        self.epsilon = self.sys_par['epsilon']
        self.epsilon_adjust_coeff = self.sys_par['epsilon_adjust_coeff']
        self.rClose = calcs.r_close(self.epsilon, self.muA, self.muB, self.aB, approx=False)
        self.rHill = calcs.hill_radius(self.aB, self.mA, self.mB, self.eB)

        while self.rClose > self.rHill:
            self.epsilon *= self.epsilon_adjust_coeff
            self.rClose = calcs.r_close(self.epsilon, self.muA, self.muB, self.aB, approx=False)
            self._vprint(f"Adjusted rClose to {self.rClose} to be within Hill radius {self.rHill}, epsilon={self.epsilon}")

        self.aC = self.aB - self.rClose
        self.rC = calcs.schwarzchild_radius(self.mC)
        self.vEscape = calcs.v_esc(self.muA, self.aC)

        self._set_importance_sampling()

    def _save_mc_results(self, mc_results, N):
        # Code to save Monte Carlo results
        name = self.sys_par['name']
        seed = self.sys_par['seed_base']
        save_dir = self.configuration.get_save_dir_mc()

        filename = f'{save_dir}/monte_carlo_results_v{mc_results["v_inf"]/1e3:.1f}_s{seed}.npz'
        np.savez(filename, **mc_results)

    def _set_importance_sampling(self):
        self.importance_sampling = self.configuration.get_simulation_param('importance_sampling', all=False)
        self.sample_size = self.configuration.get_simulation_param('sample_size', all=False)
        self.trials = self.configuration.get_simulation_param('trials', all=False)
        self.max_trials = self.configuration.get_simulation_param('max_trials', all=False)
        self.e_lim = self.configuration.get_simulation_param('max_e', all=False)
        self.max_execution_time = self.configuration.get_simulation_param('max_execution_time', all=False)

    def set_importance_sampling(self, sampling, sample_size=None, e_lim=None, max_execution_time=None, max_trials=1_000_000_000):
        if getattr(self, 'importance_sampling'):
            self._vprint("Importance sampling parameters already set. Overriding with new values.")
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
        cap_lambda, cap_beta, cap_b, cap_phi = [], [], [], []
        cap_C_pos, cap_C_v2, cap_B_pos, cap_B_v = [], [], [], []
        cap_system_energy, cap_capture_energy, cap_a, cap_e = [], [], [], []
        cap_capture_cross_sections, cap_collision_cross_sections = [], []
        cap_bmin, cap_bmax = [], []
        cap_v1, cap_v1prime = [], []

        nocap_lambda, nocap_beta, nocap_b, nocap_phi = [], [], [], []
        no_cap_C_pos, no_cap_C_v2, no_cap_B_pos, no_cap_B_v = [], [], [], []
        nocap_system_energy, nocap_capture_energy, nocap_a, nocap_e = [], [], [], []
        nocap_bmin, nocap_bmax = [], []

        all_lambda, all_beta, all_phi = [], [], []
        all_bmax, all_bmin = [], []
        all_v1, all_v1prime = [], []
        sampled, failed, check0, check1, check2, check3, check4 = 0, 0, 0, 0, 0, 0, 0
        now = dt.datetime.now()
        while sampled < N and not quota_condition and (dt.datetime.now() - now).total_seconds() < self.max_execution_time:
            remaining = N - sampled
            batch_size = min(remaining, 1_000_000)  # process in batches of 1,000,000

            # 1) sample direction of incoming PBH (λ, β)
            lambda_samples = self.rng.uniform(0, 2 * np.pi, size=batch_size)
            sinbeta_samples = self.rng.uniform(-1, 1, size=batch_size)    # sinβ
            beta_samples = np.arcsin(sinbeta_samples)
            # beta_samples = self.rng.uniform(0, 2*np.pi, size=batch_size)  # Directly sample β using uniform(-π/2, π/2)
            # 3) sample scattering-plane angle φ
            phi_samples = self.rng.uniform(0, 2 * np.pi, size=batch_size)

            # Shared constants
            for lam, beta, phi in zip(lambda_samples, beta_samples, phi_samples):
                check0 += 1
                all_lambda.append(lam)
                all_beta.append(beta)
                all_phi.append(phi)

                v1Mag = calcs.v_1_mag(v_inf, self.muA, self.muB, self.aC, self.rClose)
                v1Vec = calcs.v_1_vec(v1Mag, beta)
                vBVec = calcs.v_B_vec(self.vBMag, lam)

                v1primeVec = calcs.v_1_prime_vec(v1Vec, vBVec)
                v1primeMag = np.linalg.norm(v1primeVec)

                all_v1.append(v1Vec)
                all_v1prime.append(v1primeVec)

                spec_UE1 = calcs.potential_energy(self.muA, self.aC, self.muB, self.rClose)

                b_min = calcs.b_min(self.muB, self.rB, v1primeMag)

                b_max = calcs.compute_b_max(self.rClose, b_min,
                                            v1primeVec, v1primeMag, vBVec,
                                            self.muB, phi,
                                            spec_UE1, self.rClose)
                all_bmin.append(b_min)
                all_bmax.append(b_max)
                if b_max <= b_min:
                    # print("Skipped due to b_max <= b_min:", b_max, "<=", b_min)
                    failed += 1
                    if failed > 100_000:
                        self._vprint("Too many consecutive failures in finding valid b range. Exiting early.")
                        quota_condition = True
                        break
                    continue
                failed = 0
                check1 += 1

                u_b = self.rng.uniform(b_min**2, b_max**2)  # Sample uniformly in b^2 to ensure uniform distribution in area
                b = np.sqrt(u_b)

                if b < b_min:
                    # print("Skipped due to b < bmin:", b, "<", bmin)
                    continue

                check2 += 1 
                sampled += 1



                v2primeVec = calcs.v_2_prime_vec(v1primeVec, v1primeMag, self.muB, b, phi)

                v2Vec = calcs.v_2_vec(v2primeVec,vBVec)
                v2Mag = calcs.v_2_mag(v2Vec)
                rABVec = calcs.r_AB_vec(self.aB, lam)

                bVec_unit = calcs.b_unit_vector(v1primeMag, v1primeVec, phi)
                bVec = b * bVec_unit
                exit_point_B_frame = calcs.exit_point_from_scatter(v1primeMag, v1primeVec, bVec=bVec, muB=self.muB, rClose=self.rClose, b=b)
                exitVec = exit_point_B_frame + rABVec # transfering exit point from B frame to A frame
                exitMag = np.linalg.norm(exitVec)
                # bVec = calcs.b_vector(self.muB, v1primeMag, v1primeVec, phi, b)
                # exit_point_B_frame2 = calcs.exit_point_from_scatter(v1primeVec, v2primeVec, bVec, self.muB, self.rClose, b=b)
                # exit_point2 = exit_point_B_frame2 + rABVec
                # distance_A_to_exit2 = np.linalg.norm(exit_point2)
                # E2 = 0.5 * v2Mag**2 + calcs.potential_energy(self.muA, distance_A_to_exit2, self.muB, self.rClose)

                spec_UE2 = calcs.potential_energy(self.muA, exitMag, self.muB, self.rClose)
                E2_system = 0.5 * v2Mag**2 + spec_UE2
                E2_capture = 0.5 * v2Mag**2 - (self.muA / exitMag)

                # percentage_diff = np.linalg.norm(exit_point - exit_point2) / np.linalg.norm(exit_point2) * 100
                # percentage_diff_E = np.abs(E2 - E2_system) / np.abs(E2_system) * 100
                # cap_system_energy.append(percentage_diff_E)
                # cap_C_pos.append(percentage_diff)
                L2_val = calcs.specific_L2(exitVec, v2Vec)
                a_val, e_val = calcs.a_e(self.muA, E2_capture, L2_val)

                if E2_system < 0:
                    # print("Skipped due to E2_system < 0:", E2_system)
                    check3 += 1

                if E2_capture < 0:
                    check4 += 1

                    if a_val > 0 and 0 <= e_val < (1 if self.e_lim is None else self.e_lim):
                        capture_crossec = calcs.capture_cross_section(b_min, b_max, b, 1)[0]
                        collision_crossec = calcs.collision_cross_section(self.rB, self.v_escB, v1primeMag)
                        cap_a.append(a_val)
                        cap_e.append(e_val)
                        cap_lambda.append(lam)
                        cap_beta.append(beta)
                        cap_b.append(b)
                        cap_phi.append(phi)
                        cap_bmin.append(b_min)
                        cap_bmax.append(b_max)
                        cap_C_pos.append(exitVec)
                        cap_C_v2.append(v2Vec)
                        cap_B_pos.append(rABVec)
                        cap_B_v.append(vBVec)
                        cap_system_energy.append(E2_system)
                        cap_capture_energy.append(E2_capture)
                        cap_capture_cross_sections.append(capture_crossec.value)
                        cap_collision_cross_sections.append(collision_crossec)
                        cap_v1.append(v1Vec)
                        cap_v1prime.append(v1primeVec)
                        n_captured += 1
                else:
                    nocap_lambda.append(lam)
                    nocap_beta.append(beta)
                    nocap_b.append(b)
                    nocap_phi.append(phi)
                    nocap_bmin.append(b_min)
                    nocap_bmax.append(b_max)
                    nocap_system_energy.append(E2_system)
                    nocap_capture_energy.append(E2_capture)
                    nocap_a.append(a_val)
                    nocap_e.append(e_val)
                    no_cap_C_pos.append(exitVec)
                    no_cap_C_v2.append(v2Vec)
                    no_cap_B_pos.append(rABVec)
                    no_cap_B_v.append(vBVec)

                if self.importance_sampling and n_captured >= self.sample_size:
                    quota_condition = True
                    break
                

        
        # estimate capture cross-section
        sigma_MC = (n_captured / sampled) * float(np.pi) * self.rClose**2 if n_captured > 0 else 0
        sigma_MC_dsigma = calcs.capture_cross_section_MC(cap_b, sampled).value if n_captured> 0 else 0
        self._vprint(f"for v_inf={v_inf/1e3} km/s and mC {self.mC/const.M_sun.value} M_sun MC capture cross-section: {sigma_MC} m^2")
        self._vprint(f"for v_inf={v_inf/1e3} km/s and mC {self.mC/const.M_sun.value} M_sun MC capture cross-section (dσ avg): {sigma_MC_dsigma} au^2")

        # orbital element stats
        cap_a = np.array(cap_a)
        cap_e = np.array(cap_e)
        cap_a_au = (cap_a * u.m).to(u.au).value if cap_a.size else np.array([])

        nocap_a = np.array(nocap_a)
        nocap_e = np.array(nocap_e)
        nocap_a_au = (nocap_a * u.m).to(u.au).value if nocap_a.size else np.array([])
        self._vprint(f"Number of captured orbits: {n_captured}, e condition met: {check3}, out of {sampled} samples.")

        mc_results = {
            'v_inf': v_inf,
            'n_captured': n_captured,
            'sigma_MC_m2': sigma_MC,
            'sigma_MC_dsigma_au2': sigma_MC_dsigma,
            'cap_a_au': cap_a_au,
            'cap_e': cap_e,
            'cap_lambda': np.array(cap_lambda),
            'cap_beta': np.array(cap_beta),
            'cap_b': np.array(cap_b),
            'cap_phi': np.array(cap_phi),
            'cap_C_pos': np.array(cap_C_pos),
            'cap_C_v2': np.array(cap_C_v2),
            'cap_B_pos': np.array(cap_B_pos),
            'cap_B_v': np.array(cap_B_v),
            'cap_system_energy': np.array(cap_system_energy),
            'cap_capture_energy': np.array(cap_capture_energy),
            'cap_capture_cross_sections': np.array(cap_capture_cross_sections),
            'cap_collision_cross_sections': np.array(cap_collision_cross_sections),
            'cap_bmin': np.array(cap_bmin),
            'cap_bmax': np.array(cap_bmax),
            'cap_v1': np.array(cap_v1),
            'cap_v1prime': np.array(cap_v1prime),
            'nocap_a_au': nocap_a_au,
            'nocap_e': nocap_e,
            'nocap_lambda': np.array(nocap_lambda),
            'nocap_beta': np.array(nocap_beta),
            'nocap_b': np.array(nocap_b),
            'nocap_phi': np.array(nocap_phi),
            'nocap_bmin': np.array(nocap_bmin),
            'nocap_bmax': np.array(nocap_bmax),
            'nocap_system_energy': np.array(nocap_system_energy),
            'nocap_capture_energy': np.array(nocap_capture_energy),
            'nocap_C_pos': np.array(no_cap_C_pos),
            'nocap_C_v2': np.array(no_cap_C_v2),
            'nocap_B_pos': np.array(no_cap_B_pos),
            'nocap_B_v': np.array(no_cap_B_v),
            'all_lambda': np.array(all_lambda),
            'all_beta': np.array(all_beta),
            'all_phi': np.array(all_phi),
            'all_v1': np.array(all_v1),
            'all_v1prime': np.array(all_v1prime),
            'all_bmin': np.array(all_bmin),
            'all_bmax': np.array(all_bmax),
            'epsilon': self.epsilon,
            'sample_number': sampled,
            'b_min': np.array(cap_bmin),
            'b_max': np.array(cap_bmax),
            'checks': (sampled, failed, check0, check1, check2, check3, check4, n_captured),
            'execution_time': (dt.datetime.now() - now).total_seconds()/60  # in minutes
        }
        self.mc_results = mc_results
        self._save_mc_results(mc_results, N)
        self._vprint(f'Results saved for v_inf={v_inf/1e3} km/s with N={sampled} trials.')
        return 
        
    def get_mc_results(self):
        return self.mc_results
    
    def get_system_params(self):
        return self.sys_par
    

        
    
