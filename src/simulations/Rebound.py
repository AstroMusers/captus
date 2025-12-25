import rebound
import numpy as np
from astropy import units as u
import astropy.constants as const
import os
import datetime
import src.utils.exceptions as exc
import src.utils.calculations as calcs

class OrbitalSimulation:

    def __init__(self, system_param_dict, rng):
        self.sys_par = system_param_dict
        self.rng = rng
        # Expect SI; convert to ('AU','yr','Msun') as needed
        self._set_system()

    def _set_system(self):

        G_MsunAuYr = const.G.to(u.au**3 / (u.Msun * u.yr**2)).value  # Gravitational constant in AU^3 / (Msun * yr^2)
        self.mA = (self.sys_par['mA']*u.kg).to(u.Msun).value
        self.mB = (self.sys_par['mB']*u.kg).to(u.Msun).value
        self.mC = (self.sys_par['mC']*u.kg).to(u.Msun).value
        self.epsilon = self.sys_par.get('epsilon', 0.1)
        self.aB = (self.sys_par['aB'] * u.m).to(u.au).value
        self.rA = (self.sys_par.get('rA', 0.0) * u.m).to(u.au).value if self.sys_par.get('rA') else 0.0
        self.rB = (self.sys_par['rB'] * u.m).to(u.au).value
        self.rC = (self.sys_par.get('rC', 0.0) * u.m).to(u.au).value if self.sys_par.get('rC') else 0.0
        self.eB = self.sys_par.get('eB', 0.0489)  # default Jupiter eccentricity
        self.iB = self.sys_par.get('iB', 0.0)  # default Jupiter inclination in radians
        self.name = self.sys_par['name']
        self.seed_base = self.sys_par['seed_base']

        self.rClose = calcs.r_close(self.epsilon, self.mA, self.mB, self.aB)


    def _save_result(self, v, result):
        i = result[0]
        result_keys = ['i', 'v_inf', 'start_info', 'errors', 'start_distances', 'lifetime', 'final_energy_c', 'termination_flag', 'eccentricities', 'semi_major_axes', 'times']
        result_dict = {key: value for key, value in zip(result_keys, result)}
        v_str = f"{v:.0f}"
        save_dir = f'../runs/{self.name}/Rebound_Simulation_Results/v{v_str}'
        out_npz = os.path.join(save_dir, f"sim_{i}_{self.seed_base}.npz")
        os.makedirs(os.path.dirname(out_npz), exist_ok=True)
        np.savez(out_npz, **result_dict)

    def _save_figure(self, op, i, v, flag, sim_t):
        v_str = f"{v:.0f}"
        save_dir = f'../plots/{self.name}/Rebound_Simulation_Results/v{v_str}'
        os.makedirs(save_dir, exist_ok=True)
        fig_name = f"sim_{i}_{flag}_t{sim_t:.0f}_s{self.seed_base}.png"
        fig_path = os.path.join(save_dir, fig_name)
        op.fig.savefig(fig_path)
        if 'final' in flag:
            op.fig.clf()

    def _track_preprocess_errors(self, errors):
        error_list = ['v1primeMag is zero', 'invalid impact parameter b']
        l = []
        for e, e_l in zip(errors, error_list):
            if e:
                l.append(e_l)
        return l
    

    def run_orbital_integration(self, i, v_inf, lambda1, beta, phi, b, a_c, e_c):


        rng_R = self.rng  # Different seed for each process

        print(f"Simulation {i} of starting at {datetime.datetime.now()}")

        epsilon = self.epsilon
        # Convert impact parameter from meters to AU (MC upstream is in SI)
        b = (b*u.m).to(u.au).value
        rclose = self.rClose

        eB = self.eB
        eC = e_c  # from MC

        mA = self.mA  # Msun
        mB = self.mB  # Jupiter mass in Msun
        mC = self.mC  # PBH mass in Msun

        # Radii in AU
        rA = self.rA
        rB = self.rB
        rC = self.rC # PBH radius in AU for 1e-13 Msun

        # Semi-major axis in AU
        aB = self.aB   # Jupiter semi-major axis in AU
        aC = a_c  # from MC in AU

        iB = self.iB  # Jupiter inclination in radians
        iC = rng_R.uniform(0, np.pi)  # PBH inclination in radians

        thetaB = 0.0  # longitude of ascending node for Jupiter
        thetaC = rng_R.uniform(0, 2*np.pi)  # PBH longitude of ascending node

        # Velocity at infinity in AU/yr
        v_inf_kms = v_inf / 1e3  # km/s
        v_inf = (v_inf_kms * u.km/u.s).to(u.au/u.yr).value

        # Gravitational parameter in these units: G = 4*pi^2 AU^3/(Msun*yr^2)
        sim = rebound.Simulation()
        sim.units = ('AU', 'yr', 'Msun')
        sim.integrator = "MERCURIUS"
        sim.add(m=mA, r=rA)  # Sun
        sim.add(m=mB, a=aB, e=eB, r=rB, inc=iB, theta=thetaB, primary=sim.particles[0])  # Jupiter
        sim.add(m=mC, a=aC, e=eC, r=rC, inc=iC, theta=thetaC, primary=sim.particles[0])  # PBH

        G_unit = sim.G  # 4*pi^2 in these units
        sim.move_to_com()



        # # Compute params in AU/yr/Msun
        muA = G_unit * mA
        muB = G_unit * mB
        v1Mag = calcs.v_1_mag(v_inf, muA, muB, aB, rclose)

        # Build incoming velocity vector in AU/yr
        v1Vec = calcs.v_1_vec(v1Mag, lambda1, beta)  # returns 3-vector consistent with our convention
        v1_normalised = v1Vec / np.linalg.norm(v1Vec)

        vBVec = np.array(sim.particles[1].vxyz)  # AU/yr
        vBMag = np.linalg.norm(vBVec)

        v1prime = calcs.v_1_prime_vec(v1Vec, vBVec, lambda1, beta)
        v1primeMag = np.linalg.norm(v1prime)

        if v1primeMag == 0:
            vprimemag_error = True
        else:
            vprimemag_error = False

        # # Impact parameter limits in AU
        bmax = rclose
        bmin = calcs.b_min(muB, rB, v1primeMag)

        if (b < bmin) or (b > bmax) or (bmax < bmin):
            b_error = True
        else:
            b_error = False

        error_list = self._track_preprocess_errors([vprimemag_error, b_error])

        initial_BC_distance = sim.particles[1] ** sim.particles[2]
        initial_AB_distance = sim.particles[0] ** sim.particles[1]
        initial_AC_distance = sim.particles[0] ** sim.particles[2]
        start_separation = f'Initial distances: BC {initial_BC_distance}, AB {initial_AB_distance}, AC {initial_AC_distance}'



        P_C = sim.particles[2].P
        P_B = sim.particles[1].P

        sim.dt = abs(min(P_C, P_B) * 0.05)

        snap_rate = 20
        snap_interval = sim.dt * snap_rate
        t_end = int(1e6) * P_C
        t_min = int(1e3) * P_C
        t_max = int(1e7) * P_B
        # aC_max = aB * 40

        eccentricities = []
        semi_major_axes = []
        orbital_periods = []
        times = []
        E_c_cond, flag, result = None, None, None
        max_steps = int(np.ceil(t_end / sim.dt))
        sim.collision = "direct"

        start_info = (f'dt: {sim.dt}, snapshot_rate: {snap_rate}, snap_interval: {snap_interval}, max steps: {max_steps}, '
            f't_end: {t_end}, t_min: {t_min}, t_max: {t_max}, orbital period C: {P_C}, orbital period B: {P_B}, '
            f'rclose {rclose}, vC wrt vB: {np.linalg.norm(vBVec - np.array(sim.particles[2].vxyz))}, '
            f'b: {b} bmin: {bmin}, bmax: {bmax}')

        try:
            if not np.isfinite(P_C) or P_C <= -0.1:
                error_list.append('invalid orbital period C')
                flag = 'preprocessing_error'
                result = [i, v_inf, start_info, error_list, start_separation, sim.t, E_c_cond, flag, eccentricities, semi_major_axes, times]
                self._save_result(v_inf_kms, result)
                raise exc.InvalidPeriodError(f"Invalid orbital period for C: {P_C}")

            if not np.isfinite(P_B) or P_B <= -0.1:
                error_list.append('invalid orbital period B')
                flag = 'preprocessing_error'
                result = [i, v_inf, start_info, error_list, start_separation, sim.t, None, None, None, None, None]
                self._save_result(v_inf_kms, result)
                raise exc.InvalidPeriodError(f"Invalid orbital period for B: {P_B}")

            if not np.isfinite(sim.dt) or sim.dt <= 0:
                error_list.append('invalid time step')
                flag = 'preprocessing_error'
                result = [i, v_inf, start_info, error_list, start_separation, sim.t, None, None, None, None, None]
                self._save_result(v_inf_kms, result)
                raise exc.InvalidTimeStepError(f"Invalid time step: {sim.dt}")
        
        except (exc.InvalidPeriodError, exc.InvalidTimeStepError) as e:
            print(f"Error during pre-integration checks: {e}, Simulation failed for system {i}. Script continues...")

            return


        counter = 0
        # plot_counter = 0
        flag = None
        E_initial = sim.energy()
        ops = None
        yr = 0
        try:
            # ops = rebound.OrbitPlotSet(sim, slices=True, unitlabel="[AU]", color=["black", "red"])
            # self._save_figure(ops, i, v_inf_kms, 'initial', sim.t)

            for j in range(1, max_steps):
                time = j * sim.dt
                sim.integrate(time, exact_finish_time=0)
                E_current = sim.energy()
                error = abs(E_current - E_initial)/E_initial
                if counter == snap_rate:
                    # Store positions and orbital elements
                    # positions_jup[j] = [sim.particles[1].x, sim.particles[1].y, sim.particles[1].z]
                    # positions_pbh[j] = [sim.particles[2].x, sim.particles[2].y, sim.particles[2].z]

                    orbit = sim.particles[2].orbit(primary=sim.particles[0])
                    eccentricities.append(orbit.e)
                    semi_major_axes.append(orbit.a)
                    orbital_periods.append(orbit.P)
                    times.append(sim.t)
                    counter = 0
                    # if plot_counter > snap_plot_rate:
                    #     ops.update()
                    #     ops.fig.savefig(f'{directoryp}Orbits_Simulation_{i}_Time_{int(sim.t)}.png')
                    #     plot_counter = 0

                # Specific orbital energy of C around A (AU^2/yr^2)
                Evc = 0.5 * (sim.particles[2].vx**2 + sim.particles[2].vy**2 + sim.particles[2].vz**2)
                rAC = np.abs(sim.particles[2] ** sim.particles[0])
                E_c_cond = Evc - (G_unit*sim.particles[0].m / rAC)

                # Specific orbital energy of B around A (AU^2/yr^2)
                Evb = 0.5 * (sim.particles[1].vx**2 + sim.particles[1].vy**2 + sim.particles[1].vz**2)
                rAB = np.abs(sim.particles[1] ** sim.particles[0])
                E_b_cond = Evb - (G_unit*sim.particles[0].m / rAB)

                # if firstCaptureA != True:
                #     capture = capture + 1 if E_cond < 0 else 0
                #     if capture > 3:
                #         firstCaptureA = True

                # for p in sim.particles:
                #     if p.last_collision == sim.t:
                #         flag = 'collision'
                #         raise CollisionError(f"Collision detected at time {sim.t}")
                    
                if error > 1e-5:
                    flag = 'energy_conservation'
                    raise exc.EnergyError(f"Error in energy conservation: {error}")

                if (E_c_cond > 0):
                    flag = 'escape_C'
                    raise exc.EscapeError(f"Particle C is free: E_c:  {E_c_cond}, rAC: {np.abs(sim.particles[2] ** sim.particles[0])}")
                if (E_b_cond > 0):
                    flag = 'escape_B'
                    raise exc.EscapeError(f"Particle B is free: E_b: {E_b_cond}, rAB: {np.abs(sim.particles[1] ** sim.particles[0])}")

                if sim.t > t_max:
                    flag = 'time_exceeded'
                    raise exc.MaxYearForBError(f"Maximum time for B exceeded: {sim.t} > {t_max}")



                # yr += step
                counter += 1
                # plot_counter += 1
                # if yr == int(t_min):
                #     print(f"Simulation {i} at time {sim.t}, step {j}")
                #     print(f'Eccentricity of C : {sim.particles[2].e}')
                #     print(f'PBH E_cond: {E_c_cond}, Jupiter E_cond {E_b_cond} rCB {sim.particles[1]  ** sim.particles[2]}, rAC: {rAC}, error {error}')
                #     yr = 0
        except (exc.PlottingError, rebound.Collision, exc.EnergyError, exc.EscapeError, exc.MaxYearForBError) as e:
            print(f"Error during integration: {e}, Simulation failed for system {i}, at time {sim.t}. Script continues...")

        # E_final = sim.energy()
        # energy_change = abs(E_final - E_initial)/E_initial
        # print(f'final energy {E_final}, final time {sim.t}, final distance {sim.particles[1]  ** sim.particles[2]}, energy change {energy_change}')

        # print(f'r_close = {rclose} and initial BC separation =  {initial_BC_distance}')
        # if ops is not None:
        #     ops.update()
        #     self._save_figure(ops, i, v_inf_kms, f'final_{flag}', sim.t)

        # result = {
        #     "i": i,
        #     "start_info": start_info,
        #     "errors": error_list,
        #     "start_distances": start_separation,
        #     "lifetime": sim.t,
        #     "final_energy_c": E_c_cond,
        #     "termination_flag": flag,
        #     "eccentricities": np.array(eccentricities, dtype=float),
        #     "semi_major_axes": np.array(semi_major_axes, dtype=float),
        #     'times': np.array(times, dtype=float),
        # }
        if result is None:
            result = [i, v_inf, start_info, error_list, start_separation, sim.t, E_c_cond, flag,
                      np.array(eccentricities, dtype=float), np.array(semi_major_axes, dtype=float),
                      np.array(times, dtype=float)]
            self._save_result(v_inf_kms, result)

        print(f"Simulation {i}  completed at {datetime.datetime.now()}")