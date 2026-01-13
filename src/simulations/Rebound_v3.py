import rebound
import numpy as np
from astropy import units as u
import astropy.constants as const
import os
import datetime
import src.utils.exceptions as exc
import src.utils.calculations as calcs

def get_script_version():
    """Get the last modification time of this script file"""
    script_path = os.path.abspath(__file__)
    mod_time = os.path.getmtime(script_path)
    return datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')

class OrbitalSimulation:

    def __init__(self, configuration, rng):
        self.sys_par = configuration.get_system_param(all=True)
        self.rng = rng
        self.configuration = configuration
        # Expect SI; convert to ('AU','yr','Msun') as needed
        self._set_system()
        # self._print_script_version()

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
        result_keys = ['i', 'v_inf', 'start_info', 'errors', 'start_distances', 'lifetime', 'integrations','final_energy_c', 'termination_flag', 'eccentricities', 'semi_major_axes', 'orbital_periods', 'times', 'collision_info', 'script_version']
        result_dict = {key: value for key, value in zip(result_keys, result)}
        v_str = f"{v:.0f}"
        save_dir = self.configuration.get_save_dir_rebound()
        dir_npz = os.path.join(save_dir, f"v{v_str}")
        os.makedirs(dir_npz, exist_ok=True)
        out_npz = os.path.join(dir_npz, f"sim_{i}_{self.seed_base}.npz")
        np.savez(out_npz, **result_dict)

    def _check_result_exists(self, v, i):
        v_str = f"{v:.0f}"
        save_dir = self.configuration.get_save_dir_rebound()
        dir_npz = os.path.join(save_dir, f"v{v_str}")
        out_npz = os.path.join(dir_npz, f"sim_{i}_{self.seed_base}.npz")
        return os.path.isfile(out_npz)

    def _save_figure(self, op, i, v, flag, sim_t):
        v_str = f"{v:.0f}"
        save_dir = self.configuration.get_save_dir_plots()
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
    
    def _print_script_version(self):
        version = get_script_version()
        print(f"Running Rebound_v3.py, last modified on {version}")
    

    def run_orbital_integration(self, i, v_inf, lambda1, beta, phi, b, a_c, e_c, check_exists=True):

        # if check_exists:
        #     if self._check_result_exists(v_inf / 1e3, i):
        #         print(f"Simulation {i} for v_inf {v_inf/1e3} km/s already exists. Skipping...")
        #         return



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



        P_C = sim.particles[2].orbit(primary=sim.particles[0]).P
        P_B = sim.particles[1].orbit(primary=sim.particles[0]).P

        sim.dt = min(abs(P_C), abs(P_B)) * 0.05

        snap_rate = 20
        snap_interval = sim.dt * snap_rate
        t_end = int(1e7)  # years
        t_min = int(1e3) * abs(P_C)
        t_max = int(1e8) 
        # aC_max = aB * 40

        eccentricities = []
        semi_major_axes = []
        orbital_periods = []
        times = []
        E_c_cond, flag, result = None, None, None
        max_steps = int(np.ceil(t_end / sim.dt))
        sim.collision = "direct"
        sim.collision_resolve = "halt"  # Stop integration on collision
        
        # Variable to store collision info
        collision_info = None

        start_info = (f'dt: {sim.dt}, snapshot_rate: {snap_rate}, snap_interval: {snap_interval}, max steps: {max_steps}, '
            f't_end: {t_end}, t_min: {t_min}, t_max: {t_max}, orbital period C: {P_C}, orbital period B: {P_B}, '
            f'rclose {rclose}, vC wrt vB: {np.linalg.norm(vBVec - np.array(sim.particles[2].vxyz))}, '
            f'b: {b} bmin: {bmin}, bmax: {bmax}')

        # try:
        if not np.isfinite(P_C) or P_C <= 0:
            error_list.append('invalid orbital period C')
            # flag = 'preprocessing_error'
            # result = [i, v_inf, start_info, error_list, start_separation, sim.t, E_c_cond, flag, eccentricities, semi_major_axes, times]
            # self._save_result(v_inf_kms, result)
            # raise exc.InvalidPeriodError(f"Invalid orbital period for C: {P_C}")

        if not np.isfinite(P_B) or P_B <= 0:
            error_list.append('invalid orbital period B')
            # flag = 'preprocessing_error'
            # result = [i, v_inf, start_info, error_list, start_separation, sim.t, None, None, None, None, None]
            # self._save_result(v_inf_kms, result)
            # raise exc.InvalidPeriodError(f"Invalid orbital period for B: {P_B}")

        if not np.isfinite(sim.dt) or sim.dt <= 0:
            error_list.append('invalid time step')
            # flag = 'preprocessing_error'
            # result = [i, v_inf, start_info, error_list, start_separation, sim.t, None, None, None, None, None]
            # self._save_result(v_inf_kms, result)
            # raise exc.InvalidTimeStepError(f"Invalid time step: {sim.dt}")
        
        # except (exc.InvalidPeriodError, exc.InvalidTimeStepError) as e:
        #     # print(f"Error during pre-integration checks: {e}, Simulation failed for system {i}. Script continues...")
        #     pass
        #     # return


        counter = 0
        booster = 10
        # plot_counter = 0
        flag = None
        E_initial = sim.energy()
        ops = None
        int_start = datetime.datetime.now()
        try:
            # ops = rebound.OrbitPlotSet(sim, slices=True, unitlabel="[AU]", color=["black", "red"])
            # self._save_figure(ops, i, v_inf_kms, 'initial', sim.t)
            j = 0
            while (sim.t < t_end):
                P_C = sim.particles[2].orbit(primary=sim.particles[0]).P
                P_B = sim.particles[1].orbit(primary=sim.particles[0]).P
                if (sim.dt > np.min([abs(P_C), abs(P_B)]) * 0.05) or (sim.dt < np.min([abs(P_C), abs(P_B)]) * 0.03):
                    sim.dt = np.min([abs(P_C), abs(P_B)]) * 0.05
                    # print(f"Adjusted time step: {sim.dt}")
                j += booster
                time = sim.t + sim.dt * booster
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
                int_current = datetime.datetime.now()
                elapsed_int = (int_current - int_start).total_seconds()

                if error > 1e-5:
                    flag = 'energy_conservation'
                    raise exc.EnergyError(f"Error in energy conservation: {error}")

                if (E_c_cond > 0):
                    flag = 'escape_C'
                    raise exc.EscapeError(f"Particle C is free: E_c:  {E_c_cond}, rAC: {np.abs(sim.particles[2] ** sim.particles[0])}")
                if (E_b_cond > 0):
                    flag = 'escape_B'
                    raise exc.EscapeError(f"Particle B is free: E_b: {E_b_cond}, rAB: {np.abs(sim.particles[1] ** sim.particles[0])}")

                if elapsed_int > 1800:  # 30 minutes
                    flag = 'time_exceeded'
                    raise exc.MaxIntegrationTimeError(f"Maximum time for integration exceeded: {elapsed_int} > 1800 seconds")



                # yr += step
                counter += booster
                # plot_counter += 1
                # if yr == int(t_min):
                #     print(f"Simulation {i} at time {sim.t}, step {j}")
                #     print(f'Eccentricity of C : {sim.particles[2].e}')
                #     print(f'PBH E_cond: {E_c_cond}, Jupiter E_cond {E_b_cond} rCB {sim.particles[1]  ** sim.particles[2]}, rAC: {rAC}, error {error}')
                #     yr = 0
        except rebound.Collision as e:
            # Extract collision information from Rebound
            flag = 'collision'
            collision_info = {
                'time': sim.t,
                'colliding_particles': str(e)
            }
            # Try to identify which particles collided
            # Rebound collision exception message typically contains particle indices
            try:
                # Check distances between all particles to identify collision
                dist_AB = np.abs(sim.particles[0] ** sim.particles[1])
                dist_AC = np.abs(sim.particles[0] ** sim.particles[2])
                dist_BC = np.abs(sim.particles[1] ** sim.particles[2])
                
                collision_pairs = []
                if dist_AB < (rA + rB):
                    collision_pairs.append('A-B')
                if dist_AC < (rA + rC):
                    collision_pairs.append('A-C')
                if dist_BC < (rB + rC):
                    collision_pairs.append('B-C')
                
                collision_info['collision_pairs'] = collision_pairs
                collision_info['distances'] = {'AB': dist_AB, 'AC': dist_AC, 'BC': dist_BC}
                
                print(f"Collision detected in system {i} at time {sim.t}: {collision_pairs}")
            except:
                print(f"Collision detected in system {i} at time {sim.t}, but couldn't identify particles")
            
            print(f"Collision during integration: {e}, Simulation for system {i} at step {j}.")

        except (exc.PlottingError, exc.EnergyError, exc.EscapeError, exc.MaxIntegrationTimeError) as e:
            print(f"Error during integration: {e}, Simulation failed for system {i}, at step {j}. Script continues...")

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
        if flag is None:
            flag = 'completed'  # If no flag was set, assume collision occurred
        if result is None:
            result = [i, v_inf, start_info, error_list, start_separation, sim.t, j, E_c_cond, flag,
                      np.array(eccentricities, dtype=float), np.array(semi_major_axes, dtype=float),
                      np.array(orbital_periods, dtype=float), np.array(times, dtype=float), collision_info, get_script_version()]
            self._save_result(v_inf_kms, result)

        print(f"Simulation {i}  completed at {datetime.datetime.now()}")