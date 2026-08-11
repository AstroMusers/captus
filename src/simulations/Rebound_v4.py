import rebound
import numpy as np
from astropy import units as u
import astropy.constants as const
import os
import datetime
import src.utils.exceptions as exc
import src.utils.calculations_v3 as calcs

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
directoryp = os.path.join(REPO_ROOT, f'plots/ReboundSim/Plots')
def get_script_version():
    """Get the last modification time of this script file"""
    script_path = os.path.abspath(__file__)
    mod_time = os.path.getmtime(script_path)
    return datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')

class OrbitalSimulation:

    def __init__(self, configuration, rng):
        self.sys_par = configuration.get_system_param(all=True)
        self.sim_par = configuration.get_simulation_param(all=True)
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
        self.rA = (self.sys_par['rA'] * u.m).to(u.au).value
        self.rB = (self.sys_par['rB'] * u.m).to(u.au).value
        self.rC = (self.sys_par['rC'] * u.m).to(u.au).value
        self.eB = self.sys_par['eB']  # default Jupiter eccentricity
        self.iB = self.sys_par['iB']  # default Jupiter inclination in radians
        self.name = self.sys_par['name']
        self.seed_base = self.sys_par['seed_base']
        self.max_execution_time = self.sim_par['max_execution_time'] # in seconds
        self.rClose = calcs.r_close(self.epsilon, self.mA, self.mB, self.aB)

    def _save_set_parameters(self):
        # Save parameters for reproducibility
        save_dir = self.configuration.get_save_dir_rebound()
        os.makedirs(save_dir, exist_ok=True)
        param_file = os.path.join(save_dir, f'system_parameters_{self.seed_base}.npz')

        if os.path.isfile(param_file):
            # print(f"Parameter file {param_file} already exists. Skipping save to avoid overwriting.")
            return
        
        params = {
            'mA': self.mA,
            'mB': self.mB,
            'mC': self.mC,
            'aB': self.aB,
            'rA': self.rA,
            'rB': self.rB,
            'rC': self.rC,
            'eB': self.eB,
            'iB': self.iB,
            'epsilon': self.epsilon,
            'name': self.name,
            'seed_base': self.seed_base,
            'max_execution_time': self.max_execution_time,
            'rClose': self.rClose
        }

        np.savez(param_file, **params)


    def _save_result(self, v, result):
        i = result[0]

        if i == 0:
            self._save_set_parameters()

        result_keys = ['i', 'v_inf', 'input_info', 'errors', 'a_init', 'e_init', 'E_init', 'start_distances', 'lifetime', 'integrations', 'final_energy_c', 'termination_flag', 'eccentricities', 'semi_major_axes', 'orbital_periods', 'times', 'collision_info', 'script_version']
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
        dir_v = os.path.join(save_dir, f"v{v_str}")
        os.makedirs(dir_v, exist_ok=True)
        fig_name = f"sim_{i}_t{sim_t:.0f}_{flag}_s{self.seed_base}.png"
        fig_path = os.path.join(dir_v, fig_name)
        op.fig.savefig(fig_path)
       
        op.fig.clf()

    # def _track_preprocess_errors(self, error):
    #     if self.error_list is None:
    #         self.error_list = []
    #     self.error_list.append(error)

    # def get_error_list(self):
    #     if self.error_list:
    #         return self.error_list
    #     else:
    #         return []

    def _print_script_version(self):
        version = get_script_version()
        print(f"Running Rebound_v3.py, last modified on {version}")
    

    def run_orbital_integration(self, i, v_inf, lambda1, beta, phi, b, pos_C, v_C, pos_B, v_B):

        # if check_exists:
        #     if self._check_result_exists(v_inf / 1e3, i):
        #         print(f"Simulation {i} for v_inf {v_inf/1e3} km/s already exists. Skipping...")
        #         return
        input_info = {'i': i, 'v_inf': v_inf, 'lambda1': lambda1, 'beta': beta, 'phi': phi, 'b': b, 'pos_C': pos_C, 'v_C': v_C, 'pos_B': pos_B, 'v_B': v_B}
        
        # print(f"Simulation {i} of starting at {datetime.datetime.now()}")

        mA = self.mA  # Msun
        mB = self.mB  # Jupiter mass in Msun
        mC = self.mC  # PBH mass in Msun

        # Radii in AU
        rA = self.rA 
        rB = self.rB
        rC = self.rC # PBH radius in AU for 1e-13 Msun

        # Velocity at infinity in AU/yr
        v_inf_kms = v_inf / 1e3  # km/s
        v_inf = (v_inf_kms * u.km/u.s).to(u.au/u.yr).value
        pos_C = (np.array(pos_C) * u.m).to(u.au).value
        v_C = (np.array(v_C) * u.m/u.s).to(u.au/u.yr).value
        pos_B = (np.array(pos_B) * u.m).to(u.au).value
        v_B = (np.array(v_B) * u.m/u.s).to(u.au/u.yr).value
        # Gravitational parameter in these units: G = 4*pi^2 AU^3/(Msun*yr^2)
        sim = rebound.Simulation()
        sim.units = ('AU', 'yr', 'Msun')
        sim.integrator = "MERCURIUS"
        sim.add(m=mA, r=rA, x=0, y=0, z=0)  # Sun
        sim.add(m=mB, r=rB, x=pos_B[0], y=pos_B[1], z=pos_B[2], vx=v_B[0], vy=v_B[1], vz=v_B[2])  # Jupiter
        sim.add(m=mC, r=rC, x=pos_C[0], y=pos_C[1], z=pos_C[2], vx=v_C[0], vy=v_C[1], vz=v_C[2])  # PBH

        G_unit = sim.G  # 4*pi^2 in these units
        sim.move_to_com()

        a_init = sim.particles[2].orbit(primary=sim.particles[0]).a
        e_init = sim.particles[2].orbit(primary=sim.particles[0]).e
        E_init = sim.energy()

        error_list = []
        if E_init >= 0:
            error_list.append('system not bound at start')

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
        t_max_execution = self.max_execution_time  # seconds 
        # aC_max = aB * 40

        eccentricities = []
        semi_major_axes = []
        orbital_periods = []
        times = []
        E_c_cond, flag = None, None
        max_steps = int(np.ceil(t_end / sim.dt))
        sim.collision = "direct"
        sim.collision_resolve = "halt"  # Stop integration on collision
        
        # # Variable to store collision info
        collision_info = None

        # start_info = (f'dt: {sim.dt}, snapshot_rate: {snap_rate}, snap_interval: {snap_interval}, max steps: {max_steps}, '
        #     f't_end: {t_end}, t_min: {t_min}, t_max_execution: {t_max_execution}, orbital period C: {P_C}, orbital period B: {P_B}, '
        #     ')

        # try:
        if not np.isfinite(P_C) or P_C <= 0:
            error_list.append('invalid orbital period C')
            # self._track_preprocess_errors('invalid orbital period C')
            # flag = 'preprocessing_error'
            # result = [i, v_inf, start_info, error_list, start_separation, sim.t, E_c_cond, flag, eccentricities, semi_major_axes, times]
            # self._save_result(v_inf_kms, result)
            # raise exc.InvalidPeriodError(f"Invalid orbital period for C: {P_C}")

        if not np.isfinite(P_B) or P_B <= 0:
            error_list.append('invalid orbital period B')

            # self._track_preprocess_errors('invalid orbital period B')
            # flag = 'preprocessing_error'
            # result = [i, v_inf, start_info, error_list, start_separation, sim.t, None, None, None, None, None]
            # self._save_result(v_inf_kms, result)
            # raise exc.InvalidPeriodError(f"Invalid orbital period for B: {P_B}")

        if not np.isfinite(sim.dt) or sim.dt <= 0:
            # self._track_preprocess_errors('invalid time step')
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
        plot_counter = 0
        flags = set()
        E_initial = sim.energy()
        int_start = datetime.datetime.now()
        try:
            # if i<10:
            #     ops = rebound.OrbitPlotSet(sim, slices=True, unitlabel="[AU]", color=["black", "red"])
            #     self._save_figure(ops, i, v_inf_kms, 'initial', sim.t)
            j = 0
            while (sim.t < t_end):
                P_C = sim.particles[2].orbit(primary=sim.particles[0]).P
                P_B = sim.particles[1].orbit(primary=sim.particles[0]).P
                if (sim.dt > np.min([abs(P_C), abs(P_B)]) * 0.05) or (sim.dt < np.min([abs(P_C), abs(P_B)]) * 0.04):
                    sim.dt = np.min([abs(P_C), abs(P_B)]) * 0.05
                    # print(f"Adjusted time step: {sim.dt}")
                j += booster
                counter += booster
                time = sim.t + sim.dt * booster
                sim.integrate(time, exact_finish_time=0)

                # Specific orbital energy of C around A (AU^2/yr^2)
                Evc = 0.5 * (sim.particles[2].vx**2 + sim.particles[2].vy**2 + sim.particles[2].vz**2)
                rAC = np.abs(sim.particles[2] ** sim.particles[0])
                E_c_cond = Evc - (G_unit*(sim.particles[0].m + sim.particles[2].m) / rAC)

                # Specific orbital energy of B around A (AU^2/yr^2)
                Evb = 0.5 * (sim.particles[1].vx**2 + sim.particles[1].vy**2 + sim.particles[1].vz**2)
                rAB = np.abs(sim.particles[1] ** sim.particles[0])
                E_b_cond = Evb - (G_unit*(sim.particles[0].m + sim.particles[1].m) / rAB)

                if (E_c_cond > 0):
                    flags.add('escape_C')
                    raise exc.EscapeError(f"Particle C is free: E_c:  {E_c_cond}, rAC: {np.abs(sim.particles[2] ** sim.particles[0])}")
                if (E_b_cond > 0):
                    # error_list.append('escape_B')
                    flags.add('escape_B')
                    # raise exc.EscapeError(f"Particle B is free: E_b: {E_b_cond}, rAB: {np.abs(sim.particles[1] ** sim.particles[0])}")

                if counter >= snap_rate:
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
                    #     ops.fig.savefig(f'{directoryp}Orbits_Simulation_{i}_step_{j}.png')
                    #     plot_counter = 0

                    E_current = sim.energy()
                    error = abs(E_current - E_initial)/E_initial
                    if error > 1e-5:
                        flags.add('energy_conservation')
                        raise exc.EnergyError(f"Error in energy conservation: {error}")
                    
                    int_current = datetime.datetime.now()
                    elapsed_int = (int_current - int_start).total_seconds()
                    if elapsed_int > t_max_execution:
                        flags.add('time_exceeded')
                        raise exc.MaxIntegrationTimeError(f"Maximum time for integration exceeded: {elapsed_int}, evolution stopped at time {sim.t} years, rAC: {np.abs(sim.particles[2] ** sim.particles[0])}, rAB: {np.abs(sim.particles[1] ** sim.particles[0])}, E_c_cond: {E_c_cond}, E_b_cond: {E_b_cond}")

                    dist_BC = np.abs(sim.particles[1] ** sim.particles[2])
                    dist_AB = np.abs(sim.particles[0] ** sim.particles[1])
                    dist_AC = np.abs(sim.particles[0] ** sim.particles[2])

                    if dist_BC < (rB + rC) or dist_AB < (rA + rB) or dist_AC < (rA + rC):
                        # Handle collision
                        flags.add('collision_manual')
                        raise exc.CollisionManualError(f"Collision detected: BC {dist_BC}, AB {dist_AB}, AC {dist_AC} at time {sim.t}")
                
                # plot_counter += 1
                # if yr == int(t_min):
                #     print(f"Simulation {i} at time {sim.t}, step {j}")
                #     print(f'Eccentricity of C : {sim.particles[2].e}')
                #     print(f'PBH E_cond: {E_c_cond}, Jupiter E_cond {E_b_cond} rCB {sim.particles[1]  ** sim.particles[2]}, rAC: {rAC}, error {error}')
                #     yr = 0
        except rebound.Collision as e:
            # Extract collision information from Rebound
            flags.add('collision')
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

        except (exc.EnergyError, exc.EscapeError, exc.MaxIntegrationTimeError, exc.CollisionManualError) as e:
            print(f"Error during integration: {e}, Simulation failed for system {i}, at step {j}. Script continues...")

        except rebound.OrbitPlotSetError as e:
            print(f"Plotting error during integration: {e}, Simulation for system {i} at step {j}. Continuing without plotting...")
        # E_final = sim.energy()
        # energy_change = abs(E_final - E_initial)/E_initial
        # print(f'final energy {E_final}, final time {sim.t}, final distance {sim.particles[1]  ** sim.particles[2]}, energy change {energy_change}')

        # print(f'r_close = {rclose} and initial BC separation =  {initial_BC_distance}')
        if sim.t >= t_end:
            flags.add('completed')
            print(f"Simulation {i}  completed at {datetime.datetime.now()}")

        flag = '_'.join(sorted(flags)) if flags else 'none'
        # try:
        #     if i<10:
        #         ops = rebound.OrbitPlotSet(sim, slices=True, unitlabel="[AU]", color=["black", "red"])
        #         self._save_figure(ops, i, v_inf_kms, f'final_{flag}_{j}', sim.t)
        # except rebound.OrbitPlotSetError as e:
        #     print(f"Final plotting error: {e}, Simulation for system {i}. Continuing without plotting...")
        #     pass


        result = [i, v_inf, input_info, error_list, a_init, e_init, E_init, start_separation, sim.t, j, E_c_cond, flag,
                  np.array(eccentricities, dtype=float), np.array(semi_major_axes, dtype=float),
                  np.array(orbital_periods, dtype=float), np.array(times, dtype=float), collision_info, get_script_version()]
        self._save_result(v_inf_kms, result)

        