# Sun-Jupiter- system, fixed mC = 1e-7
# lambda1 in interval [0, 2pi] beta in interval [-pi/2, pi/2] uniform random sampling
# Multiprocessing
import rebound
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import datetime
from astropy.table import Table
import logging
import sys
import pandas as pd
from astropy import constants as const
from astropy import units as u  
import multiprocessing
import shutil
from mpl_toolkits.mplot3d import Axes3D


s = datetime.datetime.now()
# Displays Time
current_time = s.strftime('%H%M')
print("current time :", current_time)
if not os.path.exists(f'/data/a.saricaoglu/repo/COMPAS/Files/Capture/{str(s.strftime("%m.%d"))}/{current_time}/LBImpB_Variation/'): 
    os.makedirs(f'/data/a.saricaoglu/repo/COMPAS/Files/Capture/{str(s.strftime("%m.%d"))}/{current_time}/LBImpB_Variation/') 
directoryf = f'/data/a.saricaoglu/repo/COMPAS/Files/Capture/{str(s.strftime("%m.%d"))}/{current_time}/LBImpB_Variation/' 
if not os.path.exists(f'{directoryf}/discarded/'):
    os.makedirs(f'{directoryf}/discarded/')
directoryf_discarded = f'{directoryf}/discarded/'
if not os.path.exists(f'/data/a.saricaoglu/repo/COMPAS/Plots/Capture/{str(s.strftime("%m.%d"))}/{current_time}/'): 
    os.makedirs(f'/data/a.saricaoglu/repo/COMPAS/Plots/Capture/{str(s.strftime("%m.%d"))}/{current_time}/') 
directoryp = f'/data/a.saricaoglu/repo/COMPAS/Plots/Capture/{str(s.strftime("%m.%d"))}/{current_time}/' 



# Redirect stdout and stderr to the log file
class StreamToLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

class EnergyError(Exception): pass
class EscapeError(Exception): pass
class CollisionError(Exception): pass
class RangeError(Exception): pass
class MaxYearForBError(Exception): pass

sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

import time
SA = []
def run_simulation(pars):
    global SA
    firstCaptureA = False
    i, beta, lambda1, b = pars
    np.random.seed(os.getpid() + int(time.time() * 1000000) % 100000)
    # Get the script name
    script_name = os.path.basename(__file__)
    # Configure logging
    log_filename = f"{directoryf}{script_name}_{current_time}/{i}.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(filename=log_filename, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    print(f"Simulation {i} of {numSim} starting at {datetime.datetime.now()}")

    #print(sim.G)
    epsilon = 0.1 
    # lambda1 = np.random.uniform(0, 2*np.pi)  # Random angle in radians
    # beta = np.random.uniform(-np.pi/2, np.pi/2)  # Random angle in radians
    e_b = 0.0489 # Eccentricity

    # mA = 1 # mass of A in Msun
    # mB = 0.0009543  # mass of B (jupiter) in Msun
    # mC = 1e-13 # mass of C in Msun
    # rA = 1 * (const.R_sun)  # radius of A in AU
    # rA = rA.to(u.au).value
    # rB = 0.000477895  * (const.R_sun)   # radius of B in AU
    # rB = rB.to(u.au).value
    # aB = 5.2    # semi-major axis of A-B in AU
    # mA = 2.01 # mass of A in Msun
    # mB = 0.172  # mass of B (jupiter) in Msun
    # mC = 1e-13 # mass of C in Msun
    # rA = 1.87*1e-5 * (const.R_sun)  # radius of A in AU
    # rA = rA.to(u.au).value
    # rB = 0.065  * (const.R_sun)   # radius of B in AU
    # rB = rB.to(u.au).value
    # aB = (832000 *u.km ).to(u.au).value  # semi-major axis of A-B in AU
    # v_inf = 4.2161 # Velocity at infinity in AU/yr, 20 km/s

    mA = 1.4 # mass of A in Msun
    mB = 1.2912e-5 # mass of B (jupiter) in Msun
    mC = 1e-13 # mass of C in Msun
    rA = 1.5e-5  * (const.R_sun)  # radius of A in AU
    rA = rA.to(u.au).value
    rB = 0.01751077 * (const.R_sun)   # radius of B in AU
    rB = rB.to(u.au).value
    aB = 0.36 # semi-major axis of A-B in AU
    v_inf = 4.2161/2.5# Velocity at infinity in AU/yr, 20 km/s

    print(f"mA: {mA}, mB: {mB}, rA: {rA}, rB: {rB}, aB: {aB}, v_inf: {v_inf}")


    mc = "{:.1E}".format(mC)
    # Set Mercurius as the integrator
    # sim.ri_mercurius.hillfac = 3  # Adjust factor for close encounters
    # sim.ri_mercurius.r_hill = ep # Hill radius factor for close encounters
    # Add primary body (A) with higher mass

    # Create a new simulation
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.integrator = "MERCURIUS"
    sim.add(m=mA)  # Label as "A"

    # Add secondary body (B) in orbit around A
    sim.add(m=mB, a=aB, e=e_b)
    opB = sim.particles[1].P  # Orbital period of B
    #print(sim.particles[1].vxyz)
    r_close =  aB * (mB*epsilon / mA)**(1/3)  

   
    v1 = np.sqrt(v_inf**2 + 2*sim.G*mA/aB + 2*sim.G*mB/r_close)

    # r_close_x = np.random.uniform(-1, 1) * r_close
    # r_close_y = np.random.uniform(-np.sqrt(r_close**2 - r_close_x**2), np.sqrt(r_close**2 - r_close_x**2))
    # r_close_z = np.sqrt(r_close**2 - r_close_x**2 - r_close_y**2)
    # r_close_x = r_close * np.cos(beta) * np.sin(lambda1)
    # r_close_y = r_close *  np.cos(beta) * np.cos(lambda1)
    # r_close_z = r_close * np.sin(beta)

    vz_c =  np.sin(beta)
    vx_c =  np.cos(beta) * np.sin(lambda1)
    vy_c =  np.cos(beta) * np.cos(lambda1)
    v = np.array([vx_c, vy_c, vz_c])
    v_normalised = v / np.linalg.norm(v) 
    v = v_normalised * v1  # Scale to the desired velocity

    v1prime = np.sqrt(np.abs((sim.particles[1].vx - v[0])**2 + (sim.particles[1].vy- v[1])**2 + (sim.particles[1].vz- v[2])**2))
    # print('v1prime', v1prime)
    bmax = r_close
    bmin = (1/v1prime) * np.sqrt(2*sim.G*mB*rB + (rB*v1prime)**2)  # Minimum impact parameter for capture

    if (b < bmin) or (b > bmax) or (bmax < bmin):
        print(f"Error in impact parameter: {b}")
        return "Skipping to next system..."
    # b = np.random.uniform(bmin, r_close)   # Initial distance of C from A-B in AU

    r_c = -v_normalised * r_close

     # Normalize the velocity vector
    # Choose an arbitrary vector that is not parallel to v
    if v[0] == 0 and v[1] == 0:  # If v is parallel to [0, 0, 1]
        arbitrary_vector = np.array([0, 1, 0])
    else:
        arbitrary_vector = np.array([0, 0, 1])

    # Compute the cross product to get an orthogonal vector
    orthogonal_vector = np.cross(v, arbitrary_vector)

    # Normalize the orthogonal vector
    orthogonal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)

    # Scale the vector to have magnitude r_close
    result_vector = orthogonal_vector * b

    print("Orthogonal vector:", result_vector)
    offset_x = result_vector[0] + r_c[0]
    offset_y = result_vector[1] + r_c[1]
    offset_z = result_vector[2] + r_c[2]
    # Add third body (C) with a hyperbolic trajectory
    # sim.add(m=mC, x=r_close_x + a_b*np.cos(lambda1), y=-r_close_y + a_b*np.sin(lambda1),  z=r_close_z ,  vx=vx_c, vy=vy_c, vz = vz_c) 
    sim.add(m=mC, x=sim.particles[1].x + offset_x, y=sim.particles[1].y + offset_y,  z=sim.particles[1].z + offset_z ,  vx=v[0], vy=v[1], vz = v[2])
    print(f'aC: {sim.particles[2].a} vs rAC: {sim.particles[0]  ** sim.particles[2]}')  
    print(f'aB: {sim.particles[1].a} vs rAB: {sim.particles[0]  ** sim.particles[1]}')
    #print(sim.particles[2].vxyz)
    v1prime2 = np.sqrt(np.abs((sim.particles[1].vx- sim.particles[2].vx)**2 + (sim.particles[1].vy- sim.particles[2].vy)**2 + (sim.particles[1].vz- sim.particles[2].vz)**2))

    # print('vprime2', v1prime2)
    # bmin = 1/v1primme * np.sqrt(2*sim.G*mB*rB + (rB*v1primme)**2) 
    print('bmin', bmin, 'bmax', bmax)
    # sim.add(m=0, a=100)  # Add a distant body to make Spock work
    initial_BC_distance = sim.particles[1] ** sim.particles[2]
    # Integration parameters
    sim.dt = sim.particles[1].P * 0.05  # timestep is 5% of orbital period
    snap_rate = 30  # snapshot every 10% of the time
    snapshot_interval = sim.dt * snap_rate  # Number of steps to integrate
    t_end = int(1e7)  # Max integration time in years
    t_min = int(1e3) * sim.particles[1].P  # Minimum time survival for recording the simulation
    t_max = int(1e6) * sim.particles[1].P # Maximum time for recording the simulation
    aC_max = aB * 40 # 40 times the semi-major axis of B
    # N_steps = int(t_end / sim.dt)  # Number of steps to integrate
    # N_steps = 100 * sim.particles[1].P # 100 orbital period duration
    sim.collision = "direct"  # Turn on collisions#
    print(f'dt: {sim.dt}, snapshot_rate: {snap_rate}, snapshot_int: {snapshot_interval}, t_end(Max intgr time in years): {t_end}, t_min(1e3*opB): {t_min}, t_max(1e6*opB): {t_max}')
    print(f'r_close {r_close}, vC wrt vB: {v1prime2},v1 wrt vB: {v1prime},b: {b} bmin: {bmin}, bmax: {bmax}, hill radius: {sim.ri_mercurius.r_crit_hill}')
    print(f'orbital period (of B) = {opB}, total years of integration = {t_end}, total years for B = {t_end/sim.particles[1].P}')
    print(f"vB: {sim.particles[1].vxyz}, vBx: {sim.particles[1].vx}, vC: {sim.particles[2].vxyz}, vA: {sim.particles[0].vxyz}")
    collided = []

    counter = 0

    # sim.heartbeat = heartbeat
    E_initial = sim.energy()
    print(f'time zero {sim.t}, initial energy {E_initial}')
    sim.save_to_file(f"{directoryf}{script_name}_{current_time}/sim_{i}_{mc}.bin",interval=snapshot_interval, delete_file=True)
    
    step = 1  
    capture = 0
    yr = 0
    try:
        for j in range(0, int(t_end/step)):
            sim.integrate(sim.t+step)
            E_current = sim.energy()
            error = abs(E_current - E_initial)/E_initial
            Evc = 0.5 * (sim.particles[2].vx**2 + sim.particles[2].vy**2 + sim.particles[2].vz**2)
            E_cond = Evc - (sim.G*sim.particles[0].m / np.abs(sim.particles[2] ** sim.particles[0]))

            if firstCaptureA != True:
                if E_cond < 0:
                    capture += 1
                else:
                    capture = 0

                if capture > 10:
                    SA.append([i, sim.particles[2].a])
                    firstCaptureA = True

            if error > 1e-5:
                raise EnergyError(f"Error in energy conservation: {error}")
            
            if (np.abs(sim.particles[2] ** sim.particles[0]) > aC_max):
                if (E_cond > 0):
                    print(f'Energy condition: {E_cond}')
                    raise EscapeError(f"Particle C is too far away and free: {np.abs(sim.particles[2] ** sim.particles[0])}")
            if sim.t > t_max:
                print(f"Simulation {i} {mc} ended at time {sim.t}, step {j}, loop break")
                break
            
            for p in sim.particles:
                if p.last_collision == sim.t:
                    raise CollisionError(f"Collision detected at time {sim.t}")
                
            yr += step
            if yr == int(t_min):
                print(f"Simulation {i} {mc} at time {sim.t}, step {j}")
                print(f'Eccentricity of C : {sim.particles[2].e}')
                print(f'E_cond: {E_cond}, rCB distance {sim.particles[1]  ** sim.particles[2]}, rAC: {np.abs(sim.particles[2] ** sim.particles[0])}, error {error}')
                yr = 0
    except (EnergyError, EscapeError, CollisionError) as e:
        print(f"Error during integration: {e}")
        print(f"Simulation failed for system {i} {mc}, at time {sim.t}")
        print("Script continues...")
        if sim.t < t_min:
            print(f"Time {sim.t} < t_min {t_min}")
            print(f"Checking if file exists: {directoryf}sim_{i}_{mc}.bin")
            if not os.path.exists(f"{directoryf}sim_{i}_{mc}.bin"):
                print(f"File not found: {directoryf}sim_{i}_{mc}.bin")
            os.remove(f"{directoryf}sim_{i}_{mc}.bin")
            print(f"File {directoryf}sim_{i}_{mc}.bin has been deleted.")
            destination_file = f"{directoryf_discarded}{script_name}_{i}_script.log"
            shutil.move(log_filename, destination_file)    
            print(f"Log file {log_filename} has been moved to {directoryf_discarded}.")       
            return "Skipping to next system..."
        else:
            pass
        


    print(f"Simulation {i} {mc} ended at {sim.t}")
    E_final = sim.energy()
    energy_change = abs(E_final - E_initial)/E_initial
    print(f'final energy {E_final}, final time {sim.t}, final distance {sim.particles[1]  ** sim.particles[2]}, energy change {energy_change}')

    print(f'r_close = {r_close} and initial BC separation =  {initial_BC_distance}')

    fit_filename = f"{directoryf}{script_name}_{current_time}/rebound_{i}_{mc}.fits"
    hdu_pr = fits.PrimaryHDU()
    hdu_pr.writeto(fit_filename, overwrite=True)
    hdu = fits.open(fit_filename, mode='update')

    hdul_status = fits.BinTableHDU()
    hdul_status.header['rClose'] = "{:.3f}".format(r_close)
    hdul_status.header['vInf'] = (v_inf)
    hdul_status.header['vPrm1'] = (v1prime)
    hdul_status.header['vPrm2'] = (v1prime2)
    hdul_status.header['bmin'] = (bmin)
    hdul_status.header['bmax'] = (bmax)
    hdul_status.header['b'] = (b)
    hdul_status.header['beta'] = (beta)  
    hdul_status.header['lambda1'] = (lambda1) 
    hdul_status.header['inBCdist'] = (initial_BC_distance)
    hdul_status.header['deltaE'] = (energy_change)
    hdul_status.header['colls'] = len(collided)
    hdul_status.header['mA'] = (mA)
    hdul_status.header['mB'] = (mB)
    hdul_status.header['mC'] = (mC)
    hdul_status.header['rB'] = (rB)
    hdul_status.header['rA'] = (rA)
    hdul_status.header['aB'] = (aB)
    hdul_status.header['opB'] = "{:.3E}".format(opB)
    hdul_status.header['totSimYr'] = (t_end)
    hdul_status.header['fSimYr'] = (sim.t)
    hdul_status.header['fBYr'] = (sim.t/sim.particles[1].P)
    hdul_status.header['snpInt'] = (snapshot_interval)
    hdul_status.header['dt'] = "{:.3E}".format(sim.dt)
    hdul_status.header['snpRt'] = (snap_rate)

    hdu.append(hdul_status)
    hdu.close()

    # sa = rebound.Simulationarchive(f"{directoryf}sim_{i}_{mc}.bin")
    # print(len(sa))
    return f"Simulation {i+1} completed"

# Number of simulations
numSim = 120

if __name__ == "__main__":


    # Create a pool of workers
    num_cores = 60  # Get the number of available CPU cores
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Distribute the simulations across the cores
        pars = []
        i = 0
        for beta in np.linspace(-np.pi/2, np.pi/2, numSim):
            for lamda in np.linspace(0, 2*np.pi, numSim):
                for b in np.linspace(0.00008146,0.0035, numSim): #bmin set to radius of B, B max set to r_close | PSR J0348+0432 bmin, bmax:0.0003, 0.00113  bmin.max : 2.22e-06 0.237 for jupiter sun 
                    pars.append((i, beta, lamda, b))
                    i += 1
        results = pool.map(run_simulation, pars)
    pool.close()
    pool.join()
    script_name = os.path.basename(__file__)
    # Configure logging
    log_filename = f"{directoryf}{script_name}_{current_time}/general_script.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(filename=log_filename, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    # Print results
    for result in results:
        print(result)
    
    np.savetxt(f"{directoryf}SA.txt", SA, fmt="%d", delimiter=",")


    print(f'Execution time: {datetime.datetime.now() - s} for {numSim} simulations with {1e7} years of integration')
