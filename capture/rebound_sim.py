
# Sun-Jupiter- system, fixed mC = 1e-7
# lambda1 in interval [0, 2pi] beta in interval [-pi/2, pi/2] uniform random sampling

import rebound
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import datetime
from astropy.table import Table
import logging
import sys
from mpl_toolkits.mplot3d import Axes3D
import shutil
import astropy.units as u
import astropy.constants as const
s = datetime.datetime.now()
# Displays Time
current_time = s.strftime('%H%M')
print("current time :", current_time)
if not os.path.exists(f'/data/a.saricaoglu/repo/COMPAS/Files/Capture/{str(s.strftime("%m.%d"))}/{current_time}/'): 
    os.makedirs(f'/data/a.saricaoglu/repo/COMPAS/Files/Capture/{str(s.strftime("%m.%d"))}/{current_time}/') 
directoryf = f'/data/a.saricaoglu/repo/COMPAS/Files/Capture/{str(s.strftime("%m.%d"))}/{current_time}/' 
if not os.path.exists(f'/data/a.saricaoglu/repo/COMPAS/Plots/Capture/{str(s.strftime("%m.%d"))}/{current_time}/'): 
    os.makedirs(f'/data/a.saricaoglu/repo/COMPAS/Plots/Capture/{str(s.strftime("%m.%d"))}/{current_time}/') 
directoryp = f'/data/a.saricaoglu/repo/COMPAS/Plots/Capture/{str(s.strftime("%m.%d"))}/{current_time}/' 

# Get the script name
script_name = os.path.basename(__file__)
# Configure logging
log_filename = f"{directoryf}{script_name}_script.log"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

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

sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)



def status_logger(Si):
    global S

    S = np.vstack((S, Si))

def heartbeat(sim_pointer):
    global A
    global B
    global C
    global T
    global L
    global Si
    global E_initial
    sim = sim_pointer.contents



    Evc = 0.5 * (sim.particles[2].vx**2 + sim.particles[2].vy**2 + sim.particles[2].vz**2)
    E_cond = Evc - (sim.G*sim.particles[0].m / np.abs(sim.particles[0] ** sim.particles[2]))

    v1prime = np.sqrt(np.abs((sim.particles[1].vx- sim.particles[2].vx)**2 + (sim.particles[1].vy- sim.particles[2].vy)**2 + (sim.particles[1].vz- sim.particles[2].vz)**2))
    Evcprime = 0.5 * v1prime**2
    E_con_bound_b = Evcprime - (sim.G*sim.particles[1].m / np.abs(sim.particles[1] ** sim.particles[2])) 
    # print(E_cond, E_con_bound_b)

    t = int(sim.t)

    Si[0] = t

    if (E_cond > 0):
        if (E_con_bound_b >= 0):
            L.append(f"Free! at time {sim.t}")
            Si[1] = 1
        else:
            L.append(f"Bound to B at time {sim.t}")
            Si[2] = 1    
    else:
        if (E_con_bound_b >= 0):
            L.append(f"Capture (not bound to B ) at time {sim.t}")
            Si[3] = 1
        else:
            L.append(f"Capture (still bound to B ) at time {sim.t}")
            Si[4] = 1

    status_logger(Si)
    Si = np.zeros(5)
    A.append(sim.particles[0].xyz)
    B.append(sim.particles[1].xyz)
    C.append(sim.particles[2].xyz)
    T.append(sim.t)


numSim = 10000
for i in range(numSim):
    # Create a new simulation
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    print(sim.G)
    epsilon = 0.1 
    lambda1 = np.random.uniform(0, 2*np.pi)  # Random angle in radians
    beta = np.random.uniform(-np.pi/2, np.pi/2)  # Random angle in radians

    mA = 1 # mass of A in Msun
    mB = 0.0009543  # mass of B (jupiter) in Msun
    mC = 1e-13 # mass of C in Msun
    rA = 1 * (const.R_sun)  # radius of A in AU
    rA = rA.to(u.au).value
    rB = 0.000477895  * (const.R_sun)   # radius of B in AU
    rB = rB.to(u.au).value
    aB = 5.2   # semi-major axis of A-B in AU
    mc = '{:.3f}'.format(mC)
    # Set Mercurius as the integrator
    sim.integrator = "MERCURIUS"

    mA = 1.4 # mass of A in Msun
    mB = 1.2912e-5 # mass of B (jupiter) in Msun
    mC = 1e-13 # mass of C in Msun
    rA = 1.5e-5  * (const.R_sun)  # radius of A in AU
    rA = rA.to(u.au).value
    rB = 0.01751077 * (const.R_sun)   # radius of B in AU
    rB = rB.to(u.au).value
    aB = 0.36 # semi-major axis of A-B in AU
    v_inf = 4.2161/2.5# Velocity at infinity in AU/yr, 20 km/s
    # sim.ri_mercurius.hillfac = 3  # Adjust factor for close encounters
    # sim.ri_mercurius.r_hill = ep # Hill radius factor for close encounters
    # Add primary body (A) with higher mass
    sim.add(m=mA)  # Label as "A"

    # Add secondary body (B) in orbit around A
    a_b = 5.2  # Semi-major axis of A-B system in AU (Sun-Jupiter distance)
    e_b = 0.0489 # Eccentricity
    sim.add(m=mB, a=a_b, e=e_b)
    #print(sim.particles[1].vxyz)
    r_close =  a_b * (mB*epsilon / mA)**(1/3)  

    v_inf = 0.2108  # Velocity at infinity in AU/yr, 20 km/s
    v1 = np.sqrt(v_inf**2 + 2*sim.G*mA/a_b + 2*sim.G*mB/r_close)

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
    b = np.random.uniform(bmin, r_close)   # Initial distance of C from A-B in AU

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
    initial_BC_distance = sim.particles[1]  ** sim.particles[2]
    # Integration parameters
    sim.dt = sim.particles[1].P * 0.05  # timestep is 5% of orbital period
    print('dt ', sim.dt)
    t_end = int(1)  # Total integration time in years
    # N_steps = int(t_end / sim.dt)  # Number of steps to integrate
    # N_steps = 100 * sim.particles[1].P # 100 orbital period duration
    sim.collision = "direct"  # Turn on collisions

    # Store data for visualization
    A = []
    B = []
    C = []
    T = []
    L = []
    collided = []
    Si = np.zeros(5)
    S = np.empty((0,5))
    #print(np.shape(S))
    counter = 0

    sim.heartbeat = heartbeat
    E_initial = sim.energy()
    print('time zero', sim.t, ' initial energy', E_initial)
    sim.save_to_file(directoryf +f"/sim_{i}_{mc}.bin",interval=sim.dt, delete_file=True)
    
    step = 1  
    yr  = 0
    try:
        for j in range(0, int(t_end/step)):
            sim.integrate(sim.t+step)
            E_current = sim.energy()
            error = abs(E_current - E_initial)/E_initial
            Evc = 0.5 * (sim.particles[2].vx**2 + sim.particles[2].vy**2 + sim.particles[2].vz**2)
            E_cond = Evc - (sim.G*sim.particles[0].m / np.abs(sim.particles[2] ** sim.particles[0]))

            if error > 1e-5:
                raise EnergyError(f"Error in energy conservation: {error}")
            
            if (np.abs(sim.particles[2] ** sim.particles[0]) > 200):
                if (E_cond > 0):
                    print(f'Energy condition: {E_cond}')
                    raise EscapeError(f"Particle C is too far away and free: {np.abs(sim.particles[2] ** sim.particles[0])}")
            
            for p in sim.particles:
                if p.last_collision == sim.t:
                    raise CollisionError(f"Collision detected at time {sim.t}")
                
            yr += step
            if yr == 100:
                print(f"Simulation {i} {mc} at time {sim.t}, step {j}")
                print(f'Eccentricity of C: {sim.particles[2].e}')
                print(f'E_cond: {E_cond}, rCB distance {sim.particles[1]  ** sim.particles[2]}, rAC: {np.abs(sim.particles[2] ** sim.particles[0])}, error {error}')
                yr = 0
    except (EnergyError, EscapeError, CollisionError) as e:
        print(f"Error during integration: {e}")
        print(f"Simulation failed for system {i} {mc}, at time {sim.t}")
        if sim.t < 1e3:
            os.remove(directoryf +f"sim_{i}_{mc}.bin")
            print(f"File {directoryf +f"sim_{i}_{mc}.bin"} has been deleted.")      
        continue
    
    E_final = sim.energy()
    print('final time', sim.t, ' final energy', E_final)
    energy_change = abs(E_final - E_initial)/E_initial
    #print(sim.particles[0].xyz)
    print(counter)
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    body_names = ["A", "B", "C"]
    marker = ["*", ".", "v"]
    n = 0
    for body in [A, B, C]:
        x, y, z = zip(*body[::100])
        #print(len(x))
        t = np.linspace(0, t_end, len(x)) 
        ax.scatter(x, y, z, c=t,cmap='viridis', marker=marker[n], s=100)  # Small dots for trails
        if n == 1:
            plt.quiver(x[0], y[0], z[0], result_vector[0], result_vector[1], result_vector[2],     # Components of the vector
            color='red', label='b',linewidth=2, arrow_length_ratio=0.1)
            plt.quiver(x[0], y[0], z[0], offset_x, offset_y, offset_z, 
            color='green', label='o',linewidth=2, arrow_length_ratio=0.1)     # Components of the vector
        if n == 2:
            plt.quiver(x[0], y[0], z[0], v_normalised[0], v_normalised[1], v_normalised[2] ,     # Components of the vector
            color='blue', label='v', linewidth=2, arrow_length_ratio=0.1)
        n += 1
        
    # Set axis labels
    ax.set_xlabel("X (AU)")
    ax.set_ylabel("Y (AU)")
    ax.set_zlabel("Z (AU)")
    # ax.legend()
    # Set limits for x, y, and z dimensions
    # ax.set_xlim([-50, 50])  # Replace with your desired x-axis limits
    # ax.set_ylim([-50, 50])  # Replace with your desired y-axis limits
    # ax.set_zlim([-50, 50])  # Replace with your desired z-axis limits

    # plt.title(f'$M_A$ = {"{:.2E}".format(mA)}, $M_B$= {"{:.2E}".format(mB)}, $M_C$= {"{:.2E}".format(mC)}, $v_i$= {"{:.2E}".format(v_inf)}, $r_c$= {"{:.2E}".format(r_close)}')
    plt.savefig(directoryp + f'capture_{i}.png', dpi=200)
    # plt.show()
    # print(L)
    #print(S)
    print(f'r_close = {r_close} and initial BC separation =  {initial_BC_distance}')

    #print(len([s for s in S if  s[0]==0]))
    #print([s for s in S if  s[0]==0])
    #print(S[25:39])
    print(np.sum(S[:,1]), np.sum(S[:,2]), np.sum(S[:,3]), np.sum(S[:,4]))
    print(np.sum(S[:,1]) + np.sum(S[:,2]) + np.sum(S[:,3]) + np.sum(S[:,4]))
    #print(sim.particles[0])
    #print(sim.orbits())

    print('s shape', np.shape(S))
    # fit_filename = f"{directoryf}/rebound_{i}.fits"
    # hdu_pr = fits.PrimaryHDU()
    # hdu_pr.writeto(fit_filename, overwrite=True)
    # hdu = fits.open(fit_filename, mode='update')

    # hdul_status = fits.BinTableHDU(Table(data=S, names=('Time','Free','Bound_B','Capture_Not_Bound_B','Capture_Bound_B')))
    # hdul_status.header['r_close'] = (r_close)
    # hdul_status.header['v_inf'] = (v_inf)
    # hdul_status.header['v1primme'] = (v1primme)
    # hdul_status.header['bmin'] = (bmin)
    # hdul_status.header['bmax'] = (bmax)
    # hdul_status.header['inBCdist'] = (initial_BC_distance)
    # hdul_status.header['energy_change'] = (energy_change)
    # hdul_status.header['collision'] = len(collided)
    # hdul_status.header['mA'] = (mA)
    # hdul_status.header['mB'] = (mB)
    # hdul_status.header['mC'] = (mC)
    # hdul_status.header['rB'] = (rB)


    # hdu.append(hdul_status)
    # hdu.close()

    # sa = rebound.Simulationarchive(f"{directoryf}/sim_{i}.bin")

print(f'Execution time: {datetime.datetime.now() - s} for {numSim} simulations with {t_end} years of integration time having time step dt = {sim.dt}')
    # print(feature_model.predict_stable(sim))
    # # >>> 0.011505529

    # # Bayesian neural net-based regressor
    # median, lower, upper = deep_model.predict_instability_time(sim, samples=10000)
    # print(int(median))