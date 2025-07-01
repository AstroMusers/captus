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

sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)


sim = rebound.Simulation()
# sim.start_server(1234)

# sim.dt = 0.0012 * 2 * np.pi
# sim.integrator = "mercurius"
# sim.ri_mercurius.r_crit_hill = .1


# sim.add(m=5.0, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0)
# sim.add(m=3, x=2.0, vy = np.sqrt(1/2))
# sim.add(m=1, x=4.0, vy=1.0)

# sim.move_to_com()
# e_initial = sim.energy()


# sim.integrate(2000)
# sim.status()

# for p in sim.particles:
#     print(p.x, p.y, p.z)
# for o in sim.orbits(): 
#     print(o)




import rebound
import numpy as np
import matplotlib.pyplot as plt


# def status_logger(Si):
#     global S

#     S = np.vstack((S, Si))


def runtime_checks(sim):
    E_current = sim.energy()
    error = abs(E_current - E_initial)/E_initial

    Evc = 0.5 * (sim.particles[2].vx**2 + sim.particles[2].vy**2 + sim.particles[2].vz**2)
    E_cond = Evc - (sim.G*sim.particles[0].m / sim.particles[2].a)

    if error > 1e-5:
        raise Exception(f"Error in energy conservation: {error}")
    
    if (sim.particles[2].a > 200):
        if (E_cond > 0):
            print(f'Energy condition: {E_cond}')
            raise Exception(f"Particle C is too far away and free: {sim.particles[2].a}")
    
    for p in sim.particles:
        if p.last_collision == sim.t:
            raise Exception(f"Collision detected at time {sim.t}")

# def heartbeat(sim_pointer):
    # global A
    # global B
    # global C
    # global T
    # global L
    # global Si
    # global E_initial
    # sim = sim_pointer.contents



    # Evc = 0.5 * (sim.particles[2].vx**2 + sim.particles[2].vy**2 + sim.particles[2].vz**2)
    # E_cond = Evc - (sim.G*sim.particles[0].m / sim.particles[2].a)

    # v1prime = sim.particles[1].vx**2 + sim.particles[1].vy**2 + sim.particles[1].vz**2 - sim.particles[2].vx**2 - sim.particles[2].vy**2 - sim.particles[2].vz**2
    # Evcprime = 0.5 * v1prime
    # E_con_bound_b = Evcprime - (sim.G*sim.particles[1].m / (sim.particles[1] ** sim.particles[2])) 
    # # print(E_cond, E_con_bound_b)

    # t = int(sim.t)

    # Si[0] = t

    # if (E_cond > 0):
    #     if (E_con_bound_b >= 0):
    #         L.append(f"Free! at time {sim.t}")
    #         Si[1] = 1
    #     else:
    #         L.append(f"Bound to B at time {sim.t}")
    #         Si[2] = 1    
    # else:
    #     if (E_con_bound_b >= 0):
    #         L.append(f"Capture (not bound to B ) at time {sim.t}")
    #         Si[3] = 1
    #     else:
    #         L.append(f"Capture (still bound to B ) at time {sim.t}")
    #         Si[4] = 1

    # status_logger(Si)
    # Si = np.zeros(5)
    # A.append(sim.particles[0].xyz)
    # B.append(sim.particles[1].xyz)
    # C.append(sim.particles[2].xyz)
    # T.append(sim.t)
pathToData = '/data/a.saricaoglu/repo/COMPAS/Files/msstarvsx.fits'
with fits.open(pathToData) as hdul:
    data = hdul[1].data
    subset = data[100:300]
    print(data.columns)
    comp_mass = subset["Companion_mass"]
    msstar_mass = subset["MSstar_mass"]
    comp_radius  = subset["Companion_radius"]
    msstar_radius  = subset["MSstar_radius"]
    semajax = subset["Semimajor_axis"]
    orbperiod = subset["Orbital_period"]
    # print(comp_mass)
    # print(msstar_mass)
    # print(semaj)
    binary_df = pd.DataFrame()
    print(len(binary_df))
    cmas = []
    msmas = []
    crad = []
    msrad = []
    semaj = []
    orbper = []
    for cm, mm, cr, mr, a, t  in zip(comp_mass, msstar_mass, comp_radius, msstar_radius, semajax, orbperiod):
        if (cm > 0) and (mm > 0) and (a > 0):
            cmas.append(cm)
            msmas.append(mm)
            crad.append(cr)
            msrad.append(mr)
            semaj.append(a)
            orbper.append(t)

    binary_df["Companion_mass"] = cmas
    binary_df["MSstar_mass"] = msmas
    binary_df["Companion_radius"] = crad
    binary_df["MSstar_radius"] = msrad
    binary_df["Semimajor_axis"] = semaj
    binary_df["Orbital_period"] = orbper
hdul.close()

print(len(binary_df["Companion_mass"]))
numSim = 100
for i in range(numSim):
    print(f"Simulation {i+1} of {numSim} starting at {datetime.datetime.now()}")

    #print(sim.G)
    epsilon = 0.1 
    lambda1 = 0
    beta = np.pi/3
    e_b = 0.0489 # Eccentricity

    mA = binary_df["MSstar_mass"][i] # mass of A in Msun
    mB = binary_df["Companion_mass"][i]  # mass of B in Msun
    rA = binary_df["MSstar_radius"][i] * (const.R_sun)  # radius of A in AU
    rA = rA.to(u.au).value
    rB = binary_df["Companion_radius"][i] * (const.R_sun)   # radius of B in AU
    rB = rB.to(u.au).value
    aB = binary_df["Semimajor_axis"][i]    # semi-major axis of A-B in AU
    opB = binary_df["Orbital_period"][i] * (u.day)  # orbital period of A-B in yr
    opB = opB.to(u.yr).value

    print(f"mA: {mA}, mB: {mB}, rA: {rA}, rB: {rB}, aB: {aB}, opB: {opB}")

    for mC in np.logspace(np.log10(mB)-6,np.log10(mB)-6, 7): # mass of C in Msun
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
        #print(sim.particles[1].vxyz)

        # Add third body (C) with a hyperbolic trajectory
        v_inf = 4.2161  # Velocity at infinity in AU/yr, 20 km/s
        bmin = 0.0018018197621958202 
        bmax =  aB * (mB*epsilon / mA)**(1/3)  
        r_close = np.random.uniform(bmin, bmax)   # Initial distance of C from A-B in AU
        r_close_x = np.random.uniform(-1, 1) * r_close
        r_close_y = np.random.uniform(-np.sqrt(r_close**2 - r_close_x**2), np.sqrt(r_close**2 - r_close_x**2))
        r_close_z = np.sqrt(r_close**2 - r_close_x**2 - r_close_y**2)
        # r_close_x = r_close * np.cos(beta) * np.sin(lambda1)
        # r_close_y = r_close *  np.cos(beta) * np.cos(lambda1)
        # r_close_z = r_close * np.sin(beta)

        vz_c = - v_inf * np.sin(beta)
        vx_c = v_inf * np.cos(beta) * np.sin(lambda1)
        vy_c = v_inf * np.cos(beta) * np.cos(lambda1)

        # sim.add(m=mC, x=r_close_x + a_b*np.cos(lambda1), y=-r_close_y + a_b*np.sin(lambda1),  z=r_close_z ,  vx=vx_c, vy=vy_c, vz = vz_c) 
        sim.add(m=mC, x=sim.particles[1].x - r_close_x, y=sim.particles[1].y - r_close_y,  z=sim.particles[1].z - r_close_z ,  vx=vx_c, vy=vy_c, vz = vz_c)
        #print(sim.particles[2].vxyz)
        v1primme = sim.particles[1].vx**2 + sim.particles[1].vy**2 + sim.particles[1].vz**2 - sim.particles[2].vx**2 - sim.particles[2].vy**2 - sim.particles[2].vz**2
        # bmin = 1/v1primme * np.sqrt(2*sim.G*mB*rB + (rB*v1primme)**2) 
        # sim.add(m=0, a=100)  # Add a distant body to make Spock work
        initial_BC_distance = sim.particles[1]  ** sim.particles[2]
        # Integration parameters
        sim.dt = sim.particles[1].P*0.05   # timestep is 5% of orbital period
        t_end = int(1e7)  # Total integration time in years
        # N_steps = int(t_end / sim.dt)  # Number of steps to integrate
        # N_steps = 100 * sim.particles[1].P # 100 orbital period duration
        sim.collision = "direct"  # Turn on collisions
        print(f'vC wrt vB: {v1primme}, bmin: {bmin}, bmax: {bmax}, hill radius: {sim.ri_mercurius.r_crit_hill}')
        print(f'dt = {sim.dt}, orbital period (of B) = {sim.particles[1].P}, total years of integration = {t_end}, total years for B = {t_end/sim.particles[1].P}')

        # # Store data for visualization
        # A = []
        # B = []
        # C = []
        # T = []
        # L = []
        collided = []
        # Si = np.zeros(5)
        # S = np.empty((0,5))
        #print(np.shape(S))
        counter = 0

        # sim.heartbeat = heartbeat
        E_initial = sim.energy()
        print(f'time zero {sim.t}, initial energy {E_initial}')
        sim.save_to_file(directoryf +f"sim_{i}_{mc}.bin",interval=sim.dt, delete_file=True)
        
        step = 1  
        try:
            for j in range(0, int(t_end/step)):
                sim.integrate(sim.t+step)
                runtime_checks(sim)
        except Exception as e:
            print(f"Error during integration: {e}")
            print(f"Simulation failed for system {i} {mc}, at time {sim.t}")
            print("Skipping to next system...")
            if sim.t < t_end*0.1:
                os.remove(directoryf +f"sim_{i}_{mc}.bin")
                print(f"File {directoryf +f"sim_{i}_{mc}.bin"} has been deleted.")
            continue
        E_final = sim.energy()
        energy_change = abs(E_final - E_initial)/E_initial
        print(f'final energy {E_final}, final time {sim.t}, final distance {sim.particles[1]  ** sim.particles[2]}, energy change {energy_change}')

        #print(sim.particles[0].xyz)

        # fig, ax = plt.subplots(figsize=(20,20))

        # body_names = ["A", "B", "C"]
        # marker = ["*", ".", "v"]
        # n = 0
        # for body in [A, B, C]:
        #     x, y, z = zip(*body[::10])
        #     #print(len(x))
        #     t = np.linspace(0, t_end, len(x)) 
        #     ax.scatter(x, y, c=t,cmap='viridis', marker=marker[n], s=5)  # Small dots for trails
        #     n += 1
            
        # ax.set_xlabel("X (AU)")
        # ax.set_ylabel("Y (AU)")
        # plt.title(f'$M_A$ = {"{:.2E}".format(mA)}, $M_B$= {"{:.2E}".format(mB)}, $M_C$= {"{:.2E}".format(mC)}, $v_i$= {"{:.2E}".format(v_inf)}, $r_c$= {"{:.2E}".format(r_close)}')
        # plt.savefig(directoryp + f'capture_{i}.png', dpi=150)

        # # print(L)
        # #print(S)
        print(f'r_close = {r_close} and initial BC separation =  {initial_BC_distance}')
        # #print(len([s for s in S if  s[0]==0]))
        # #print([s for s in S if  s[0]==0])
        # #print(S[25:39])
        # print(np.sum(S[:,1]), np.sum(S[:,2]), np.sum(S[:,3]), np.sum(S[:,4]))
        # print(np.sum(S[:,1]) + np.sum(S[:,2]) + np.sum(S[:,3]) + np.sum(S[:,4]))
        # #print(sim.particles[0])
        # #print(sim.orbits())

        # print('s shape', np.shape(S))
        fit_filename = f"{directoryf}rebound_{i}_{mc}.fits"
        hdu_pr = fits.PrimaryHDU()
        hdu_pr.writeto(fit_filename, overwrite=True)
        hdu = fits.open(fit_filename, mode='update')

        hdul_status = fits.BinTableHDU(Table(names=('Time','Free','Bound_B','Capture_Not_Bound_B','Capture_Bound_B')))
        hdul_status.header['rClose'] = (r_close)
        hdul_status.header['vInf'] = (v_inf)
        hdul_status.header['vPrm'] = (v1primme)
        hdul_status.header['bmin'] = (bmin)
        hdul_status.header['bmax'] = (bmax)
        hdul_status.header['inBCdist'] = (initial_BC_distance)
        hdul_status.header['deltaE'] = (energy_change)
        hdul_status.header['colls'] = len(collided)
        hdul_status.header['mA'] = (mA)
        hdul_status.header['mB'] = (mB)
        hdul_status.header['mC'] = (mC)
        hdul_status.header['rB'] = (rB)
        hdul_status.header['rA'] = (rA)
        hdul_status.header['aB'] = (aB)
        hdul_status.header['opB'] = (opB)



        hdu.append(hdul_status)
        hdu.close()

        sa = rebound.Simulationarchive(f"{directoryf}sim_{i}_{mc}.bin")
        print(len(sa))


print(f'Execution time: {datetime.datetime.now() - s} for {numSim} simulations with {t_end} years of integration time having time step dt = {sim.dt}')
    # print(feature_model.predict_stable(sim))
    # # >>> 0.011505529

    # # Bayesian neural net-based regressor
    # median, lower, upper = deep_model.predict_instability_time(sim, samples=10000)
    # print(int(median))