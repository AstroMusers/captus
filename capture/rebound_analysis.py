import rebound
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import datetime
from astropy.table import Table
import logging
import pandas as pd

import sys

s = datetime.datetime.now()
# Displays Time
current_time = s.strftime('%H%M')
print("current time :", current_time)
if not os.path.exists(f'/data/a.saricaoglu/repo/COMPAS/Files/Capture/'): 
    os.makedirs(f'/data/a.saricaoglu/repo/COMPAS/Files/Capture/') 
directoryf = f'/data/a.saricaoglu/repo/COMPAS/Files/Capture/' 
if not os.path.exists(f'/data/a.saricaoglu/repo/COMPAS/Plots/Capture/{str(s.strftime("%m.%d"))}/{current_time}/'): 
    os.makedirs(f'/data/a.saricaoglu/repo/COMPAS/Plots/Capture/{str(s.strftime("%m.%d"))}/{current_time}/') 
directoryp = f'/data/a.saricaoglu/repo/COMPAS/Plots/Capture/{str(s.strftime("%m.%d"))}/{current_time}/' 

# Get the script name
script_name = os.path.basename(__file__)
# Configure logging
log_filename = f"{directoryf}{str(s.strftime("%m.%d"))}/{current_time}/{script_name}_script.log"
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


folder = '04.03/2220'
runs = [f for f in os.listdir(directoryf + folder) if ".fits" in f]
sims = [sim for sim in os.listdir(directoryf + folder) if ".bin" in sim]

# Extract the numeric part of the filenames and create dictionaries for matching
run_dict = {run.split('_')[1].split('.')[0]: run for run in runs}  # Map run numbers to .fits files
sim_dict = {sim.split('_')[1].split('.')[0]: sim for sim in sims}  # Map sim numbers to .bin files

# Match files based on their numeric identifiers
matched_files = [(run_dict[num], sim_dict[num]) for num in run_dict.keys() if num in sim_dict]

print("Matched files (run, sim):", matched_files)




# Update the sims array to only include the filtered files
print(runs)
print(sims)

def find_first_event(statArray):
    for i in range(len(statArray)):
        if statArray[i] == 1:
            return i
    return False
def find_last_event(statArray):
    for i in range(len(statArray)-1, -1, -1):
        if statArray[i] == 1:
            return i
    return False

def find_stat_change(statArray):
    changes = []
    for i in range(len(statArray)-1):
        if statArray[i] != statArray[i+1]:
            changes.append([i, statArray[i], statArray[i+1]])
    if len(changes) == 0:
        changes.append(['no change, initial state: '+ str([0, statArray[0], statArray[-1]])])
    return changes

i = 0
# Proceed with the matched files
for run, sim in matched_files:
    print(f'Processing run: {run}, sim: {sim}')

    with fits.open(directoryf + folder + '/' + run) as hdul:
        data = hdul[1].data
        time = data["Time"]
        free = data["Free"]
        bound_b = data["Bound_B"]
        capture_not_bound_b = data["Capture_Not_Bound_B"]
        capture_bound_b = data["Capture_Bound_B"]

        header = hdul[1].header
        r_close = header['r_close']
        v_inf = header['v_inf']
        v1primme = header['v1primme']
        bmin = header['bmin']
        bmax = header['bmax']
        initial_BC_distance = header['inBCdist']
        mA = header['mA']
        mB = header['mB']
        mC = header['mC']
        rB = header['rB']
        
        print(f"r_close: {r_close}, v_inf: {v_inf}, v1primme: {v1primme}, bmin: {bmin}, bmax: {bmax}, initial_BC_distance: {initial_BC_distance}, mA: {mA}, mB: {mB}, mC: {mC}, rB: {rB}")
    hdul.close()

    # fig, ax = plt.subplots(figsize=(25,4), dpi=150)
    # time = np.arange(0, len(free), 1)

    # ax.bar(time, free, label='Free', color='blue', alpha=0.5 )
    # ax.bar(time, bound_b label='Capture Bound', color='red', alpha=0.5)
    # ax.bar(time, capture_not_bound_b, label='Capture Not Bound', color='green', alpha=0.5)
    # ax.bar(time, capture_bound_b, label='Capture Bound B', color='orange', alpha=0.5)
    # # ax.set_xlim(0,50)
    # ax.set_xlabel('Time (s)')
    # ax.legend(loc='upper right')
    # plt.savefig(directoryp + f'stats_{run}.png')
    # plt.close()

    crosscond1 = np.logical_and(bound_b == 1, capture_not_bound_b == 1)
    crosscond2 = np.logical_and(bound_b == 1, capture_bound_b == 1)
    crosscond3 = np.logical_and(bound_b == 1, free == 1)
    crosscond4 = np.logical_and(capture_not_bound_b == 1, free == 1)
    crosscond5 = np.logical_and(capture_bound_b == 1, free == 1)
    crosscond6 = np.logical_and(capture_bound_b == 1, capture_not_bound_b == 1)

    # print(f'crosscond1: {np.sum(crosscond1)}')
    # print(f'crosscond2: {np.sum(crosscond2)}')
    # print(f'crosscond3: {np.sum(crosscond3)}')
    # print(f'crosscond4: {np.sum(crosscond4)}')
    # print(f'crosscond5: {np.sum(crosscond5)}')
    # print(f'crosscond6: {np.sum(crosscond6)}')

    print(f'sum of crossconds: {np.sum(crosscond1) + np.sum(crosscond2) + np.sum(crosscond3) + np.sum(crosscond4) + np.sum(crosscond5) + np.sum(crosscond6)}')
    j = 0
    status_names = ['Free', 'Bound B', 'Capture Not Bound B', 'Capture Bound B']
    for status in [free, bound_b, capture_not_bound_b, capture_bound_b]:
        print(f'status: {status_names[j]} with duration time {len(status)}')
        print(f'first event: {find_first_event(status)}')
        print(f'last event: {find_last_event(status)}')
        print(f'status change: {find_stat_change(status)[-1]}')
        j += 1
        

    # op1 = rebound.OrbitPlot(sim, particles=[1,2])
    # op1.orbits[1].set_linestyle("--")
    # op1.particles.set_color(["green","red"])
    # op1.particles.set_sizes([50, 10])
    # op1.primary.set_sizes([100])

    # os.makedirs(f"{directoryp}/Sys_{i}")

    # for j in range(0, t_end):
    #     try:
    #         sim.integrate(sim.t+1)
    #         op1.update(updateLimits=True)
    #         op1.fig.suptitle(f"t = {sim.t:.2f} years, s")
    #         op1.fig.savefig(f"{directoryp}/Sys_{i}/orbit_{j}.png", dpi=300)
    #     except rebound.Collision:

    #         for p in sim.particles:
    #             if p.last_collision == sim.t:
    #                 collided.append(p.index)
    #         L.append(f"Collision at time {sim.t} particles {collided}")
    #     except Exception as e:
    #         print(f"Energy conservation error during integration: {e}")
    
    i += 1

    # sa = rebound.Simulationarchive(f"{directoryf}/sim_{i}.bin")
    # print(len(sa))
    # print(feature_model.predict_stable(sim))
    # # >>> 0.011505529

    # # Bayesian neural net-based regressor
    # median, lower, upper = deep_model.predict_instability_time(sim, samples=10000)
    # print(int(median))

