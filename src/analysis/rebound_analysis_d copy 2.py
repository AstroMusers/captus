import rebound
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import datetime
from astropy.table import Table
import logging
import pandas as pd
from astropy import units as u
import shutil

import sys
s = datetime.datetime.now()
mode1 = 'Mass_Variation'
mode2 = 'MbVinfA_Variation'
mode3 = 'McVinfImpB_Variation'
mode = mode2
# # Get the script name
script_name = os.path.basename(__file__)
script_name = script_name.split('.')[0]
folder = f'04.29/1639/MbVinfA_Variation/rebound_sim_mp_d.py_1639' 
# Displays Time
current_time = s.strftime('%H%M')
print("current time :", current_time)
if not os.path.exists(f'/data/a.saricaoglu/repo/COMPAS/Files/Capture/'): 
    os.makedirs(f'/data/a.saricaoglu/repo/COMPAS/Files/Capture/') 
directoryf = f'/data/a.saricaoglu/repo/COMPAS/Files/Capture/' 
if not os.path.exists(f'/data/a.saricaoglu/repo/COMPAS/Plots/Capture/{str(s.strftime("%m.%d"))}/{script_name}_{current_time}/{mode}/'): 
    os.makedirs(f'/data/a.saricaoglu/repo/COMPAS/Plots/Capture/{str(s.strftime("%m.%d"))}/{script_name}_{current_time}/{mode}/') 
directoryp = f'/data/a.saricaoglu/repo/COMPAS/Plots/Capture/{str(s.strftime("%m.%d"))}/{script_name}_{current_time}/{mode}/' 




# # Configure logging
# log_filename = f"{directoryf}{str(s.strftime("%m.%d"))}/{current_time}/{script_name}_script.log"
# os.makedirs(os.path.dirname(log_filename), exist_ok=True)
# logging.basicConfig(filename=log_filename, level=logging.INFO, 
#                     format='%(asctime)s - %(levelname)s - %(message)s')

# # Redirect stdout and stderr to the log file
# class StreamToLogger:
#     def __init__(self, logger, log_level):
#         self.logger = logger
#         self.log_level = log_level
#         self.linebuf = ''

#     def write(self, buf):
#         for line in buf.rstrip().splitlines():
#             self.logger.log(self.log_level, line.rstrip())

#     def flush(self):
#         pass

# sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
# sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

def status_logger(Si):
    global S

    S = np.vstack((S, Si))


def heartbeat(sim):
    global A
    global B
    global C
    global T
    global opC
    global L
    global E
    global SA
    global Si
    global E_initial
    global snpInt



    Evc = 0.5 * (sim.particles[2].vx**2 + sim.particles[2].vy**2 + sim.particles[2].vz**2)
    E_cond = Evc - (sim.G*sim.particles[0].m / sim.particles[2].a)

    v1prime = sim.particles[1].vx**2 + sim.particles[1].vy**2 + sim.particles[1].vz**2 - sim.particles[2].vx**2 - sim.particles[2].vy**2 - sim.particles[2].vz**2
    Evcprime = 0.5 * v1prime
    E_con_Bcapture = Evcprime - (sim.G*sim.particles[1].m / (sim.particles[1] ** sim.particles[2])) 
    # print(E_cond, E_con_Bcapture)

    t = int(sim.t * snpInt)

    Si[0] = t

    if (E_cond > 0):
        if (E_con_Bcapture >= 0):
            L.append(f"Free! at time {t}")
            Si[1] = 1
        else:
            L.append(f"Bound to B at time {t}")
            Si[2] = 1    
    else:
        if (E_con_Bcapture >= 0):
            L.append(f"Capture (not bound to B ) at time {t}")
            Si[3] = 1
        else:
            L.append(f"Capture (still bound to B ) at time {t}")
            Si[4] = 1


    status_logger(Si)
    Si = np.zeros(5)
    A.append(sim.particles[0].xyz)
    B.append(sim.particles[1].xyz)
    C.append(sim.particles[2].xyz)
    T.append(t)    
    opC.append(sim.particles[2].P)
    SA.append(sim.particles[2].a)
    E.append(sim.particles[2].e)

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
        changes.append(['no change, initial state: '+ str([0, statArray[0]])])
        return changes
    else:
        return changes


runs = [f for f in os.listdir(directoryf + folder) if ".fits" in f]
sims = [sim for sim in os.listdir(directoryf + folder) if ".bin" in sim]
print(os.listdir(directoryf + folder))
# Extract the numeric part of the filenames and create dictionaries for matching
run_dict = {run.split('_')[1].split('.')[0]: run for run in runs}  # Map run numbers to .fits files
sim_dict = {sim.split('_')[1].split('.')[0]: sim for sim in sims}  # Map sim numbers to .bin files

# Match files based on their numeric identifiers
matched_files = [(run_dict[num], sim_dict[num]) for num in run_dict.keys() if num in sim_dict]

print("Matched files (run, sim):", matched_files)


# Update the sims array to only include the filtered files
print(runs)
print(sims)


i = 0
massB = []
vINF = []
semajB = []
SAmean = []
Emean = []
Lifetime = []
dframes = []



# Proceed with the matched files
for run, sim in matched_files:
    syst = f"{run.split('_')[1]}_{run.split('_')[2].split('.')[0]}"
    # if os.path.exists(f"{directoryp}Sys_{syst}/"):
    #     print(f"System {syst} already exists, skipping...")
    #     continue
    # os.makedirs(f"{directoryp}Sys_{syst}/")


    print(f'Processing run: {run}, sim: {sim}')

    A = []
    B = []
    C = []
    E = []
    opC = []
    SA = []
    T = []
    L = []
    collided = []
    Si = np.zeros(5)
    S = np.empty((0,5))
    S_info = []
    with fits.open(directoryf + folder + '/' + run, mode='update') as hdul:
        # hdul[1].data = Table(S)
        # hdul.flush()

        # data = hdul[1].data
        # time = data["Time"]
        # df_stats['Free'] = data["Free"]
        # Bcapture = data["Bcapture"]
        # df_stats['Acapture'] = data["Acapture"]
        # df_stats['ABcapture'] = data["ABcapture"]


        header = hdul[1].header
        r_close = header['rClose']
        v_inf = header['vInf']
        b = header['b']
        beta = header['beta']
        lambda1 = header['lambda1']
        v1prime = header['vPrm1']
        v1prime2 = header['vPrm2']
        bmin = header['bmin']
        bmax = header['bmax']
        initial_BC_distance = header['inBCdist']
        mA = header['mA']
        mB = header['mB']
        mC = header['mC']
        rB = header['rB']
        rA = header['rA']
        aB = header['aB']
        opB = float(header['opB'])
        totYr = header['totSimYr']
        dt = header['dt']
        fYr = header['fSimYr']
        fBYr = header['fBYr']
        snpInt = header['snpInt']
        snpRt = header['snpRt']



        print(f"snapshot interval:{snpInt} opB: {opB}, final year:{fYr}, r_close: {r_close}, v_inf: {v_inf}, v1primme: {v1prime}, bmin: {bmin}, bmax: {bmax}, initial_BC_distance: {initial_BC_distance}, mA: {mA}, mB: {mB}, mC: {mC}, rB: {rB}")
        if b < bmin or b > bmax:
            print(f"b: {b} is out of range")
            continue
        archived_sim = rebound.Simulationarchive(f'{directoryf}{folder}/{sim}')
        acs=[]
        for sim in archived_sim:
            acs.append(sim.particles[2].a)
        if (np.max(acs) < 120):
            # print(f"System {syst} has an orbit larger than 120 AU")
            # continue
            if fYr/opB > 1e7:
                print(f"fYr: {fYr} is too large")
                for simulation in archived_sim:
                    heartbeat(simulation[::100])
                    # op1 = rebound.OrbitPlot(sim, particles=[1,2])
                    # op1.orbits[1].set_linestyle("--")
                    # op1.particles.set_color(["green","red"])
                    # op1.particles.set_sizes([50, 10])
                    # op1.primary.set_sizes([100])
                    # op1.sa
                T = [t*100 for t in T]
            else:
                for simulation in archived_sim:
                    heartbeat(simulation)    
        os.makedirs(f"{directoryf}{str(s.strftime("%m.%d"))}/{script_name}_{current_time}/{mode}/", exist_ok=True)
        # Copy the FITS file
        shutil.copy(directoryf + folder + '/' + run, f"{directoryf}{str(s.strftime("%m.%d"))}/{script_name}_{current_time}/{mode}/{run}_copy.fits")

        print(f"File copied to {directoryf}{str(s.strftime("%m.%d"))}/{script_name}_{current_time}/{mode}/")
    hdul.close()
    with fits.open(f"{directoryf}{str(s.strftime('%m.%d'))}/{script_name}_{current_time}/{mode}/{run}_copy.fits", mode='update') as hdul:
        hdul.append(fits.BinTableHDU(Table(data=[A, B, C, T, E, SA], names=['xyzA', 'xyzB', 'xyzC', 'time', 'eC', 'aC'])))
        hdul.flush()
        hdul.append(fits.BinTableHDU(Table(data=S, names=['Time', 'Free', 'Bcapture', 'Acapture', 'ABcapture'])))
        hdul.flush()


        SAmean.append(np.mean(SA[1:-1]))
        Emean.append(np.mean(E[1:-1]))
        massB.append(mB)
        vINF.append(v_inf)
        semajB.append(aB)
        Lifetime.append(fYr)

        # mc = '{:.3E}'.format(mC)
        # vinf = '{:.3f}'.format(v_inf)
        # fyr = '{:.3E}'.format(fYr)

        # fig, ax = plt.subplots(2,1,figsize=(8,12), sharex=True)
        # plt.rcParams.update({'font.size': 20})
        # samean = '{:.3f}'.format(np.mean(SA))
        # emean = '{:.3f}'.format(np.mean(E))
        # beta = '{:.3f}'.format(beta)
        # lambda1 = '{:.3f}'.format(lambda1)
        # b = '{:.3f}'.format(b)
        # ax[0].set_title(r"$\bar{a}$ = " + f"{samean}")
        # ax[1].set_title(r"$\bar{e}$ = " + f"{emean}")
        # ax[0].set_ylabel('Semi-major axis (AU)')
        # ax[1].set_ylabel('Eccentricity')
        # ax[1].set_xlabel('Time (yr)')
        # ax[1].set_xticks(np.arange(0, len(T), 100000), [f'{x:.2f}' for x in T[::100000]])
        # ax[0].plot(T, SA, label='A', color='blue', alpha=1 )
        # ax[1].plot(T, E, label='E', color='red', alpha=1)
        # plt.suptitle(fr'$M_C$ = {mc}, $v_inf$= {vinf}, b = {b}')
        # plt.savefig(f"{directoryp}Sys_{syst}/"+ f'SunJptr_capture_a_e_{syst}.png', dpi=200, bbox_inches='tight')
        # plt.close()

        # fig, ax = plt.subplots(figsize=(20,20))
        # plt.rcParams.update({'font.size': 28})
        # plotInt = int((1/snpInt) * snpInt) * 10
        # body_names = ["A", "B", "C"]
        # marker = ["*", ".", "v"]
        # n = 0
        # for body in [A, B, C]:
        #     x, y, z = zip(*body[::plotInt])
        #     #print(len(x))
        #     ax.scatter(x, y, c=T[::plotInt],cmap='viridis', marker=marker[n], s=5)  # Small dots for trails
        #     n += 1        
        # ax.set_xlabel("X (AU)")
        # ax.set_ylabel("Y (AU)")
        # plt.title(fr'$M_C$ = {mc}, $v_inf$= {vinf}, b = {b}, $\tau$ = {fyr}')
        # plt.savefig(f"{directoryp}Sys_{syst}/"+ f'SunJptr_capture_{syst}.png', dpi=200)
        # plt.close()


        # j = 1
        # df_stats = pd.DataFrame(S, columns=['Time', 'F', 'Bcap', 'Acap', 'ABcap'])
        # for statname in df_stats.columns[1:]:
        #     status = df_stats[statname]
        #     print(f'status: {statname} with duration time {len(status)}')
        #     print(f'first event: {find_first_event(status)}')
        #     print(f'last event: {find_last_event(status)}')
        #     print(f'last status change: {find_stat_change(status)[-1]}')
        #     print(f'total status time: {np.sum(status)}')
        #     hdul[3].header[f'{statname}fE'] = find_first_event(status)
        #     hdul[3].header[f'{statname}fE'] = find_last_event(status)
        #     hdul[3].header[f'{statname}lC'] = str(find_stat_change(status)[-1])
        #     hdul[3].header[f'{statname}tST'] = np.sum(status)
            
        #     try:
        #         print(f"longest status duration: {np.max(find_stat_change(status)[:][3])}")
        #         hdul[3].header[f'{statname}lSD'] = np.max(find_stat_change(status)[:][3])
        #     except:
        #         hdul[3].header[f'{statname}lSD'] = 0
        #     j += 1

        # hdul.flush()
        hdul.close()


        # fig, ax = plt.subplots(figsize=(25,4), dpi=150)
        # time = np.arange(0, len(df_stats['Free']), 1)
        # plt.rcParams.update({'font.size': 20})
        # ax.bar(time, df_stats['Free'], label=f'Free ({df_stats_info['Total Status Time'][0]})', color='blue', alpha=0.5 )
        # ax.bar(time, df_stats['Bcapture'],label=f'Bcapture ({df_stats_info['Total Status Time'][1]})', color='red', alpha=0.5)
        # ax.bar(time, df_stats['Acapture'], label=f'Acapture ({df_stats_info['Total Status Time'][2]})', color='green', alpha=0.5)
        # ax.bar(time, df_stats['ABcapture'], label=f'ABcapture ({df_stats_info['Total Status Time'][3]})', color='orange', alpha=0.5)
        # # ax.set_xlim(0,50)
        # ax.set_xlabel('Time (yr)')
        # ax.legend(loc='upper right')
        # plt.savefig(f"{directoryp}Sys_{syst}/"+ f'stats_{syst}.png', bbox_inches='tight')
        # plt.close()

        # crosscond1 = np.logical_and(df_stats['Bcap'] == 1, df_stats['Acap'] == 1)
        # crosscond2 = np.logical_and(df_stats['Bcap'] == 1, df_stats['ABcap'] == 1)
        # crosscond3 = np.logical_and(df_stats['Bcap'] == 1, df_stats['F'] == 1)
        # crosscond4 = np.logical_and(df_stats['Acap'] == 1, df_stats['F'] == 1)
        # crosscond5 = np.logical_and(df_stats['ABcap'] == 1, df_stats['F'] == 1)
        # crosscond6 = np.logical_and(df_stats['ABcap'] == 1, df_stats['Acap'] == 1)


        # print(f'sum of crossconds: {np.sum(crosscond1) + np.sum(crosscond2) + np.sum(crosscond3) + np.sum(crosscond4) + np.sum(crosscond5) + np.sum(crosscond6)}')
        j = 0
        i += 1
#     status_names = ['Free', 'Bcapture', 'Acapture', 'ABcapture']


vINF = [v * (u.au/u.yr) for v in vINF]
vINF = [v.to(u.km/u.s).value for v in vINF]

os.makedirs(f"{directoryf}{str(s.strftime("%m.%d"))}/{script_name}_{current_time}/{mode}/", exist_ok=True)
fit_filename = f"{directoryf}{str(s.strftime("%m.%d"))}/{script_name}_{current_time}/{mode}/rebound_analysis_d_general_a_lim.fits"
hdu_pr = fits.PrimaryHDU()
hdu_pr.writeto(fit_filename, overwrite=True)
hdu = fits.open(fit_filename, mode='update')
hdu.append(fits.BinTableHDU(Table(data=[massB, vINF, semajB, Lifetime, SAmean, Emean], names=['massB', 'vINF', 'semajB', 'Lifetime', 'SAmean', 'Emean'])))
hdu.writeto(fit_filename, overwrite=True)
hdu.close()



# fig = plt.figure(figsize=(15, 15))
# plt.rcParams.update({'font.size': 20})
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(mPBH, vINF, ImpactB, c=Lifetime, marker='o', s=100)

# ax.set_xticks(np.logspace(-13, -4, 10), [f'{x:.2f}' for x in np.logspace(-13, -4, 10)])
# ax.set_yticks(np.linspace(0.2108, 21.08, 10), [f'{x:.2f}' for x in np.linspace(0.2108, 21.08, 10)])
# ax.set_zticks(np.linspace(np.min(ImpactB), np.max(ImpactB), 10), [f'{x:.3f}' for x in np.linspace(np.min(ImpactB), np.max(ImpactB), 10)])
# # Add labels and title
# ax.set_xlabel(r'$M_{PBH}$ ($M_\odot$)')
# ax.set_ylabel(r'$\infty$ (km/s)')
# ax.set_zlabel("b (AU)")
# plt.savefig(f"{directoryp}mC_vInf_b_values.png", dpi=300, bbox_inches='tight')
# plt.close()

# mPBHmean = '{:.3f}'.format(np.mean(mPBH))
# vINFmean = '{:.3f}'.format(np.mean(vINF))
# fig, ax = plt.subplots(3,1,figsize=(8, 6))
# ax[0].hist(mPBH, bins=20, color='blue')
# ax[0].set_title(r"$\bar{M}_{PBH}$ = " + f"{mPBHmean}")
# ax[0].set_xlabel(r'$M_{PBH}$ ($M_\odot$)')
# ax[0].set_ylabel('Frequency')
# ax[1].hist(vINF, bins=20, color='red')
# ax[1].set_title(r"$\bar{v}_{\infty}$ = " + f"{vINFmean}")
# ax[1].set_xlabel(r'$v_{\infty}$ (km/s)')
# ax[1].set_ylabel('Frequency')
# ax[2].hist(ImpactB, bins=20, color='green')
# ax[2].set_title(r"$\bar{b}$ = " + f"{np.mean(ImpactB)}")
# ax[2].set_xlabel('b (AU)')
# ax[2].set_ylabel('Frequency')
# plt.tight_layout()
# plt.savefig(f"{directoryp}histogram_mC_vInf_b.png", dpi=300, bbox_inches='tight')
# plt.close()

# samean = '{:.3f}'.format(np.mean(SAmean))
# emean = '{:.3f}'.format(np.mean(Emean))

# fig, ax = plt.subplots(2,1,figsize=(8, 6))
# ax[0].hist(SAmean, bins=20, color='blue')
# ax[0].set_title(r"$\bar{a}$ = " + f"{samean}")
# ax[0].set_xlabel('Semi-major axis (AU)')
# ax[0].set_ylabel('Frequency')
# ax[1].hist(Emean, bins=20, color='red')
# ax[1].set_title(r"$\bar{e}$ = " + f"{(emean)}")
# ax[1].set_xlabel('Eccentricity')
# ax[1].set_ylabel('Frequency')

# plt.tight_layout()
# plt.savefig(f"{directoryp}histogram_a_e.png", dpi=300, bbox_inches='tight')
# plt.close()

print(f"{len(mPBH)} survived among {120**3} systems on the grid")
for i in range(len(mPBH)):
    print(f"System {i} with mPBH: {mPBH[i]}, vINF: {vINF[i]}, ImpactB: {ImpactB[i]}, Lifetime: {Lifetime[i]}")
    print(f"Mean semimajor axis: {SAmean[i]}, Mean eccentricity: {Emean[i]}")
    print(f"Total captured time to A: {dframes[i][3][4]} + {dframes[i][4][4]} = {dframes[i][3][4]+dframes[i][4][4]}, Total free time: {dframes[i][1][4]}")
    print(f"Longest captured time: {dframes[i][3][5]} + {dframes[i][4][5]} = {dframes[i][3][5]+dframes[i][4][5]}")
    print(f'Percentage captured to A: {100*(dframes[i][3][4]+dframes[i][4][4])/(Lifetime[i])} %')

        



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
    #         L.append(f"Collision at Time (yr) {sim.t} particles {collided}")
    #     except Exception as e:
    #         print(f"Energy conservation error during integration: {e}")
    
    

    # sa = rebound.Simulationarchive(f"{directoryf}/sim_{i}.bin")
    # print(len(sa))
    # print(feature_model.predict_stable(sim))
    # # >>> 0.011505529

    # # Bayesian neural net-based regressor
    # median, lower, upper = deep_model.predict_instability_Time (yr)(sim, samples=10000)
    # print(int(median))

