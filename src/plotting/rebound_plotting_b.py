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

import sys
s = datetime.datetime.now()
mode1 = 'Mass_Variation'
mode2 = 'LBImpB_Variation'
mode3 = 'McVinfImpB_Variation'
mode = mode3
# Displays Time
current_time = s.strftime('%H%M')
print("current time :", current_time)
if not os.path.exists(f'/data/a.saricaoglu/repo/COMPAS/Files/Capture/'): 
    os.makedirs(f'/data/a.saricaoglu/repo/COMPAS/Files/Capture/') 
directoryf = f'/data/a.saricaoglu/repo/COMPAS/Files/Capture/' 
if not os.path.exists(f'/data/a.saricaoglu/repo/COMPAS/Plots/Capture/{str(s.strftime("%m.%d"))}/{current_time}/{mode}/'): 
    os.makedirs(f'/data/a.saricaoglu/repo/COMPAS/Plots/Capture/{str(s.strftime("%m.%d"))}/{current_time}/{mode}/') 
directoryp = f'/data/a.saricaoglu/repo/COMPAS/Plots/Capture/{str(s.strftime("%m.%d"))}/{current_time}/{mode}/' 

primary_folder = f'04.21/1609/McVinfImpB_Variation/rebound_sim_mp_b.py_1609/'
secondary_folder = '04.20/1508/' + mode

runs = [f for f in os.listdir(directoryf + primary_folder) if ".fits" in f]
# sims = [sim for sim in os.listdir(directoryf +secondary_folder) if "sys" in sim]

# # Extract the numeric part of the filenames and create dictionaries for matching
# run_dict = {run.split('_')[1].split('.')[0]: run for run in runs}  # Map run numbers to .fits files
# sim_dict = {sim.split('_')[1].split('.')[0]: sim for sim in sims}  # Map sim numbers to .bin files

# # Match files based on their numeric identifiers
# matched_files = [(run_dict[num], sim_dict[num]) for num in run_dict.keys() if num in sim_dict]

# print("Matched files (run, sim):", matched_files)
# for primary, secondary in matched_files:
#     print(f'Processing run: {primary}, sim: {secondary}')
directoryp = "/data/a.saricaoglu/repo/COMPAS/Plots/Capture/04.22/2300/McVinfImpB_Variation/"
for file in runs :
    print(f"Processing file:{directoryf}{primary_folder}/{file}")
    syst = file.split('_')[1].split('.')[0]
    if os.path.exists(f"{directoryp}Sys_{syst}/"):
        print(f"System {syst} already exists, skipping...")
        continue
    os.makedirs(f"{directoryp}Sys_{syst}/")

    with fits.open(f'{directoryf}{primary_folder}/{file}') as hdul:
        hdul.info()
        data = hdul[1].data
        print(data)
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
        print(f"snapshot interval:{snpInt} opB: {opB}, final year:{fYr}, r_close: {r_close}, v_inf: {v_inf}, v1primme: {v1prime}, bmin: {bmin}, bmax: {bmax}, initial_BC_distance: {initial_BC_distance}, mA: {mA}, mB: {mB}, mC: {mC}, rB: {rB}")

        # T = hdul[1].data['time']
        # syst = hdul[2].header['syst']
        T = hdul[-3].data['time']
        A = hdul[-3].data['xyzA']
        B = hdul[-3].data['xyzB']
        C = hdul[-3].data['xyzC']
        E = hdul[-3].data['eC']
        SA = hdul[-3].data['aC']
        S = hdul[-2].data
        S_info = hdul[-1].data
    hdul.close()
    os.makedirs(f"{directoryp}Sys_{syst}/", exist_ok=True)
    print(f'directory created: {directoryp}Sys_{syst}/')



    mc = '{:.3E}'.format(mC)
    vinf = '{:.3f}'.format(v_inf)
    fyr = '{:.3E}'.format(fYr)

    fig, ax = plt.subplots(2,1,figsize=(8,12), sharex=True)
    plt.rcParams.update({'font.size': 20})
    samean = '{:.3f}'.format(np.mean(SA))
    emean = '{:.3f}'.format(np.mean(E))
    beta = '{:.3f}'.format(beta)
    lambda1 = '{:.3f}'.format(lambda1)
    ab = '{:.3f}'.format(b)
    ax[0].set_title(r"$\bar{a}$ = " + f"{samean}")
    ax[1].set_title(r"$\bar{e}$ = " + f"{emean}")
    ax[0].set_ylabel('Semi-major axis (AU)')
    ax[1].set_ylabel('Eccentricity')
    ax[1].set_xlabel('Time (yr)')
    ax[1].set_xticks(np.arange(0, len(T), 100000), [f'{x:.2f}' for x in T[::100000]])
    ax[0].plot(T, SA, label='A', color='blue', alpha=1 )
    ax[1].plot(T, E, label='E', color='red', alpha=1)
    plt.suptitle(fr'$M_C$ = {mc}, $v_inf$= {vinf}, b = {b}')
    plt.savefig(f"{directoryp}Sys_{syst}/"+ f'SunJptr_capture_a_e_{syst}.png', dpi=200, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(20,20))
    plt.rcParams.update({'font.size': 28})
    plotInt = int((1/snpInt) * snpInt) * 500
    body_names = ["A", "B", "C"]
    marker = ["*", ".", "v"]
    n = 0
    for body in [A, B, C]:
        x, y, z = zip(*body[::plotInt])
        #print(len(x))
        ax.scatter(x, y, c=T[::plotInt],cmap='viridis', marker=marker[n], s=5)  # Small dots for trails
        n += 1        
    ax.set_xlabel("X (AU)")
    ax.set_ylabel("Y (AU)")
    plt.title(fr'$M_C$ = {mc}, $v_inf$= {vinf}, b = {b}, $\tau$ = {fyr}')
    plt.savefig(f"{directoryp}Sys_{syst}/"+ f'SunJptr_capture_{syst}.png', dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(25,4), dpi=150)
    time = np.arange(0, len(S['Free']), 1)
    plt.rcParams.update({'font.size': 20})
    ax.bar(time, S['Free'], label=f'Free ({S_info['F'][4]})', color='blue', alpha=0.5 )
    ax.bar(time, S['Bcapture'],label=f'Bcapture ({S_info['Bcap'][4]})', color='red', alpha=0.5)
    ax.bar(time, S['Acapture'], label=f'Acapture ({S_info['Acap'][4]})', color='green', alpha=0.5)
    ax.bar(time, S['ABcapture'], label=f'ABcapture ({S_info['Abcap'][4]})', color='orange', alpha=0.5)
    # ax.set_xlim(0,50)
    ax.set_xlabel('Time (yr)')
    ax.legend(loc='upper right')
    plt.savefig(f"{directoryp}Sys_{syst}/"+ f'stats_{syst}.png', bbox_inches='tight')
    plt.close()


with fits.open(directoryf + secondary_folder + "rebound_analysis_b_general_.fits") as hdul:
    data = hdul[1].data
    mPBH = data["mPBH"]
    vINF = data["vINF"]
    ImpactB = data["ImpactB"]
    Lifetime = data["Lifetime"]
    SAmean = data["SAmean"]
    Emean = data["Emean"]
    dframes = data["Status"]
hdul.close()

vINF = [v * (u.au/u.yr) for v in vINF]
vINF = [v.to(u.km/u.s).value for v in vINF]


fig = plt.figure(figsize=(15, 15))
plt.rcParams.update({'font.size': 20})
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mPBH, vINF, ImpactB, c=Lifetime, marker='o', s=100)

ax.set_xticks(np.logspace(-13, -4, 10), [f'{x:.2f}' for x in np.logspace(-13, -4, 10)])
ax.set_yticks(np.linspace(0.2108, 21.08, 10), [f'{x:.2f}' for x in np.linspace(0.2108, 21.08, 10)])
ax.set_zticks(np.linspace(np.min(ImpactB), np.max(ImpactB), 10), [f'{x:.3f}' for x in np.linspace(np.min(ImpactB), np.max(ImpactB), 10)])
# Add labels and title
ax.set_xlabel(r'$M_{PBH}$ ($M_\odot$)')
ax.set_ylabel(r'$\infty$ (km/s)')
ax.set_zlabel("b (AU)")
plt.savefig(f"{directoryp}mC_vInf_b_values.png", dpi=300, bbox_inches='tight')
plt.close()

mPBHmean = '{:.3E}'.format(np.mean(mPBH))
vINFmean = '{:.3E}'.format(np.mean(vINF))
bmean = '{:.3E}'.format(np.mean(ImpactB))
fig, ax = plt.subplots(3,1,figsize=(8, 12))
ax[0].hist(mPBH, bins=20, color='blue')
ax[0].set_title(r"$\bar{M}_{PBH}$ = " + f"{mPBHmean}")
ax[0].set_xlabel(r'$M_{PBH}$ ($M_\odot$)')
ax[0].set_ylabel('Frequency')
ax[1].hist(vINF, bins=20, color='red')
ax[1].set_title(r"$\bar{v}_{\infty}$ = " + f"{vINFmean}")
ax[1].set_xlabel(r'$v_{\infty}$ (km/s)')
ax[1].set_ylabel('Frequency')
ax[2].hist(ImpactB, bins=20, color='green')
ax[2].set_title(r"$\bar{b}$ = " + f"{bmean}")
ax[2].set_xlabel('b (AU)')
ax[2].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig(f"{directoryp}histogram_mC_vInf_b.png", dpi=300, bbox_inches='tight')
plt.close()

samean = '{:.3f}'.format(np.mean(SAmean))
emean = '{:.3f}'.format(np.mean(Emean))

fig, ax = plt.subplots(2,1,figsize=(8, 6))
ax[0].hist(SAmean, bins=20, color='blue')
ax[0].set_title(r"$\bar{a}$ = " + f"{samean}")
ax[0].set_xlabel('Semi-major axis (AU)')
ax[0].set_ylabel('Frequency')
ax[1].hist(Emean, bins=20, color='red')
ax[1].set_title(r"$\bar{e}$ = " + f"{(emean)}")
ax[1].set_xlabel('Eccentricity')
ax[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(f"{directoryp}histogram_a_e.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"{len(mPBH)} survived among {120**3} systems on the grid")
# for i in range(len(mPBH)):
#     print(f"System {i} with mPBH: {mPBH[i]}, vINF: {vINF[i]}, ImpactB: {ImpactB[i]}, Lifetime: {Lifetime[i]}")
#     print(f"Mean semimajor axis: {SAmean[i]}, Mean eccentricity: {Emean[i]}")
#     print(f"Total captured time to A: {dframes[i][3][4]} + {dframes[i][4][4]} = {dframes[i][3][4]+dframes[i][4][4]}, Total free time: {dframes[i][1][4]}")
#     print(f"Longest captured time: {dframes[i][3][5]} + {dframes[i][4][5]} = {dframes[i][3][5]+dframes[i][4][5]}")
#     print(f'Percentage captured to A: {100*(dframes[i][3][4]+dframes[i][4][4])/(Lifetime[i])} %')

        



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

