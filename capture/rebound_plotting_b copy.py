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
from operator import itemgetter
from itertools import groupby
import corner


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

def advanced_stats(window, std_thresh, grad_thresh, SA, T):
    if SA.dtype.byteorder == '>':  # Big-endian
        SA = SA.byteswap().newbyteorder()

    rolling_std = pd.Series(SA).rolling(window=window, center=True).std()
    grad = np.abs(np.gradient(SA, T))
    
    stable_mask = (rolling_std < std_thresh) & (grad < grad_thresh)
    return stable_mask


def find_longest_stable(stable_mask, SA, T):
    stable_indices = np.where(stable_mask)[0]

    # Group into contiguous stable segments
    segments = []
    for k, g in groupby(enumerate(stable_indices), lambda x: x[0] - x[1]):
        group = list(map(itemgetter(1), g))
        segments.append(group)

    if not segments:
        return None, None, None, None  # no stable region found

    # Find the longest one
    longest_segment = max(segments, key=len)
    longest_mask = np.zeros_like(SA, dtype=bool)
    longest_mask[longest_segment] = True

    # Stats
    start_time = T[longest_segment[0]]
    end_time = T[longest_segment[-1]]
    mean_a = np.mean(SA[longest_segment])

    return longest_mask, mean_a, start_time, end_time

def get_stable_regions(stable_mask, T):
    """
    Analyze stable mask to find all stable regions, their durations, and stats.
    
    Parameters:
        stable_mask: boolean array, True where signal is stable
        T: time array (same length)

    Returns:
        total_duration: sum of durations of all stable segments
        longest_duration: duration of the longest stable segment
        all_segments: list of (start_time, end_time, duration)
    """
    indices = np.where(stable_mask)[0]

    if len(indices) == 0:
        return 0, 0, []

    segments = []
    for k, g in groupby(enumerate(indices), lambda x: x[0]-x[1]):
        group = list(map(itemgetter(1), g))
        start, end = group[0], group[-1]
        duration = T[end] - T[start]
        segments.append((T[start], T[end], duration))

    durations = [d for _, _, d in segments]
    total_duration = np.sum(durations)
    longest_duration = np.max(durations)

    return total_duration, longest_duration, segments
primary_folder = f'04.21/1609/McVinfImpB_Variation/rebound_sim_mp_b.py_1609/'
secondary_folder = '04.21/rebound_analysis_b_2007/McVinfImpB_Variation/rebound_analysis_b_general_.fits'

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
window = int(1e3)
std_threshA = 1
grad_threshA = 0.1
std_threshE = 0.01
grad_threshE = 0.001

SAmean_stable = []
Emean_stable = []
SAmean_stable_longest = []
Emean_stable_longest = []
SAmean_aLimit = []
Emean_aLimit = []
mPBH_aLimit = []
vINF_aLimit = []
B_aLimit = []
SAmean_stable_aLimit = []
Emean_stable_aLimit = []
mPBH_stable_aLimit = []
vINF_stable_aLimit = []
B_stable_aLimit = []


for file in runs :
    print(f"Processing file:{directoryf}{primary_folder}/{file}")
    syst = file.split('_')[1].split('.')[0]
    # if os.path.exists(f"{directoryp}Sys_{syst}/"):
    #     print(f"System {syst} already exists, skipping...")
    #     continue
    
    try:
        with fits.open(f'{directoryf}{primary_folder}/{file}') as hdul:
            #hdul.info()
            data = hdul[1].data
            #print(data)
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
            #print(f"snapshot interval:{snpInt} opB: {opB}, final year:{fYr}, r_close: {r_close}, v_inf: {v_inf}, v1primme: {v1prime}, bmin: {bmin}, bmax: {bmax}, initial_BC_distance: {initial_BC_distance}, mA: {mA}, mB: {mB}, mC: {mC}, rB: {rB}")

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
    except:
        print(f"Error reading file {file}, skipping.")
        continue

    if np.max(SA) < 120:
        SAmean_aLimit.append(np.mean(SA))
        Emean_aLimit.append(np.mean(E))
        mPBH_aLimit.append(mC)
        vINF_aLimit.append(v_inf)
        B_aLimit.append((v_inf* (u.au/u.yr)).to(u.km/u.s).value)
        


    os.makedirs(f"{directoryp}Sys_{syst}/", exist_ok=True)
    print(f'directory created: {directoryp}Sys_{syst}/')
    mc = '{:.3E}'.format(mC)
    vinf = '{:.3f}'.format(v_inf)
    fyr = '{:.3E}'.format(fYr)
    SAmean_aLimit.append(np.mean(SA))
    Emean_aLimit.append(np.mean(E))
    plt.rcParams.update({'font.size': 12})
    # --- Rolling std dev and gradient
    # Load and fix byte order
    stable_mask = advanced_stats(window,std_threshA, grad_threshA, SA,T)
    print(stable_mask)
    mean_stable_a = np.mean(SA[stable_mask])
    stable_indices = np.where(stable_mask)[0]
    total_duration, longest_duration, segments = get_stable_regions(stable_mask, T)

    # print(f"Minimum stable years: {min_stable_years}")
    result = find_longest_stable(stable_mask, SA, T)

    if result[0] is not None:
        longest_mask, mean_vala, t_start, t_end = result
        print(f"Longest stable region: t = {t_start:.2f} to {t_end:.2f}, mean a = {mean_vala:.3f} AU")
        SAmean_stable.append(mean_stable_a)
        SAmean_stable_longest.append(mean_vala)

        if np.max(SA) < 120:
            SAmean_stable_aLimit.append(mean_stable_a)
            Emean_stable_aLimit.append(np.mean(E))
            mPBH_stable_aLimit.append(mC)
            vINF_stable_aLimit.append(v_inf)
            B_stable_aLimit.append((v_inf* (u.au/u.yr)).to(u.km/u.s).value)


        # plt.figure(figsize=(7, 3.5))
        # plt.plot(T, SA, label='Semi-major axis', color='blue')
        # plt.fill_between(T, SA.min(), SA.max(), where=longest_mask,
        #                 color='red', alpha=1, label='Longest stable region', hatch='//')
        # plt.fill_between(T, SA.min(), SA.max(), where=stable_mask,
        #                 color='green', alpha=1, label='All stable regions')
        # plt.xlabel("Time")
        # plt.ylabel("Semi-major axis (AU)")
        # plt.legend()
        # plt.title(r"$\bar{a}_{stable}$" + f"={mean_stable_a:.3f}AU" +r"  $\bar{a}_{stable,longest}$" + f"={mean_vala:.3f}AU" + r"  $\tau_{stable,max}$" + f"={longest_duration:.2f} yr" +r"  $\tau_{cap}$" + f"={total_duration:.2f} yr")
        # plt.tight_layout()
        # plt.savefig(f"{directoryp}Sys_{syst}/"+ f'SunJptr_a_stable_region_{syst}.png', dpi=200, bbox_inches='tight')
        # plt.close()
    else:
        print("No stable region found.")


    # Load and fix byte order

    stable_mask = advanced_stats(window,std_threshE, grad_threshE ,E,T)
    mean_stable_e = np.mean(E[stable_mask])
    stable_indices = np.where(stable_mask)[0]

    result = find_longest_stable(stable_mask, E, T)
    total_duration, longest_duration, segments = get_stable_regions(stable_mask, T)

    if result[0] is not None:
        longest_mask, mean_vale, t_start, t_end = result
        print(f"Longest stable region: t = {t_start:.2f} to {t_end:.2f}, mean e  = {mean_vale:.3f} ")
        Emean_stable.append(mean_stable_e)
        Emean_stable_longest.append(mean_vale)
        # --- Plotting
        # plt.figure(figsize=(7, 3.5))
        # plt.plot(T, E, label='Semi-major axis', color='blue')
        # plt.fill_between(T, E.min(), E.max(), where=longest_mask,
        #                 color='red', alpha=0.3, label='Longest table region')
        # plt.fill_between(T, E.min(), E.max(), where=stable_mask,
        #                 color='green', alpha=0.3, label='All stable regions')
        # plt.xlabel("Time")
        # plt.ylabel("Eccentricity")
        # plt.title(r"$\bar{e}_{stable}$" + f"={mean_stable_e:.3f}" +r"  $\bar{e}_{stable,longest}$" + f"={mean_vale:.3f}" +r"  $\tau_{stable,longest}$" + f"={longest_duration:.2f} yr" +r"  $\tau_{cap}$" + f"={total_duration:.2f} yr")
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(f"{directoryp}Sys_{syst}/"+ f'SunJptr_e_stable_region_{syst}.png', dpi=200, bbox_inches='tight')
        # plt.close()
    else:
        print("No stable region found.")

    if np.max(SA) > 120:
        print(f"System {syst} has semaj > 120,  skipping plots...")

    fig, ax = plt.subplots(2,1,figsize=(8,12), sharex=True)
    plt.rcParams.update({'font.size': 20})
    samean = '{:.3f}'.format(np.mean(SA))
    emean = '{:.3f}'.format(np.mean(E))
    beta = '{:.3f}'.format(beta)
    lambda1 = '{:.3f}'.format(lambda1)
    ab = '{:.3f}'.format(b)
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

    # fig, ax = plt.subplots(figsize=(7, 7))
    # #plt.rcParams.update({'font.size': 20})
    # plotInt = int((1/snpInt) * snpInt) * 5
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
    # ax.set_xlim(np.min(x), np.max(x))
    # ax.set_ylim(np.min(y), np.max(y))
    # plt.title(fr'$M_C$ = {mc}, $v_inf$= {vinf}, aB = {ab}, $\tau$ = {fyr}')
    # plt.savefig(f"{directoryp}Sys_{syst}/"+ f'SunJptr_capture_{syst}.png', dpi=200)
    # plt.tight_layout()
    # plt.close()

    # fig, ax = plt.subplots(figsize=(25,4), dpi=150)
    # time = np.arange(0, len(S['Free']), 1)
    # plt.rcParams.update({'font.size': 20})
    # ax.bar(time, S['Free'], label=f'Free ({S_info['F'][4]})', color='blue', alpha=0.5 )
    # ax.bar(time, S['Bcapture'],label=f'Bcapture ({S_info['Bcap'][4]})', color='red', alpha=0.5)
    # ax.bar(time, S['Acapture'], label=f'Acapture ({S_info['Acap'][4]})', color='green', alpha=0.5)
    # ax.bar(time, S['ABcapture'], label=f'ABcapture ({S_info['Abcap'][4]})', color='orange', alpha=0.5)
    # # ax.set_xlim(0,50)
    # ax.set_xlabel('Time (yr)')
    # ax.legend(loc='upper right')
    # plt.savefig(f"{directoryp}Sys_{syst}/"+ f'stats_{syst}.png', bbox_inches='tight')
    # plt.close()


with fits.open("/data/a.saricaoglu/repo/COMPAS/Files/Capture/04.21/rebound_analysis_b_2007/McVinfImpB_Variation/rebound_analysis_b_general_.fits") as hdul:
    data = hdul[1].data
    mPBH = data["mPBH"]
    vINF = data["vINF"]
    ImpactB = data["ImpactB"]
    Lifetime = data["Lifetime"]
    SAmean = data["SAmean"]
    Emean = data["Emean"]
hdul.close()
SAmean = [m for m in SAmean if m > 0]
SAmean_stable = [m for m in SAmean_stable if m > 0]
SAmean_aLimit = [m for m in SAmean_aLimit if m > 0]
print(f"mPBH: {mPBH}")
vINF = [v * (u.au/u.yr) for v in vINF]
vINF = [v.to(u.km/u.s).value for v in vINF]
# print(f"SAmean minimum: {np.min([m for m in SAmean if m > 0])}")
# print(f"SAmean stable minimum: {np.min([m for m in SAmean_stable if m > 0])}")
# print(f"SAmean a limit mean: {np.mean(SAmean_aLimit)}")
# print(f"SAmean a limit median: {np.median(SAmean_aLimit)}")
# print(f"ebarabar = {1/2*(1 + (1 - 5.2/np.mean(SAmean_aLimit)))}")

# fig = plt.figure(figsize=(10.5, 10.5))
# #plt.rcParams.update({'font.size': 18})
# ax = fig.add_subplot(111, projection='3d')
# plot = ax.scatter((mPBH), vINF, ImpactB, c=Lifetime, marker='o', s=100)
# plt.colorbar(plot, label='Lifetime (yr)', shrink=0.6, pad=0.2)
# # ax.set_xticks(np.logspace(-13, -4, 6), [f'{x:.2f}' for x in np.logspace(-13, -4, 6)])
# ax.set_yticks(np.linspace(0.2108, 21.08, 6), [f'{x:.2f}' for x in np.linspace(1, 100, 6)])
# ax.set_zticks(np.linspace(np.min(ImpactB), np.max(ImpactB), 6), [f'{x:.3f}' for x in np.linspace(np.min(ImpactB), np.max(ImpactB), 6)])
# ax.set_xlabel(r'log10($M_{PBH}$) ($M_\odot$)', labelpad=20)
# ax.set_ylabel(r'$\infty$ (km/s)', labelpad=20)
# ax.set_zlabel("b (AU)", labelpad=20)
# plt.savefig(f"{directoryp}mC_vInf_b_values.png", dpi=300)
# plt.close()

# mPBHmean = '{:.3E}'.format(np.mean((mPBH)))
# vINFmean = '{:.3E}'.format(np.mean(vINF))
# bmean = '{:.3E}'.format(np.mean(ImpactB))


# fig, ax = plt.subplots(3,1,figsize=(7, 7))
# counts, bin_edges = np.histogram(mPBH, bins=20)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# ax[0].hist(mPBH, bins=20, color='blue')
# ax[0].set_title(r"$\bar{M}_{B}$ = " + f"{mPBHmean}")
# ax[0].set_xlabel(r'log10($M_{B}$) ($M_\odot$)')
# ax[0].set_ylabel('Number of systems')
# ax[0].set_xticks(bin_centers)  # Set x-ticks at bin centers
# ax[0].set_yscale('log')  # Set y-axis to logarithmic scale
# counts, bin_edges = np.histogram(vINF, bins=20)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# ax[1].hist(vINF, bins=20, color='red')
# ax[1].set_title(r"$\bar{v}_{\infty}$ = " + f"{vINFmean}")
# ax[1].set_xlabel(r'$v_{\infty}$ (km/s)')
# ax[1].set_ylabel('Number of systems')
# ax[1].set_xticks(bin_centers)  # Set x-ticks at bin centers
# ax[1].set_yscale('log')  # Set y-axis to logarithmic scale

# counts, bin_edges = np.histogram(ImpactB, bins=20)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# ax[2].hist(ImpactB, bins=20, color='green')
# ax[2].set_title(r"$\bar{a_B}$ = " + f"{bmean}")
# ax[2].set_xlabel(r"$a_B$ (AU)")
# ax[2].set_ylabel('Number of systems')
# ax[2].set_xticks(bin_centers)  # Set x-ticks at bin centers
# ax[2].set_yscale('log')  # Set y-axis to logarithmic scale

# # Rotate x-tick labels for all subplots
# ax[0].tick_params(axis='x', rotation=45)  # Rotate x-ticks by 45 degrees for the first subplot
# ax[1].tick_params(axis='x', rotation=45)  # Rotate x-ticks by 45 degrees for the second subplot
# ax[2].tick_params(axis='x', rotation=45)  # Rotate x-ticks by 45 degrees for the third subplot
# plt.tight_layout()
# plt.savefig(f"{directoryp}histogram_mC_vInf_b.png", dpi=300, bbox_inches='tight')
# plt.close()

# samean = '{:.3f}'.format(np.mean(SAmean))
# emean = '{:.3f}'.format(np.mean(Emean))
# samedian = '{:.3f}'.format(np.median(SAmean))
# emedian = '{:.3f}'.format(np.median(Emean))

# fig, ax = plt.subplots(2,1,figsize=(7, 7))
# counts, bin_edges = np.histogram(SAmean, bins=20)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# ax[0].hist(SAmean, bins=20, color='blue')
# ax[0].set_title(r"$\bar{a}$ = " + f"{samean}" + r"$a_d$ = " + f"{samedian}")
# ax[0].set_xlabel('Semi-major axis (AU)')
# ax[0].set_ylabel('Number of systems')
# ax[0].set_xticks(bin_centers)  # Set x-ticks at bin centers
# counts, bin_edges = np.histogram(Emean, bins=20)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# ax[1].hist(Emean, bins=20, color='red')
# ax[1].set_title(r"$\bar{e}$ = " + f"{(emean)}" + r"$e_d$ = " + f"{(emedian)}")
# ax[1].set_xlabel('Eccentricity')
# ax[1].set_ylabel('Number of systems')
# ax[1].set_xticks(bin_centers)  # Set x-ticks at bin centers
# # Rotate x-tick labels for all subplots
# ax[0].tick_params(axis='x', rotation=45)  # Rotate x-ticks by 45 degrees for the first subplot
# ax[1].tick_params(axis='x', rotation=45)  # Rotate x-ticks by 45 degrees for the second subplot
# plt.tight_layout()
# plt.savefig(f"{directoryp}histogram_a_e.png", dpi=300, bbox_inches='tight')
# plt.close()

# samean_stable = '{:.3f}'.format(np.mean(SAmean_stable))
# emean_stable = '{:.3f}'.format(np.mean(Emean_stable))
# samedian_stable = '{:.3f}'.format(np.median(SAmean_stable))
# emedian_stable = '{:.3f}'.format(np.median(Emean_stable))

# fig, ax = plt.subplots(2,1,figsize=(7, 7))
# counts, bin_edges = np.histogram(SAmean_stable, bins=20)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# plt.suptitle(f'Treshold values for stable region: std = {std_threshA,std_threshE}, grad = {grad_threshA, std_threshE} window = {window}')
# ax[0].hist(SAmean_stable, bins=20, color='blue')
# ax[0].set_title(r"$\bar{a}$ = " + f"{samean_stable} " + r"$a_d$ = " + f"{samedian_stable}" + f" {len(SAmean_stable)} systems are stable out of {len(SAmean)} ")
# ax[0].set_xlabel('Semi-major axis (AU)')
# ax[0].set_ylabel('Number of systems')
# ax[0].set_xticks(bin_centers)  # Set x-ticks at bin centers
# counts, bin_edges = np.histogram(Emean_stable, bins=20)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# ax[1].hist(Emean_stable, bins=20, color='red')
# ax[1].set_title(r"$\bar{e}$ = " + f"{(emean_stable)} " + r"$e_d$ = " + f"{(emedian_stable)}" + f" {len(Emean_stable)} systems are stable out of {len(Emean)}")
# ax[1].set_xlabel('Eccentricity')
# ax[1].set_ylabel('Number of systems')
# ax[1].set_xticks(bin_centers)  # Set x-ticks at bin centers
# # Rotate x-tick labels for all subplots
# ax[0].tick_params(axis='x', rotation=45)  # Rotate x-ticks by 45 degrees for the first subplot
# ax[1].tick_params(axis='x', rotation=45)  # Rotate x-ticks by 45 degrees for the second subplot
# plt.tight_layout()
# plt.savefig(f"{directoryp}histogram_a_e_stable.png", dpi=300, bbox_inches='tight')
# plt.close()

# samean_stable_longest = '{:.3f}'.format(np.mean(SAmean_stable_longest))
# emean_stable_longest = '{:.3f}'.format(np.mean(Emean_stable_longest))
# samedian_stable_longest = '{:.3f}'.format(np.median(SAmean_stable_longest))
# emedian_stable_longest = '{:.3f}'.format(np.median(Emean_stable_longest))
# # First histogram (Stable Longest)
# fig, ax = plt.subplots(2, 1, figsize=(7, 7))

# # Semi-major axis histogram
# counts, bin_edges = np.histogram(SAmean_stable_longest, bins=20)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# ax[0].hist(SAmean_stable_longest, bins=20, color='blue')
# ax[0].set_title(r"$\bar{a}$ = " + f"{samean_stable_longest} " + r"$a_d$ = " + f"{samedian_stable_longest}" + f" {len(SAmean_stable_longest)} systems are stable out of {len(SAmean)} ")
# ax[0].set_xlabel('Semi-major axis (AU)')
# ax[0].set_ylabel('Number of systems')
# ax[0].set_xticks(bin_centers)  # Set x-ticks at bin centers

# # Eccentricity histogram
# counts, bin_edges = np.histogram(Emean_stable_longest, bins=20)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# ax[1].hist(Emean_stable_longest, bins=20, color='red')
# ax[1].set_title(r"$\bar{e}$ = " + f"{(emean_stable_longest)} " + r"$e_d$ = " + f"{(emedian_stable_longest)}" + f" {len(Emean_stable_longest)} systems are stable out of {len(Emean)}")
# ax[1].set_xlabel('Eccentricity')
# ax[1].set_ylabel('Number of systems')
# ax[1].set_xticks(bin_centers)  # Set x-ticks at bin centers

# plt.tight_layout()
# plt.savefig(f"{directoryp}histogram_a_e_stable_longest.png", dpi=300, bbox_inches='tight')
# plt.close()
# samean_aLimit = '{:.3f}'.format(np.mean(SAmean_aLimit))
# emean_aLimit = '{:.3f}'.format(np.mean(Emean_aLimit))
# samedian_aLimit = '{:.3f}'.format(np.median(SAmean_aLimit))
# emedian_aLimit = '{:.3f}'.format(np.median(Emean_aLimit))
# # Second histogram (aLimit)
# fig, ax = plt.subplots(2, 1, figsize=(7, 7))

# # Semi-major axis histogram
# counts, bin_edges = np.histogram(SAmean_aLimit, bins=20)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# ax[0].hist(SAmean_aLimit, bins=20, color='blue')
# ax[0].set_title(r"$\bar{a}$ = " + f"{samean_aLimit}" + r"$a_d$ = " + f"{samedian_aLimit}")
# ax[0].set_xlabel('Semi-major axis (AU)')
# ax[0].set_ylabel('Number of systems')
# ax[0].set_xticks(bin_centers)  # Set x-ticks at bin centers

# # Eccentricity histogram
# counts, bin_edges = np.histogram(Emean_aLimit, bins=20)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# ax[1].hist(Emean_aLimit, bins=20, color='red')
# ax[1].set_title(r"$\bar{e}$ = " + f"{(emean_aLimit)}" + r"$e_d$ = " + f"{(emedian_aLimit)}")
# ax[1].set_xlabel('Eccentricity')
# ax[1].set_ylabel('Number of systems')
# ax[1].set_xticks(bin_centers)  # Set x-ticks at bin centers
# # Rotate x-tick labels for all subplots
# ax[0].tick_params(axis='x', rotation=45)  # Rotate x-ticks by 45 degrees for the first subplot
# ax[1].tick_params(axis='x', rotation=45)  # Rotate x-ticks by 45 degrees for the second subplot
# plt.tight_layout()
# plt.savefig(f"{directoryp}histogram_a_e_aLimit.png", dpi=300, bbox_inches='tight')
# plt.close()

plt.rcParams['lines.markersize'] = 50
# Prepare the data array with mPBH, vINF, ImpactB, SAmean, and Emean
data = np.vstack([mPBH, vINF, vINF, SAmean, Emean]).T  # Transpose to make each column a variable
# Create the corner plot
figure = corner.corner(
    data,
    labels=[
        r"$\beta$ (rad)",  # mPBH
        r"$\lambda$ (rad)",  # vINF
        r"v$_{\infty}$ (km/s)",  # ImpactB
        r"$\bar{a}$ (AU)",  # SAmean
        r"$\bar{e}$",  # Emean
    ],
    range=[(-np.pi, np.pi), (0, 2 * np.pi), (1, 100), (0, 200), (0, 1)],
    quantiles=[0.16, 0.5, 0.84],  # Show 16th, 50th, and 84th percentiles
    show_titles=True,  # Show titles with statistics
    title_fmt=".2f",  # Format for the titles
    title_kwargs={"fontsize": 12},  # Font size for titles
    label_kwargs={"fontsize": 14},  # Font size for axis labels
    plot_contours=False,  # Show contours
    plot_density=True,  # Show density plots
    figsize=(3.5, 3.5),  # Set figure size
    scatter_kwargs={'s': 100},
    plot_datapoints=True,  # Set figure size
)

# Save the plot
plt.savefig(f"{directoryp}corner_plot_beta_lambda_impactb_samean_emean.png", dpi=300)
plt.close()

# mPBH_aLimit = mPBH_aLimit.reshape([6, 5])
# vINF_aLimit = vINF_aLimit.reshape([6, 5])
# B_aLimit = B_aLimit.reshape([6, 5])
# SAmean_aLimit = SAmean_aLimit.reshape([6, 5])
# Emean_aLimit = Emean_aLimit.reshape([6, 5])

data = np.vstack([mPBH_aLimit, vINF_aLimit, B_aLimit, SAmean_aLimit, Emean_aLimit]).T  # Transpose to make each column a variable
figure = corner.corner(
    data,
    labels=[
        r"$\beta$ (rad)",  # mPBH
        r"$\lambda$ (rad)",  # vINF
        r"v$_{\infty}$ (km/s)",  # ImpactB
        r"$\bar{a}$ (AU)",  # SAmean
        r"$\bar{e}$",  # Emean
    ],
    range=[(-np.pi, np.pi), (0, 2 * np.pi), (1,100), (0, 120), (0, 1)],
    quantiles=[0.16, 0.5, 0.84],  # Show 16th, 50th, and 84th percentiles
    show_titles=True,  # Show titles with statistics
    title_fmt=".2f",  # Format for the titles
    title_kwargs={"fontsize": 12},  # Font size for titles
    label_kwargs={"fontsize": 14},  # Font size for axis labels
    plot_contours=False,  # Show contours
    plot_density=True,  # Show density plots
    figsize=(3.5, 3.5),  # Set figure size
    scatter_kwargs={'s': 100},
    plot_datapoints=True,  # Show data points
)
# Save the plot
plt.savefig(f"{directoryp}corner_plot_beta_lambda_impactb_samean_emean_aLimit.png", dpi=300)
plt.close()

# mPBH_stable_aLimit = mPBH_stable_aLimit.reshape([6, 5])
# vINF_stable_aLimit = vINF_stable_aLimit.reshape([6, 5])
# B_stable_aLimit = B_stable_aLimit.reshape([6, 5])
# SAmean_stable_aLimit = SAmean_stable_aLimit.reshape([6, 5])
# Emean_stable_aLimit = Emean_stable_aLimit.reshape([6, 5])

data = np.vstack([mPBH_stable_aLimit, vINF_stable_aLimit, B_stable_aLimit, SAmean_stable_aLimit, Emean_stable_aLimit]).T  # Transpose to make each column a variable
figure = corner.corner(
    data,
    labels=[
        r"$\beta$ (rad)",  # mPBH
        r"$\lambda$ (rad)",  # vINF
        r"v$_{\infty}$ (km/s)",  # ImpactB
        r"$\bar{a}$ (AU)",  # SAmean
        r"$\bar{e}$",  # Emean
    ],
    range=[(-np.pi, np.pi), (0, 2 * np.pi), (1, 100), (0, 120), (0, 1)],
    # Set the range for each variable
    quantiles=[0.16, 0.5, 0.84],  # Show 16th, 50th, and 84th percentiles
    show_titles=True,  # Show titles with statistics
    title_fmt=".2f",  # Format for the titles
    title_kwargs={"fontsize": 12},  # Font size for titles
    label_kwargs={"fontsize": 14},  # Font size for axis labels
    plot_contours=False,  # Show contours
    plot_density=True,  # Show density plots
    figsize=(3.5, 3.5),  # Set figure size
    scatter_kwargs={'s': 100},
    plot_datapoints=True,
)
# Save the plot
plt.savefig(f"{directoryp}corner_plot_beta_lambda_impactb_samean_emean_stable_aLimit.png", dpi=300)
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

