import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u
import os
import sys
import astropy.io.fits as fits
from datetime import datetime
import pandas as pd
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as ticker


plt.rcParams["axes.formatter.use_mathtext"]
c = datetime.now()
# Displays Time
current_time = c.strftime('%H%M')
print("current time :", current_time)
path = '/data/a.saricaoglu/repo/COMPAS'

# # Import COMPAS specific scripts
compasRootDir = os.environ['COMPAS_ROOT_DIR']
sys.path.append(compasRootDir + '/postProcessing/PythonScripts')
print(sys.path)

pathToData = '/data/a.saricaoglu/repo/COMPAS/Files/bhvscomp.fits'

mod = 'Capture/'
if not os.path.exists(path+ "/Plots/" + mod +  str(c.strftime("%m.%d")+ "/" + current_time + "/") ): 
    os.makedirs(path + "/Plots/"  + mod +  str(c.strftime("%m.%d")+ "/" + current_time + "/") ) 
directoryp = path + "/Plots/"  + mod +  str(c.strftime("%m.%d") + "/" + current_time + "/")  

mpl.rc('font',family='DeJavu Serif')
font = FontProperties(family='DeJavu Serif')
plt.rcParams['savefig.dpi'] = 600  # Set the resolution for saved figures
# Set the number of lines to plot and the colormap
n_lines = 10
cmap = mpl.colormaps['plasma']
# Take colors at regular intervals spanning the colormap.
colors = cmap(np.linspace(0, 1, n_lines))
# with fits.open(pathToData) as hdul:
#     data = hdul[1].data
#     print(data.columns)
#     comp_mass = data["Companion_mass"]
#     bh_mass = data["BlackHole_mass"]
#     comp_radius  = data["Companion_radius"]

#     systems = np.linspace(0, len(comp_mass), len(comp_mass))

#     # plt.plot(systems,comp_mass)
#     # plt.plot(systems,bh_mass)
#     # plt.plot(systems,comp_radius)
#     # plt.legend(["comp_mass", "bh_mass", "comp_radius"])
#     # plt.show()
#     binary_df = pd.DataFrame()
#     print(len(binary_df))
#     i = 0
#     cmas = []
#     bmas = []
#     crad = []
#     for c, b, r in zip(comp_mass, bh_mass, comp_radius):
#         if (c > 0) and (b > 0) and (r > 0):
#             cmas.append(c)
#             bmas.append(b)
#             crad.append(r)
#             i += 1
#     binary_df["Companion_mass"] = cmas
#     binary_df["BlackHole_mass"] = bmas
#     binary_df["Companion_radius"] = crad
# hdul.close()

# j = np.random.randint(0,i)
# print(f'Random simulated system params: companion mass = {binary_df["Companion_mass"][j]}, black hole mass = {binary_df["BlackHole_mass"][j]}, companion radius = {binary_df["Companion_radius"][j]}')


G = const.G.value
print(const.c)
G = G * const.M_sun.value * 1/(const.c.value*const.R_sun.value)**3 # in 1/c^2 
# # Regulus
# star = 'Regulus'
# M_star =  4.15 # in M_sun
# R_star = 3.22 * 1/const.c.value # in R_sun / c

# S Monocerotis
# star = 'S_Monocerotis'
# M_star =  29.1# in M_sun
# R_star = 9.9 * 1/const.c.value # in R_sun / c

# paper values
# star = 'paper'
# M_star =  2 # in M_sun
# R_star = 0.000015 * 1/const.c.value # in R_sun / c

# Beta Persei Aa1
# star = 'Beta_Persei_Aa1'
# M_star = 3.17 # in M_sun
# R_star = 2.73 * 1/const.c.value # in R_sun / c t0 0.68

# V518 Persei
star = 'V518_Persei'
M_star =  0.47 # in M_sun 0.26-0.68
M_bh = 6.5 # in M_sun 3.6 to 9.5
R_star = M_star**0.8  * 1/const.c.value # in R_sun / c

# star = 'GW200115'
# M_star =  1.25 # in M_sun 0.26-0.68
# M_bh = 7 # in M_sun 3.6 to 9.5
# R_star = 0.000015 * 1/const.c.value # in R_sun / c 

gridsize = 100

# print(R_star)
# print(R_star.to(u.au))

def p_e(e):
    return (e + 1)**-3.5 * (np.arccos(-1/e) * (24 + 73*e**2 + (37/4)*e**4 + (602+673*e**2)*np.sqrt(e**2 - 1)/12))

def eccentricity(b, M_pbh, M_star, v_inf):
    return 1 + 1e-9 * (b/(1e5 *1/const.c.value *1/const.R_sun.value))**2 * (v_inf/(2.2e5 *1/const.c.value ))**4 * (M_star)**-2

def delta_E_GW(b, M_pbh, M_star, v_inf):
    e = eccentricity(b, M_pbh, M_star, v_inf)
    pre_factor = (8/15) * (M_pbh**2 * M_star**2) / (M_pbh + M_star)**3
    return pre_factor * p_e(e) / ((e - 1)**3.5) * v_inf**7

def equation(b, M_pbh, M_star, v_inf):
    return delta_E_GW(b, M_pbh, M_star, v_inf) - (0.5 * M_pbh * v_inf**2)

def compute_b_max(M_pbh, M_star, v_inf, b_min):
    b_guess_min = 0 *1/const.c.value
    b_guess_max = 1e18 *1/const.c.value *1/const.R_sun.value
    b_max = brentq(equation, b_guess_min, b_guess_max, args=(M_pbh, M_star, v_inf))
    return b_max 

def compute_capture_crossec(b_min, b_max):
    if b_max < b_min:
        return 0
    else:
        return np.pi * (b_max - b_min)**2 
def compute_semimajor_axis(M_pbh, M_star, v_inf):

    return np.sqrt(G**2 * (M_star + M_pbh)**2 / (v_inf**4))
# def compute_semimajor_axis(M_pbh, M_star, v_inf, bmin, bmax):
#     # Calculate the semimajor axis using the formula
#     a = (G * (M_star + M_pbh) / (v_inf**2)) * (bmax - bmin)
#     return a


m_pbh_values = np.logspace(-10, 10, gridsize) # in M_sun
v_inf_values = np.logspace(-6, -2, gridsize) #in c
print(v_inf_values)  # 100 to 100,000 m/s

M_grid, V_grid = np.meshgrid(m_pbh_values, v_inf_values)
cross_section_grid = np.zeros_like(M_grid)

# Figure 1
b_max_values = np.zeros_like(M_grid)
b_min_values = np.zeros_like(M_grid)
for i in range(gridsize):
    for j in range(gridsize):
        b_min = np.sqrt(R_star**2 + (2 * G * M_star * R_star) / V_grid[i, j]**2)
        b_min_values[i, j] = b_min
        b_max = compute_b_max(M_grid[i, j], M_star, V_grid[i, j], b_min)
        b_max_values[i, j] = b_max
        cs = compute_capture_crossec(b_min, b_max)
        cross_section_grid[i, j] = cs
cross_section_grid = cross_section_grid * (const.R_sun.value**2 * const.c.value**2) * 4.46837e-23
print('crossec max', np.max(cross_section_grid))
print('b_max max', np.max(b_max_values))
print('b_min max', np.max(b_min_values))

plt.figure(figsize=(3.5,3))
plt.contourf((cross_section_grid), origin='lower', extent=[np.log10(m_pbh_values[0]),
                                                       np.log10(m_pbh_values[-1]),
                                                       np.log10(v_inf_values[0]),
                                                       np.log10(v_inf_values[-1])],
           aspect='auto', cmap='plasma', norm=plt.cm.colors.LogNorm())
# plt.hlines(np.log10(2/(3*10**4)), np.log10(m_pbh_values[0]), np.log10(m_pbh_values[-1]), color='blue', linestyle='--')
plt.vlines(np.log10(M_bh), np.log10(v_inf_values[0]), np.log10(v_inf_values[-1]), color='red', linestyle='--')
plt.xlabel(r'log10($M_{PBH}$) [$M_\odot$]')
plt.ylabel(r'log10($v_{\infty}$) [c]')
plt.text(0.5, 0.7, r'Prohibited', fontsize=10, ha='center', va='center', transform=plt.gca().transAxes)
# plt.text(0.5, 0.6, r'(Capture without collision is not possible)', fontsize=8, ha='center', va='center', transform=plt.gca().transAxes)

cbar = plt.colorbar()
cbar.set_label(r'$\sigma_{cap}^{GW}$ (AU$^2$)')
# plt.title('Capture cross-section as function of PBH mass and velocity')
plt.savefig(directoryp + f'{star}_capture_cross_section_grid.png',bbox_inches='tight')
# plt.show()

# Figure 2
plt.figure(figsize=(3.5,3.5))
for v in  v_inf_values:
    b_min= np.sqrt(R_star**2 + (2 * G * M_star * R_star) / v**2)
    crossecc = []
    for m in m_pbh_values:
        # print(m, M_star, R_star, v)
        b_max = compute_b_max(m, M_star, v, b_min) 
        crossecc.append(compute_capture_crossec(b_min, b_max)*(const.R_sun.value**2 * const.c.value**2) * 4.46837e-23)   
    plt.loglog(m_pbh_values, crossecc )
plt.xlabel(r"PBH mass ($M_{\odot}$)")
plt.ylabel(r"$\sigma_{cap}$ (m)")
plt.title(r"$\sigma_{cap}$ ")
plt.grid(True, which="both")
plt.vlines(np.log10(M_bh), np.log10(v_inf_values[0]), np.log10(v_inf_values[-1]), color='red', linestyle='--')

plt.savefig(directoryp + f'{star}_capture_cross_section.png')

# Figure 3
plt.figure(figsize=(3.5,3.5))
rate = []
dv = 1 / gridsize
for m in m_pbh_values:
    v_int = 0
    for v in  v_inf_values:
        if v < (5.5e5 *1/const.c.value):
            # print(m, M_star, R_star, v)
            b_min= np.sqrt(R_star**2 + (2 * G * M_star * R_star) / v**2)
            f = v**2 * np.exp(-v**2 /(2.2e5 *1/const.c.value)**2) 
            b_max = compute_b_max(m, M_star, v, b_min) 
            caprate = dv * f * (compute_capture_crossec(b_min, b_max)*(const.R_sun.value**2 * const.c.value**2)) * v * 3e19
            v_int = caprate + v_int
    rate.append(v_int)
rate = rate 
plt.plot(m_pbh_values, rate)
plt.xscale('log')
plt.xlabel(r"PBH mass ($M_{\odot}$)")
plt.ylabel(r"$\sigma_{cap}$ (m)")
plt.title(r"$\sigma_{cap}$ ")
plt.vlines(np.log10(M_bh), np.log10(v_inf_values[0]), np.log10(v_inf_values[-1]), color='red', linestyle='--')

plt.grid(True, which="both")
plt.savefig(directoryp + f'{star}_capture_rate.png')

semaj_grid = np.zeros_like(M_grid)
for i in range(gridsize):
    for j in range(gridsize):
        a = compute_semimajor_axis(M_grid[i, j], M_star, V_grid[i, j])
        semaj_grid[i, j] = a
semaj_grid = semaj_grid * (const.R_sun.value**2 * const.c.value**2) * 6.68459e-12
print('crossec max', np.max(cross_section_grid))
print('b_max max', np.max(b_max_values))
print('b_min max', np.max(b_min_values))

# R_star_au = np.zeros_like(M_grid)
# for i in range(m_pbh_values.size):
#     for j in range(v_inf_values.size):
#         va = np.sqrt(G * (M_star + M_grid[i,j] / R_star))
#         ra = compute_semimajor_axis(M_grid[i,j], M_star, va)
#         R_star_au[i,j] = ra
# R_star_au = R_star_au * const.R_sun.value *const.c.value *  6.68459e-12


plt.figure(figsize=(3.5,3.5))
# Create a black-and-white contour plot
contour = plt.contour(np.log10(m_pbh_values), np.log10(v_inf_values),(semaj_grid),
                      colors='black', linewidths=0.8,  norm=plt.cm.colors.LogNorm())


# Add labels to the contour lines
plt.clabel(contour, inline=True, fontsize=8, fmt=lambda x: f"{x:.1E} AU")
plt.xlabel(r'log10($M_{PBH}$) [$M_\odot$]')
plt.ylabel(r'log10($v_{\infty}$) [c]')
plt.vlines(np.log10(M_bh), np.log10(v_inf_values[0]), np.log10(v_inf_values[-1]), color='red', linestyle='--')

# cbar = plt.colorbar()
# cbar.set_label(r'$a_{semimajor}$ (AU)')
# plt.title('Capture cross-section as function of PBH mass and velocity')
plt.savefig(directoryp + f'{star}_semimajor_axis_grid.png',bbox_inches='tight')
# plt.show()
# import pandas as pd
# M, V = np.meshgrid(m_pbh_values / const.M_sun, v_inf_values)
# data = pd.DataFrame({
#     "M_pbh (M_sun)": M.flatten(),
#     "v_inf (km/s)": V.flatten(),
#     "Cross section (au^2)":cross_section_grid.flatten()
# })

# plt.figure(figsize=(3.5,3.5))
# sns.kdeplot(x=data["M_pbh (M_sun)"], y=data["v_inf (km/s)"], weights=data["Cross section (au^2)"], fill=True, cmap='plasma')

# plt.yscale('log')
# plt.xlabel("PBH mass [$M_\\odot$]")
# plt.ylabel("Velocity [km/s]")
# plt.title("Capture cross section contour plot")
# plt.show()
plt.figure(figsize=(3.5, 3))
v_inf_valuess = [v.to(u.km/u.s).value for v in (v_inf_values * const.c)]

plt.contourf((cross_section_grid), origin='lower', extent=[np.log10(m_pbh_values[0]),
                                                       np.log10(m_pbh_values[-1]),
                                                       np.log10(v_inf_values[0]),
                                                       np.log10(v_inf_values[-1])],
           aspect='auto', cmap='plasma', norm=plt.cm.colors.LogNorm())
# plt.hlines(np.log10(2/(3*10**4)), np.log10(m_pbh_values[0]), np.log10(m_pbh_values[-1]), color='blue', linestyle='--')
cbar = plt.colorbar()
cbar.set_label(r'$\sigma_{cap}^{GW}$ (AU$^2$)')
contour = plt.contour(np.log10(m_pbh_values), np.log10(v_inf_values),(semaj_grid),
                      colors='black', linewidths=0.8,  norm=plt.cm.colors.LogNorm(),
                      linestyles='--', alpha=0.5)
a = compute_semimajor_axis(M_bh, M_star, v_inf_values)
# semaj_grid = np.zeros_like(M_grid)
# for i in range(gridsize):
#     for j in range(gridsize):
#         a = compute_semimajor_axis(M_bh, M_star, V_grid[i, j])
#         semaj_grid[i, j] = a
# semaj_grid = semaj_grid * (const.R_sun.value**2 * const.c.value**2) * 6.68459e-12


# contour2 = plt.contour(np.log10(m_pbh_values), np.log10(v_inf_values),(R_star_au),
#                       colors='blue', linewidths=0.8,  norm=plt.cm.colors.LogNorm(),
#                       linestyles='--', alpha=0.5)
plt.vlines(np.log10(M_bh), np.log10(v_inf_values[0]), np.log10(v_inf_values[-1]), color='red', linestyle='--', alpha=0.5)
# Add labels to the contour lines
plt.clabel(contour, inline=True, fontsize=8, fmt=lambda x: f"{x:.1g} AU")
# plt.contour(np.log10(m_pbh_values), np.log10(v_inf_values),(semaj_grid),
#                         colors='blue', linewidths=0.8,  norm=plt.cm.colors.LogNorm(),
#                         linestyles='--', alpha=0.5)
# Add labels to the contour lines
# plt.clabel(contour, inline=True, fontsize=8, fmt=lambda x: f"{x:.1E} AU")
# plt.clabel(contour2, inline=True, fontsize=8, fmt=lambda x: f"R_star {x:.1E} AU")
plt.xlabel(r'log10($M_{PBH}$) [$M_\odot$]')
plt.ylabel(r'$v_{\infty}$ [km/s]')
plt.text(0.2, 0.6, r'Prohibited', fontsize=10, ha='center', va='center', transform=plt.gca().transAxes)
# plt.text(0.5, 0.6, r'(Capture without collision is not possible)', fontsize=10, ha='center', va='center', transform=plt.gca().transAxes)
plt.yticks(np.log10(v_inf_values)[::10], [f'{x:.1g}' for x in (v_inf_valuess)[::10]])# plt.title('Capture cross-section as function of PBH mass and velocity')
plt.savefig(directoryp + f'{star}_capture_cross_section_plus_semaj_grid.png',bbox_inches='tight')
# plt.show()