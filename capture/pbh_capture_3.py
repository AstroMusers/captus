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


G = const.G
print(G)

# Regulus
star = 'Regulus'
M_star =  4.15 * const.M_sun
R_star = 3.22 * const.R_sun

# S Monocerotis
# star = 'S_Monocerotis'
# M_star =  29.1 * const.M_sun
# R_star = 9.9 * const.R_sun

# paper values
star = 'paper'
M_star =  2 * const.M_sun
R_star = 0.000015 * const.R_sun

R_star = R_star.to(u.m) 
gridsize = 100

print(R_star)
print(R_star.to(u.au))

def p_e(e):
    return (e + 1)**-3.5 * (np.arccos(-1/e) * (24 + 73*e**2 + (37/4)*e**4 + (602+673*e**2)*np.sqrt(e**2 - 1)/12))

def eccentricity(b, M_pbh, M_star, v_inf):
    return 1 + 1e-9 * (b/(1e5))**2 * (v_inf/(2.2e5))**4 * (M_star/const.M_sun.value)**-2

def delta_E_GW(b, M_pbh, M_star, v_inf):
    e = eccentricity(b, M_pbh, M_star, v_inf)
    pre_factor = (8/15) * (M_pbh**2 * M_star**2) / (M_pbh + M_star)**3
    return pre_factor * p_e(e) / ((e - 1)**3.5) * v_inf**7

def equation(b, M_pbh, M_star, v_inf):
    return delta_E_GW(b, M_pbh, M_star, v_inf) - (0.5 * M_pbh * v_inf**2)

def compute_b_max(M_pbh, M_star, v_inf, b_min):
    b_guess_min = 0.1 * u.m
    b_guess_max = 1e18 * u.m
    b_max = brentq(equation, b_guess_min.value, b_guess_max.value, args=(M_pbh.value, M_star.value, v_inf.value))
    return b_max * u.m

def compute_capture_crossec(b_min, b_max):
    return np.pi * (b_max.value - b_min.value)**2 

m_pbh_values = np.logspace(-10, 4, gridsize) * const.M_sun
v_inf_values = np.logspace(3, 6, gridsize) * u.m / u.s  # 100 to 100,000 m/s

M_grid, V_grid = np.meshgrid(m_pbh_values, v_inf_values)
cross_section_grid = np.zeros_like(M_grid.value)

# Figure 1
b_max_values = np.zeros_like(M_grid.value)
b_min_values = np.zeros_like(M_grid.value)
for i in range(gridsize):
    for j in range(gridsize):
        b_min = np.sqrt(R_star**2 + (2 * G * M_star * R_star) / V_grid[i, j]**2)
        b_min_values[i, j] = b_min.to(u.au).value
        b_max = compute_b_max(M_grid[i, j], M_star, V_grid[i, j], b_min)
        b_max_values[i, j] = b_max.to(u.au).value
        cs = compute_capture_crossec(b_min.to(u.au), b_max.to(u.au))
        cross_section_grid[i, j] = cs
print('crossec max', np.max(cross_section_grid))
print('b_max max', np.max(b_max_values))
print('b_min max', np.max(b_min_values))

plt.figure(figsize=(8,6))
plt.imshow((cross_section_grid), origin='lower', extent=[np.log10(m_pbh_values[0].value/const.M_sun.value),
                                                       np.log10(m_pbh_values[-1].value/const.M_sun.value),
                                                       np.log10(v_inf_values[0].to(u.km/u.s).value),
                                                       np.log10(v_inf_values[-1].to(u.km/u.s).value)],
           aspect='auto', cmap='plasma', norm=plt.cm.colors.LogNorm())

plt.xlabel(r'log$_{10}$(PBH mass $M_\odot$)')
plt.ylabel('Velocity (km/s)')
cbar = plt.colorbar()
cbar.set_label('Capture cross-section (AU$^2$)')
plt.title('Capture cross-section as function of PBH mass and velocity')
plt.savefig(directoryp + f'{star}_capture_cross_section_grid.png', )

# Figure 2
for v in  v_inf_values:
    b_min= np.sqrt(R_star**2 + (2 * G * M_star * R_star) / v**2)
    crossecc = []
    for m in m_pbh_values:
        # print(m, M_star, R_star, v)
        b_max = compute_b_max(m, M_star, v, b_min) 
        crossecc.append(compute_capture_crossec(b_min.to(u.au), b_max.to(u.au)))   
    plt.loglog(m_pbh_values/const.M_sun.value, crossecc)
plt.xlabel(r"PBH mass ($M_{\odot}$)")
plt.ylabel(r"$\sigma_{cap}$ (m)")
plt.title(r"$\sigma_{cap}$ ")
plt.grid(True, which="both")
plt.savefig(directoryp + f'{star}_capture_cross_section.png')


# import seaborn as sns
# import pandas as pd
# M, V = np.meshgrid(m_pbh_values.value / const.M_sun.value, v_inf_values.value)
# data = pd.DataFrame({
#     "M_pbh (M_sun)": M.flatten(),
#     "v_inf (km/s)": V.flatten(),
#     "Cross section (au^2)":cross_section_grid.flatten()
# })

# plt.figure(figsize=(8,6))
# sns.kdeplot(x=data["M_pbh (M_sun)"], y=data["v_inf (km/s)"], weights=data["Cross section (au^2)"], fill=True, cmap='plasma')

# plt.yscale('log')
# plt.xlabel("PBH mass [$M_\\odot$]")
# plt.ylabel("Velocity [km/s]")
# plt.title("Capture cross section contour plot")
# plt.show()
