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


G = const.G.value
print(const.c)
# G = G * 1/const.c.value**2 # in 1/c^2 

# mA = 1
# muA = G * mA
# mB = 2
# muB = G * mB
# M_PBH = 1e-13 
# rAB = 5.2 *1/const.R_sun.value
# vB = 13700 *1/const.c.value
# abar = 16*1/const.R_sun.value
# kappa = 25
# v_inf = 7994 *1/const.c.value
G = G  # in 1/c^2 

mA = const.M_sun.value
muA = G * mA
mB = 0.0009543 * const.M_sun.value
muB = G * mB
M_PBH = 1e-13 * const.M_sun.value
rAB = (5.2*u.au).to(u.m).value
vB = 13732.50848563536
abar = (10*u.au).to(u.m).value
kappa = 25
v_inf = 20 * 1000


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
    
def compute_r_close(mA, mB, epsilon, rAB):
    return rAB * (mB*epsilon / (mA))**(1/3)
    
def compute_v1(muA, muB, v_inf, rAB, r_close):
    return np.sqrt(v_inf**2 + 2*muA/rAB + 2*muB/r_close)

def compute_v_esc(muA, rAB):
    return np.sqrt(2*muA/rAB)

def compute_capture_crossec_threebody(muA, muB, v_inf, rAB, epsilon):
    r_close = compute_r_close(muA, muB, epsilon, rAB)
    v1 = compute_v1(muA, muB, v_inf, rAB, r_close)
    v_esc = compute_v_esc(muA, rAB)

    sigma = np.pi * (muB/(v1**2 - v_esc**2))**2 * (-1 - ((v_esc**2 - vB**2)/(v1**2 - vB**2))**2
                                                  + ((v_esc**2 + vB**2)/(v1 * vB)) * np.arctanh(2* v1 * vB/(v1**2 + vB**2)))
    return sigma

def compute_miscelanous(muA, muB, v_inf, rAB, epsilon, a):

    e = 0.84

    ksi = 3 - rAB/a
    eta = np.sqrt(a * (1 - e**2)/rAB)
    kappa_plus = np.sqrt(-ksi + 2*eta)
    kappa_minus = np.sqrt(-ksi - 2*eta)

    return ksi, eta, kappa_plus, kappa_minus

def compute_ejection_crossec(muA, muB, v_inf, rAB, epsilon, a, vB):

    sigma = np.pi * (2 * mB * rAB / (5*mA))**2 * (-1 - (vB**2 * rAB - 2*muA)/(2*vB * rAB + muA)**2
                                                 - (vB**2 * rAB + 2*muA)/(vB * np.sqrt(rAB * muA / 2)) 
                                                 * np.arctan(2*vB * np.sqrt(2 * muA *rAB)/(muA - 2 * vB**2 * rAB)))

    return sigma

def compute_Wx(muA, muB, v_inf, a0, a , e):
    
    A = a/a0 # a is aC, a0 is aB
    e = eccentricity(rAB, muB, muA, v_inf)

    Wx = np.sqrt(2 - 1/A - A * (1 - e**2))

    return Wx

def compute_ejection_rate(K, muA, muB, v_inf, rAB, epsilon, abar, vB):

    ebar = 0.84
    ksi, eta, kappa_plus, kappa_minus = compute_miscelanous(muA, muB, v_inf, rAB, epsilon, abar)
    Wx = compute_Wx(muA, muB, v_inf, rAB, abar, ebar)
    sigma = compute_ejection_crossec(muA, muB, v_inf, rAB, epsilon, a, vB)

    Re = ( K * vB**2 * sigma ) / ( 2 * np.pi**(5/2) * rAB**(3/2) * abar**(3/2) * np.sqrt(Wx) ) * (2 * np.sqrt(ksi) -
        kappa_minus * np.arctan(np.sqrt(ksi) / kappa_minus) + kappa_plus * np.arctan(np.sqrt(ksi) / kappa_plus) +
        1j * (kappa_minus * np.arctanh(kappa_plus / kappa_minus) + kappa_plus * np.arctanh(1 + (eta * sigma) / (2 * np.pi * rAB**(2) * kappa_plus**2)) -
        2j * kappa_plus))

    return abs(Re)

Cap = compute_capture_crossec_threebody(muA, muB, v_inf, rAB, 0.1)
print(Cap)
print(f"Cap in terms of Aj = {Cap/(np.pi*(7.1492e7)**2)}")
Ej = compute_ejection_crossec(muA, muB, v_inf, rAB, 0.1, abar, vB)
print(Ej)
Re = compute_ejection_rate(kappa, muA, muB, v_inf, rAB, 0.1, abar, vB)
print(Re)
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

plt.figure(figsize=(8,6))
plt.imshow((cross_section_grid), origin='lower', extent=[np.log10(m_pbh_values[0]),
                                                       np.log10(m_pbh_values[-1]),
                                                       np.log10(v_inf_values[0]),
                                                       np.log10(v_inf_values[-1])],
           aspect='auto', cmap='plasma', norm=plt.cm.colors.LogNorm())

plt.xlabel(r'log$_{10}$(PBH mass $M_\odot$)')
plt.ylabel('Velocity (km/s)')
cbar = plt.colorbar()
cbar.set_label('Capture cross-section (AU$^2$)')
plt.title('Capture cross-section as function of PBH mass and velocity')
plt.savefig(directoryp + f'{star}_capture_cross_section_grid.png', )

# Figure 2
plt.figure(figsize=(8,6))
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
plt.savefig(directoryp + f'{star}_capture_cross_section.png')

# Figure 3
plt.figure(figsize=(8,6))
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
plt.grid(True, which="both")
plt.savefig(directoryp + f'{star}_capture_rate.png')


# import seaborn as sns
# import pandas as pd
# M, V = np.meshgrid(m_pbh_values / const.M_sun, v_inf_values)
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
