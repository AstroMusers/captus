import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const

# # Constants (example values, you can modify)
# G = 6.67430e-11  # m^3 kg^-1 s^-2
# M_star = 1.989e30  # Mass of star (solar mass in kg)
# R_star = 6.957e8   # Radius of star (solar radius in meters)
# rho_dm = 0.4 * 1.7827e-24 * 1e6  # 0.4 GeV/cm^3 to kg/m^3
# v_inf = 200e3  # 200 km/s

# # PBH mass range
# m_pbh = 1e20  # in kg (you can loop over this to generate curves)

# # Number density of PBHs
# f_pbh = 1.0  # set to 1 for normalization
# n_pbh = f_pbh * rho_dm / m_pbh

# # Capture condition threshold energy
# def E_threshold():
#     return 0.5 * m_pbh * v_inf**2

# # Define capture probability as a smooth function (example: logistic cutoff near b_cap)
# def P_cap(b, b_cap):
#     width = 0.05 * b_cap  # smoothness scale
#     return 1 / (1 + np.exp((b - b_cap) / width))

# # Calculate b_cap using the correct formula from the paper
# def b_cap():
#     return np.sqrt((4 * G * M_star * R_star) / v_inf**2)

# bcap = b_cap()

# # Numerical integration
# def integrand(b):
#     return 2 * np.pi * b * P_cap(b, bcap)

# R, err = quad(integrand, 0, bcap * 3)  # integrate up to 3 times bcap for smooth tail
# R_total = R * n_pbh

# print(f"Capture rate R: {R_total:.3e} s^-1")

# # Plot capture probability as a function of impact parameter
# b_vals = np.linspace(0, bcap * 2, 500)
# p_vals = [P_cap(b, bcap) for b in b_vals]

# plt.plot(b_vals, p_vals)
# plt.xlabel('Impact parameter b (m)')
# plt.ylabel('Capture Probability P_cap(b)')
# plt.title('Capture Probability vs Impact Parameter')
# plt.grid()
# plt.show()

# import numpy as np
# from scipy.integrate import quad
# import matplotlib.pyplot as plt

# # Constants
# G = const.G  # m^3 kg^-1 s^-2
# M_star = const.M_sun  # Mass of star (solar mass in kg)
# R_star = 10 * u.km # Radius of star (solar radius in meters)
# v_inf = 30 * u.km     # m/s (30 km/s)
# rho_dm = 0.4 * u.GeV# 0.4 GeV/cm^3 to kg/m^3
# seconds_per_year = 3.154e7 * u.second

# # Capture probability as a smooth function
# def P_cap(b, b_cap):
#     width = 0.05 * b_cap  # smoothness scale
#     return 1 / (1 + np.exp((b - b_cap) / width))

# # Calculate b_cap from paper formula
# def b_cap():
#     return np.sqrt((4 * G * M_star * R_star) / v_inf**2)

# bcap = b_cap()

# # Integrate over impact parameter for geometric factor
# def geometric_factor(bcap):
#     def integrand(b):
#         return 2 * np.pi * b * P_cap(b, bcap)
#     result, _ = quad(integrand, 0, bcap * 3)
#     return result

# geo_factor = geometric_factor(bcap)

# # Mass range for PBHs (grams to kg)
m_pbh_values = np.linspace(1e-10, 1e2, 1000) * const.M_sun  # convert g to kg
# capture_rates = []

# for m_pbh in m_pbh_values:
#     n_pbh = rho_dm / m_pbh
#     # R = n_pbh * v_inf * geometric factor
#     R = n_pbh * v_inf * geo_factor
#     R_per_year = R * seconds_per_year
#     capture_rates.append(R_per_year)

# # Plot
# plt.figure(figsize=(8,6))
# plt.loglog(m_pbh_values, capture_rates)  # convert back to g for axis labeling
# plt.xlabel(r'PBH mass $m_{\rm PBH}$ (g)', fontsize=12)
# plt.ylabel(r'Capture rate $R$ (yr$^{-1}$)', fontsize=12)
# plt.title('Reproduction of Figure 1 (right): PBH capture rate', fontsize=14)
# plt.grid(True, which="both", ls="--")
# plt.show()
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# Constants
G = const.G  # m^3 kg^-1 s^-2
print(G)
# Example stellar parameters (Sun-like)
M_star = 2* const.M_sun # kg
R_star = 10e4 * u.m # m
v_inf = 100 * 1000 * u.m / u.second    # m/s (30 km/s)
print(M_star, R_star, v_inf)
print(G*M_star)
print(u.second)


# Function for p(e) from the paper (approximation or constant if they provide one)
def p_e(e):
    # Often given in the paper; if not, use p(e) ~ 1 for simplicity
    # print(e)
    return (e + 1)**-3.5 * (np.arccos(-1/e) * (24 + 73*e**2 + (37/4)*e**4 + (602+673*e**2)*np.sqrt(e**2 - 1)/12)) # Placeholder

# Relate eccentricity e to impact parameter b
def eccentricity(b, M_pbh, M_star, v_inf):
    return 1 + 1e-9*(b/(1e2*1000))**2 * (v_inf/(220*(1000/1)))**4 * (M_star/const.M_sun.value)**-2

# Delta E_GW(b) from Eq. (2.4)
def delta_E_GW(b, M_pbh, M_star, v_inf):
    e = eccentricity(b, M_pbh, M_star, v_inf)
    pre_factor = (8/15) * (M_pbh**2 * M_star**2) / (M_pbh + M_star)**3
    return pre_factor * p_e(e) / ( (e - 1)**(3.5) ) * v_inf**7

# Equation to solve: Delta E_GW(b_max) = (1/2) M_pbh v_inf^2
def equation(b, M_pbh, M_star, v_inf):
    return delta_E_GW(b, M_pbh, M_star, v_inf) - (0.5 * M_pbh * v_inf**2)

# Function to compute b_max

def compute_b_max(M_pbh, M_star=M_star, R_star=R_star, v_inf=v_inf):
    b_guess_min = 0* u.m  # Lower bound
    b_guess_max = 1e20 * u.m # Large search range
    # print(b_guess_min, b_guess_max)
    b_max = brentq(equation, b_guess_min.value, b_guess_max.value, args=(M_pbh.value, M_star.value, v_inf.value))
    return b_max 

def compute_capture_crossec(b_min, b_max):
    return np.pi * (b_max - b_min)**2

# Example usage: plot b_max vs PBH mass
masses = m_pbh_values  # PBH mass range in kg
b_max_values = [compute_b_max(m, M_star, R_star, v_inf) for m in masses]
b_max_values = b_max_values * u.m
b_max_values = b_max_values.to(u.au) 
b_min = np.sqrt(R_star**2 + (2 * G * M_star * R_star) / v_inf**2) 
b_min = b_min.to(u.au)
print(b_min)
print(b_max_values)
crossec = compute_capture_crossec(b_min, b_max_values)
print(crossec)
plt.loglog(masses/const.M_sun.value, b_max_values)
plt.xlabel(r"PBH mass ($M_{\odot}$)")
plt.ylabel(r"$b_{max}$ (au)")
plt.title(r"$b_{max}$ from GW capture condition")
plt.grid(True, which="both")
plt.show()
plt.loglog(masses/const.M_sun.value, crossec)
plt.xlabel(r"PBH mass ($M_{\odot}$)")
plt.ylabel(r"$\sigma_{cap}$ (m)")
plt.title(r"$\sigma_{cap}$ ")
plt.grid(True, which="both")
plt.show()


m_pbh_values = np.linspace(1e-10, 1e2, 100) * const.M_sun  # PBH masses
v_inf_values = np.linspace(1e3, 1e6, 100) * u.m / u.s     # velocities from 1 km/s to 1000 km/s
M_grid, V_grid = np.meshgrid(m_pbh_values, v_inf_values)

b_max_grid = np.zeros(M_grid.shape)
for i in range(len(v_inf_values)):
    for j in range(len(m_pbh_values)):
        b_max_grid[i, j] = compute_b_max(M_grid[i, j], M_star, R_star, V_grid[i, j])

b_max_grid = b_max_grid * u.m
b_max_grid = b_max_grid.to(u.au).value
b_min_grid = np.sqrt(R_star**2 + (2 * G * M_star * R_star) / V_grid**2).to(u.au).value
crossec_grid = np.pi * ( (b_max_grid - b_min_grid)**2 )
m_pbh_values[0] = m_pbh_values[0].to(u.M_sun)
m_pbh_values[-1] = m_pbh_values[-1].to(u.M_sun)
v_inf_values[0] = v_inf_values[0].to(u.km/u.s)
v_inf_values[-1] = v_inf_values[-1].to(u.km/u.s)
plt.figure(figsize=(8,6))
plt.imshow(crossec_grid, 
           extent=[(m_pbh_values[0].value/const.M_sun.value), (m_pbh_values[-1].value/const.M_sun.value), 
                   (v_inf_values[0].value), (v_inf_values[-1].value)],
           aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label=r'$\sigma_{cap}$ (m$^2$)')
plt.xlabel(r'$\log_{10}(M_{\rm PBH}/M_{\odot})$')
plt.ylabel(r'$\log_{10}(v_{\infty}/{\rm m\,s^{-1}})$')
plt.yscale('log')
plt.title('Capture cross-section')
plt.show()

# m_pbh_values = np.linspace(1e-10, 1e2, 100) * const.M_sun
# v_inf = np.linspace(1e3,1e9,100)* u.m / u.second 
# V_inf = np.tile(v_inf, (100,1)).T
# plt.imshow(V_inf.value, origin='lower')
# plt.colorbar()
# plt.show()
# M_pbh = np.tile(m_pbh_values, (100,1))
# plt.imshow(M_pbh.value, origin='lower')
# plt.colorbar()
# plt.show()
# M_star = 2 * const.M_sun * np.ones((100,100))
# R_star = 10e4 * u.m * np.ones((100,100))
# G = const.G * np.ones((100,100))
# # print(M_pbh, M_star, R_star, V_inf)
# # print(np.shape(M_pbh), np.shape(M_star), np.shape(R_star), np.shape(V_inf), np.shape(G))
# b_max_values = np.zeros((100,100))
# b_min_values = np.sqrt(R_star**2 + (2 * G * M_star * R_star) / V_inf**2) 
# # c = 0
# # for i in range(100):
# #     for j in range(100):
# #         print(M_pbh[i,j], M_star[i,j], R_star[i,j], V_inf[i,j])
# #         b_max_values[i,j] = compute_b_max(M_pbh[i,j], M_star[i,j], R_star[i,j], V_inf[i,j])
# im = np.zeros((100,100))
# im2=[]
# i = 0
# for v in  v_inf:
#     b_min= np.sqrt(R_star[0][0]**2 + (2 * G[0][0] * M_star[0][0] * R_star[0][0]) / v**2)
#     crossecc = []
#     for m in m_pbh_values:
#         print(m, M_star[0][0], R_star[0][0], v)
#         b_max = compute_b_max(m, M_star[0][0], R_star[0][0], v) 
#         b_max = b_max * u.m
#         crossecc.append(compute_capture_crossec(b_min.to(u.au), b_max.to(u.au)).value)
#     im[i,:] = crossecc
#     im2.append(crossecc[:])    
#     i += 1
#     plt.loglog(m_pbh_values/const.M_sun.value, crossecc)
# plt.xlabel(r"PBH mass ($M_{\odot}$)")
# plt.ylabel(r"$\sigma_{cap}$ (m)")
# plt.title(r"$\sigma_{cap}$ ")
# plt.grid(True, which="both")
# plt.show()

# print(np.shape(im))
# plt.imshow(np.log10(im), origin='lower')


# import pandas as pd
# df = pd.DataFrame(index=v_inf, columns=m_pbh_values, values = im)
# print(df)
# b_max_values = b_max_values * u.m
# b_min_values = b_min_values.to(u.au)
# b_max_values = b_max_values.to(u.au) 

# print(b_min_values[0].unit)
# print(np.min(b_min_values), np.max(b_min_values))
# print(np.min(b_max_values), np.max(b_max_values))
# # mask = [b_max_values < b_min_values]
# # print(np.shape(mask))
# capture_crossec = compute_capture_crossec(b_min_values, b_max_values)
# print(np.shape(capture_crossec))
# # capture_crossec = capture_crossec * (u.m.to(u.au))**2
# print(capture_crossec)
# import seaborn as sns

# import pandas as pd

# fig, ax = plt.subplots(figsize=(10, 7))

# # Plot the logarithm of the capture cross-section
# im = ax.imshow(
#     np.log10(capture_crossec.value), 
#     origin='lower', 
#     extent=[
#         np.log10(m_pbh_values[0].value/const.M_sun.value), 
#         np.log10(m_pbh_values[-1].value/const.M_sun.value),
#         np.log10(v_inf[0].value/1000), 
#         np.log10(v_inf[-1].value/1000)
#     ],
#     aspect='auto',
#     cmap='viridis'
# )

# # Add colorbar
# cbar = plt.colorbar(im, ax=ax)
# cbar.set_label(r'$\log_{10}(\sigma_{\rm capture} / \mathrm{au}^2)$')

# # Label axes
# ax.set_xlabel(r'$\log_{10}(M_{\rm PBH}/M_{\odot})$', fontsize=12)
# ax.set_ylabel(r'$\log_{10}(v_{\infty} / km\,s^{-1})$', fontsize=12)
# ax.set_title('Capture Cross-Section from GW Capture', fontsize=14)

# plt.tight_layout()
# plt.show()