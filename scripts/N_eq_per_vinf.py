import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u
import numpy as np
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
import multiprocessing
import shutil
from mpl_toolkits.mplot3d import Axes3D



def sigma_cap(v1Mag, v1primeMag, vBMag, v_esc, muB):
    """
    Capture cross-section σ_cap(v1).

    Parameters
    ----------
    v1 : float
        Incoming velocity.
    v1_prime : float
        Incoming velocity in Body B frame.
    vB : float
        Orbital velocity of body B.
    vesc : float
        Escape velocity of body A or system.
    muB : float
        Reduced mass associated with body B (μ_B).

    Returns
    -------
    float
        σ_cap(v1)
    """
    numerator = np.pi * muB**2 * ((v1primeMag**2 - vBMag**2)**2 - v_esc**4)
    denominator = (v1Mag**2 - v_esc**2)**2 * v1primeMag**4
    return numerator / denominator

# def sigma_cap_dEps(v1, v1_prime, vB, v_esc, muB, kappa):
#     """
#     Differential capture cross-section dσ_cap/dε.

#     Parameters
#     ----------
#     v1 : float
#         Incoming velocity.
#     v1_prime : float
#         Incoming velocity in Body B frame.
#     vB : float
#         Orbital velocity of body B.
#     vesc : float
#         Escape velocity of body A or system.
#     muB : float
#         Reduced mass associated with body B (μ_B).
#     kappa : float
#         Energy dissipation parameter.

#     Returns
#     -------
#     float
#         dσ_cap/dε
#     """
#     sigma_cap_val = 16
#     return sigma_cap_val / kappa

def sigma_collision(R, b, bmin, rmin):
    """
    Collision cross-section σ_collision(v1).

    Parameters
    ----------
    v1 : float
        Incoming velocity.
    muB : float
        Reduced mass associated with body B (μ_B).
    r_close : float
        Closest approach distance.

    Returns
    -------
    float
        σ_collision(v1)
    """
    sigma_int_1 = -(0.5*(-b + R + bmin) * (b + R - bmin) * (b - R + bmin) * (b + R + bmin))**0.5
    sigma_int_2 = R**2 * np.arccos((b**2 + R**2 - bmin**2) / (2*b*R))
    sigma_int_3 = rmin**2 * np.arccos((b**2 + bmin**2 - R**2) / (2*b*bmin))

    return sigma_int_1 + sigma_int_2 + sigma_int_3
    

def crossec_circle_R_b(v1Mag,  vBMag, vBVec, v1primeMag, v1primeVec, vesc, muB):
    v1_prime_vec_xy = v1primeVec.copy()
    v1_prime_vec_xy[2] = 0.0
    v1_prime_mag_xy = np.linalg.norm(v1_prime_vec_xy)  # shape (n_mu, n_lam)

    # Per-element dot product and cross product
    dot_vB_v1p = np.einsum('i... , i... -> ...', vBVec, v1primeVec)           # shape (n_mu, n_lam)
    cross_v1p_vB = np.cross(v1primeVec, vBVec)                         # shape (3, n_mu, n_lam)

    v1prime_z = v1primeVec[2]                                                  # shape (n_mu, n_lam)
    v1prime_y = v1primeVec[1]                                                  # shape (n_mu, n_lam)
    cross_z = cross_v1p_vB[2]                                                  # shape (n_mu, n_lam)

    # Build g1..g5 as arrays
    g1 = 4*muB * dot_vB_v1p * (v1primeMag**6) * v1prime_z
    g2 = 4*(muB**3) * dot_vB_v1p * (v1primeMag**2) * v1prime_z
    g3 = 4*muB * (v1primeMag**7) * cross_z
    g4 = np.sign(v1prime_y) * (v1_prime_mag_xy**8) * v1_prime_mag_xy * (v1Mag**2 - vesc**2)
    g5 = 2 * np.sign(v1prime_y) * (muB**2) * (v1primeMag**4) * v1_prime_mag_xy * (v1primeMag**2 + vBMag**2 - vesc**2)

    # Avoid divisions by zero and negative expr with masks
    mask = (g1 != 0) & (g4 != 0)
    expr = np.zeros_like(g1)
    expr[mask] = g2[mask]/g1[mask] + (g1[mask]**2 + g3[mask]**2)/(4*g4[mask]**2) - g5[mask]/g4[mask]

    # Only keep positive expr
    pos = mask & (expr > 0)

    Rb = np.zeros_like(expr)
    b  = np.zeros_like(expr)

    Rb[pos] = np.sqrt(expr[pos])
    b[pos]  = -0.5*(g3[pos] + g1[pos]) / g4[pos]

    return Rb, b
    

def v_esc(muA, rAB):
    """
    Escape velocity v_esc.

    Parameters
    ----------
    mA : float
        Mass of body A.
    rAB : float
        Separation between bodies A and B.

    Returns
    -------
    float
        v_esc
    """

    return np.sqrt(2 * muA / rAB)

def r_close(epsilon, mA, mB, rAB):
    """
    Closest approach distance r_close.

    Parameters
    ----------
    epsilon : float
        Energy dissipation parameter.
    mA : float
        Mass of body A.
    mB : float
        Mass of body B.
    rAB : float
        Separation between bodies A and B.

    Returns
    -------
    float
        r_close
    """

    return rAB * (epsilon * mB / mA)**(1/3)

def r_min(muA, b, v1_prime):
    """
    Minimum approach distance b_min.

    Parameters
    ----------
    muA : float
        Reduced mass associated with body A (μ_A).
    b : float
        Impact parameter.
    v1_prime : float
        Incoming velocity in Body B frame.

    Returns
    -------
    float
        r_min
    """

    e1_prime = np.sqrt(1 + (b**2 * v1_prime**4) / muA**2)
    a1_prime = -b / np.sqrt(e1_prime**2 - 1)
    r_min = (np.sqrt(muA**2 + b**2 * v1_prime**4) - muA) / (v1_prime**2)

    return a1_prime * (1-e1_prime)

def r_AB_vec(rAB, lambda_1):
    """
    Separation vector between bodies A and B r_AB.

    Parameters
    ----------
    rAB : float
        Separation between bodies A and B.
    lambda_1 : float
        Scattering angle.

    Returns
    -------
    float
        r_AB
    """
    rABVec = np.array([ # broadcast over grid
        np.zeros_like(lambda_1),
        np.ones_like(lambda_1) * rAB,
        np.zeros_like(lambda_1)
    ])
    return rABVec

def b_min(muB, rB, v1_prime):
    """
    Minimum impact parameter b_min.

    Parameters
    ----------
    muB : float
        Reduced mass associated with body B (μ_B).
    rB : float
        Radius of body B.
    v1_prime : float
        Incoming velocity in Body B frame.

    Returns
    -------
    float
        b_min
    """
    b_min = np.sqrt(2*muB*rB + (rB*v1_prime)**2) / v1_prime

    return b_min

def v_1_mag(v_inf, muA, muB, rAB, rClose):
    """
    Incoming velocity v1.

    Parameters
    ----------
    v_inf : float
        Velocity at infinity.
    mA : float
        Mass of body A.
    rAB : float
        Separation between bodies A and B.


    Returns
    -------
    float
        v1
    """

    term1 = v_inf**2 + (2 * muA / rAB)
    term2 = (2 * muB / rClose)
    return np.sqrt(term1 + term2)

def v_1_vec(v1Mag, lambda_1, beta_1):
    """
    Incoming velocity v1.
    Parameters
    ----------
    v1_mag : float
        Incoming velocity magnitude.
    vB_mag : float
        Orbital velocity magnitude of body B.
    lambda_1 : float
        Scattering angle.
    beta_1 : float
        Impact parameter angle. 
    Returns
    -------
    float
        v1
    """
    # v1 direction set by (β1, λ1) as in Lehmann fig. 2
    v1_vec = v1Mag * np.array([
        np.sin(lambda_1) * np.cos(beta_1),
        np.cos(lambda_1) * np.cos(beta_1),
        -np.sin(beta_1),
    ])
    return v1_vec

def v_1_prime_mag(v1primeVec):
    """
    Magnitude of incoming velocity in Body B frame |v1'|.
    Parameters
    ----------
    v1_mag : float
        Incoming velocity magnitude.
    vB_mag : float
        Orbital velocity magnitude of body B.
    lambda_1 : float
        Scattering angle.
    beta_1 : float
        Impact parameter angle. 
    Returns
    -------
    float
        |v1'|
    """
    return np.linalg.norm(v1primeVec)

def v_1_prime_vec(v1Vec, vBVec, lambda_1, beta_1):
    """
    Incoming velocity in Body B frame v1'.
    Parameters
    ----------
    v1_mag : float
        Incoming velocity magnitude.
    vB_mag : float
        Orbital velocity magnitude of body B.
    lambda_1 : float
        Scattering angle.
    beta_1 : float
        Impact parameter angle. 
    Returns
    -------
    float
        v1'
    """
    v1_prime_vec = v1Vec - vBVec
    return v1_prime_vec

def v_2_vec(v1primeVec,v1primeMag,vBVec, muB, b, phi):
    """
    Outgoing velocity v2.
    Parameters
    ----------
    v1primeVec : float
        Incoming velocity vector in Body B frame.
    v1primeMag : float
        Magnitude of incoming velocity in Body B frame.
    muB : float
        Reduced mass associated with body B (μ_B).
    vB : float
        Orbital velocity of body B.
    b : float
        Impact parameter. 
    Returns
    -------
    float
        v2
    """

    v1prime_vec = v1primeVec.copy()
    v1prime_x = v1prime_vec[0]
    v1prime_y = v1prime_vec[1]
    v1prime_z = v1prime_vec[2]
    v1prime_xy = v1prime_vec.copy()
    v1prime_xy[2] = 0.0
    v1primeMag_xy = np.linalg.norm(v1prime_xy)

    q = np.sqrt(1 + v1prime_z**2/v1primeMag_xy**2)

    term1 = vBVec + (b**2 * v1primeMag**4 - muB**2)/(b**2 * v1primeMag**4 + muB**2) * v1primeVec
    term2 = (2 * np.sign(v1prime_y) * muB * v1primeMag * b)/(b**2 * v1primeMag**4 + muB**2) * np.array([
        q * (v1prime_x * v1prime_z * np.sin(phi) - v1primeMag * v1prime_y * np.cos(phi)),
        q * (v1prime_y * v1prime_z * np.sin(phi) + v1primeMag * v1prime_x * np.cos(phi)),
        -v1primeMag_xy**2 * np.sin(phi)
    ])
    return term1 + term2

def v_2_mag(v2Vec):
    """
    Magnitude of outgoing velocity |v2|.
    Parameters
    ----------
    v2Vec : float
        Outgoing velocity vector.
    Returns
    -------
    float
        |v2|
    """
    return np.linalg.norm(v2Vec)

def v_B_vec(vBMag, lambda_1):
    """
    Orbital velocity vector of body B vB.
    Parameters
    ----------
    vBMag : float
        Orbital velocity magnitude of body B.
    lambda_1 : float
        Scattering angle.
    Returns
    -------
    float
        vB
    """
    vBVec = vBMag * np.array([np.sin(lambda_1), np.cos(lambda_1), np.zeros_like(lambda_1)])
    return vBVec


def chi(muB, v1Mag, vBMag, v1primeMag, vBVec, v1primeVec, vesc, sigma_cap,  rclose, rB, rmin, Rb, b, bmin):
    """
    Indicator function to check kinematical allowance χ.

    Parameters
    ----------
    v_1 : float
        Incoming velocity.
    vB : float
        Orbital velocity of body B.
    lambda_1 : float
        Scattering angle.

    Returns
    -------
    boolean
        χ
    """
    condition0 = Rb > 0.0
    v_max = vesc + 2*vBMag               # scalar
    condition1 = v1Mag < v_max           # scalar, broadcasts
    condition2 = sigma_cap > 0           # array
    condition3 = Rb < rclose             # array
    # If you want to enforce rmin > rB and b > bmin, use these arrays; otherwise skip
    # condition4 = rmin > rB
    condition5 = b > bmin

    # Combine elementwise with &; convert to float (1.0/0.0)
    result_mask = condition0 & condition1 & condition2 & condition3 & condition5
    return result_mask.astype(float)
def chi_inE2(muB, vBMag, vBVec, v1primeMag, v1primeVec, v2Mag, v1Mag, U, E2, rClose):
    """
    Indicator function to check kinematical allowance χ in E2.

    Parameters
    ----------
    v_1 : float
        Incoming velocity.
    vB : float
        Orbital velocity of body B.
    lambda_1 : float
        Scattering angle.

    Returns
    -------
    boolean
        χ in E2
    """

    # Condition 1
    vesc = v2Mag
    Rb, b = crossec_circle_R_b(v1Mag, vBMag, vBVec, v1primeMag, v1primeVec, vesc, muB)
    condition1 = Rb < rClose

    return 1.0 if condition1 else 0.0

def dsigma_directional_average(sigma_cap, chi):
    """
    Differential directional average cross-section dσ/dΩ.

    Parameters
    ----------
    lambda_1 : float
        Scattering angle.
    beta_1 : float
        Impact parameter.
    sigma_cap : float
        Capture cross-section.
    chi : float
        Kinematical allowance.

    Returns
    -------
    float
        dσ/dΩ
    """

    return (sigma_cap) * chi / (2*np.pi)

def U(muA, muB, rAB, rClose):
    """
    Potential energy U.

    Parameters
    ----------
    muA : float
        Reduced mass associated with body A (μ_A).
    muB : float
        Reduced mass associated with body B (μ_B).
    rAB : float
        Separation between bodies A and B.
    rclose : float
        Closest approach distance.

    Returns
    -------
    float
        U parameter
    """
    U  = -(muA / rAB) - (muB / rClose)

    return U

def E2(v2Mag, U):
    """
    Final energy E2.

    Parameters
    ----------
    v2Mag : float
        Magnitude of the outgoing velocity vector.
    U : float
        Potential energy U.

    Returns
    -------
    float
        E2
    """
    return 0.5 * v2Mag**2 + U

def L2(rABVec, v2Vec):
    """
    Angular momentum L2.

    Parameters
    ----------
    rABVec : float
        Separation vector between bodies A and B.
    v2Vec : float
        Outgoing velocity vector.
    muA : float
        Reduced mass associated with body A (μ_A).

    Returns
    -------
    float
        L2
    """
    return np.linalg.norm(np.cross(rABVec, v2Vec))


def a_e(muA, E2, L2):
    """
    Semi-major axis a.

    Parameters
    ----------
    muA : float
        Reduced mass associated with body A (μ_A).
    E2 : float
        Final energy E2.
    L2 : float
        Angular momentum L2.

    Returns
    -------
    float
        a
    """
    a = - muA / (2*E2)
    e = np.sqrt(1 + (2*E2*L2**2)/(muA**2))
    return a, e



# --- Parameters ---
G = const.G.value
m_Sun = const.M_sun.value
m_Jup =  const.M_jup.value  # Mass of Jupiter
m_Neptune = 5.15*10**-5 * const.M_sun.value  # Mass of Neptune
m_PBH = 1e-13 * const.M_sun.value
r_SJ = (5.2*u.au).to(u.m).value
r_SN = (30.07*u.au).to(u.m).value
v_Jup = 13.06 * 1e3  # km/s to m/s
v_Neptune = 5.45 * 1e3  # km/s to m/s
epsilon = 0.1
r_jupiter = const.R_jup.to(u.m).value
r_neptune = 24622 * 1e3  # in meters
A_jupiter = np.pi * r_jupiter**2

rAB = r_SJ
rB = r_jupiter
mA = m_Sun
mB = m_Jup
muA = G * mA
muB = G * mB
vesc = v_esc(muA, rAB)
rclose = r_close(epsilon, mA, mB, rAB)
vBMag = v_Jup
vdm = 220 * 1e3  # m/s
v1_max = vesc + v_Jup
v_max = np.sqrt(v1_max**2 - 2*muA/r_SJ - 2*muB/rclose)
v_inf_grid = np.linspace(1e3, v_max, 20)  # m/s

for v_inf in v_inf_grid:
    v1Mag = v_1_mag(v_inf, muA, muB, rAB, rclose)
# --- Monte Carlo ---
N = 100_000 * 100
rng = np.random.default_rng(42)

# 1) sample direction of incoming PBH (λ, β)
lam_samples = rng.uniform(0.0, 2*np.pi, size=N)
mu_samples  = rng.uniform(-1.0, 1.0, size=N)    # cosβ
beta_samples = np.arccos(mu_samples)

# 2) sample impact parameter b with p(b) ∝ b
b_max = rclose                      # set maximum impact parameter (e.g. rclose)
u_b = rng.uniform(0.0, 1.0, size=N)
b_samples = b_max * np.sqrt(u_b)

# fixed Sun–Jupiter separation vector (choose an axis)
rABVec = np.array([0.0,rAB, 0.0])

# 3) sample scattering-plane angle φ
phi_samples = rng.uniform(0.0, 2*np.pi, size=N)

# containers for captures
a_list = []
e_list = []
lambda_list = []    
beta_list = []
b_list = []
phi_list = []

n_captured = 0

# specific potential around Sun at rAB
U_val = -muA / rAB - muB/rclose

for lam, beta, b, phi in zip(lam_samples, beta_samples, b_samples, phi_samples):
    # incoming and planet velocity
    v1Vec = v_1_vec(v1Mag, lam, beta)  # (3,)
    vBVec = v_B_vec(vBMag, lam)        # (3,)

    # relative velocity in Jupiter frame
    v1primeVec = v_1_prime_vec(v1Vec, vBVec, lam, beta)
    v1primeMag = np.linalg.norm(v1primeVec)

    if v1primeMag == 0:
        continue

    # optional: skip collisions with Jupiter
    if b < r_jupiter:
        continue

    # outgoing velocity in Sun frame
    v2Vec = v_2_vec(v1primeVec, v1primeMag, vBVec, muB, b, phi)
    v2Mag = v_2_mag(v2Vec)

    # specific energy around the Sun
    E2_val = 0.5*v2Mag**2 + U_val

    if E2_val + muB/rclose  < 0:
        # bound to Sun
        L2_val = L2(rABVec, v2Vec)
        a_val, e_val = a_e(muA, E2_val, L2_val)

        # physical sanity checks: a>0, 0<=e<1
        if a_val > 0 and 0 <= e_val < 1:
            a_list.append(a_val)
            e_list.append(e_val)
            lambda_list.append(lam)    
            beta_list.append(beta)
            b_list.append(b)
            phi_list.append(phi)
            n_captured += 1

# convert to arrays
a_arr = np.array(a_list)   # meters
e_arr = np.array(e_list)

# estimate capture cross-section:
sigma_MC = (n_captured / N) * np.pi * b_max**2

print("MC capture cross-section:", sigma_MC, "m^2")
print("In units of Jupiter area:", sigma_MC / A_jupiter)

# orbital element stats
a_au = (a_arr*u.m).to(u.au).value
print("Number of captured orbits:", len(a_au))
print("a (au): mean =", np.mean(a_au), "std =", np.std(a_au))
print("e: mean =", np.mean(e_arr), "std =", np.std(e_arr))

s = datetime.datetime.now()
# Displays Time
current_time = s.strftime('%H%M')
print("current time :", current_time)
if not os.path.exists(f'/data/a.saricaoglu/repo/COMPAS/capture/runs/Sun_Jupyter_System/{str(s.strftime("%m.%d"))}/{current_time}/LBImpB_Variation/'): 
    os.makedirs(f'/data/a.saricaoglu/repo/COMPAS/capture/runs/Sun_Jupyter_System/{str(s.strftime("%m.%d"))}/{current_time}/LBImpB_Variation/')
directoryf = f'/data/a.saricaoglu/repo/COMPAS/capture/runs/Sun_Jupyter_System/{str(s.strftime("%m.%d"))}/{current_time}/LBImpB_Variation/'

if not os.path.exists(f'/data/a.saricaoglu/repo/COMPAS/Plots/Capture/{str(s.strftime("%m.%d"))}/{current_time}/'): 
    os.makedirs(f'/data/a.saricaoglu/repo/COMPAS/Plots/Capture/{str(s.strftime("%m.%d"))}/{current_time}/') 
directoryp = f'/data/a.saricaoglu/repo/COMPAS/Plots/Capture/{str(s.strftime("%m.%d"))}/{current_time}/' 



# Redirect stdout and stderr to the log file
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


    

# Number of simulations

if __name__ == "__main__":


    # Create a pool of workers
    num_cores = 30  # Get the number of available CPU cores
    failed_systems = [32, 408, 135, 136, 264, 360, 11, 12, 406, 407, 184, 31]
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Distribute the simulations across the cores
        pars = []
        i = 0
        for lambda1, beta, phi, b, a_c, e_c in zip(lambda_list, beta_list, phi_list, b_list, a_au, e_list):
            i += 1
            if i in failed_systems:
                pars.append((i, beta, lambda1, phi, b, a_c, e_c))
        pool.map(run_simulation, pars)
    pool.close()
    pool.join()

    print(f'Execution time: {datetime.datetime.now() - s} for {i} simulations')


