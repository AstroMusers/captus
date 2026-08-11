import astropy.constants as const
from math import pi
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy import linalg
from scipy.optimize import brentq, minimize_scalar
from astropy import units as u
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
def schwarzchild_radius(mass):
    """Calculate the Schwarzschild radius for a given mass.

    Parameters:
    mass (float): Mass in kilograms.

    Returns:
    float: Schwarzschild radius in meters.
    """ # Gravitational constant in m^3 kg^-1 s^-2

    return 2 * const.G.value * mass / const.c.value**2

def semimajor_axis(period, mass1, mass2):
    """Calculate the semi-major axis of a binary system.

    Parameters:
    period (float): Orbital period in seconds.
    mass1 (float): Mass of the first body in kilograms.
    mass2 (float): Mass of the second body in kilograms.

    Returns:
    float: Semi-major axis in meters.
    """
    G = const.G.value  # Gravitational constant in m^3 kg^-1 s^-2
    return (G * (mass1 + mass2) * (period / (2 * pi))**2)**(1/3)

def orbital_period(semimajor_axis, mass1, mass2):
    """Calculate the orbital period of a binary system.

    Parameters:
    semimajor_axis (float): Semi-major axis in meters.
    mass1 (float): Mass of the first body in kilograms.
    mass2 (float): Mass of the second body in kilograms.

    Returns:
    float: Orbital period in seconds.
    """
    G = const.G.value  # Gravitational constant in m^3 kg^-1 s^-2
    return 2 * pi * np.sqrt(semimajor_axis**3 / (G * (mass1 + mass2)))

def hill_radius(semimajor_axis, mass1, mass2, eccentricity=0):
    """Calculate the Hill radius of a body in a binary system.

    Parameters:
    semimajor_axis (float): Semi-major axis of the orbit in meters.
    mass1 (float): Mass of the primary body in kilograms.
    mass2 (float): Mass of the secondary body in kilograms.
    eccentricity (float): Orbital eccentricity (default is 0 for circular orbits).

    Returns:
    float: Hill radius in meters.
    """
    return semimajor_axis * (1 - eccentricity) * (mass2 / (3 * (mass1 + mass2)))**(1/3)

def collision_cross_section(rB, v_escB, v1prime):
    """Calculate the collision cross-section.

    Parameters:
    rB (float): Radius of body B in meters.
    v_escB (float): Escape velocity from body B in m/s.
    v1prime (float): Incoming velocity in Body B frame in m/s.

    Returns:
    float: Collision cross-section in m^2.
    """
    return pi * rB**2 * (1 + (v_escB / v1prime)**2)

def area_of_circle(radius):
    """Calculate the area of a circle given its radius.

    Parameters:
    radius (float): Radius of the circle.

    Returns:
    float: Area of the circle.
    """
    return pi * radius**2

def potential_energy(muA, rAB, muB, rClose):
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
        Potential energy U.
    """
    return -(muA / rAB) - (muB / rClose)

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
    
def reduced_mass(mass1, mass2, approx=False):
    """
    Reduced mass μ.

    Parameters
    ----------
    mass1 : float
        Mass of body 1.
    mass2 : float
        Mass of body 2.
    approx : bool
        If True, use approximation for mass1 >> mass2.

    Returns
    -------
    float
        Reduced mass μ.
    """
    if approx:
        return mass2
    else:
        return (mass1 * mass2) / (mass1 + mass2)
    
def standard_gravitational_parameter(mass1, mass2, approx=False):
    """
    Standard gravitational parameter μ.

    Parameters
    ----------
    mass1 : float
        Mass of body 1.
    mass2 : float
        Mass of body 2.
    approx : bool
        If True, use approximation for mass1 >> mass2.

    Returns
    -------
    float
        Standard gravitational parameter μ.
    """
    G = const.G.value  # Gravitational constant in m^3 kg^-1 s^-2
    if approx:
        return G * mass1
    else:
        return G * (mass1 + mass2)

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


def r_close(epsilon, muA, muB, rAB, approx=False):
    """
    Closest approach distance r_close (exact).

    Parameters
    ----------
    epsilon : float
        Energy dissipation parameter.
    muA : float
        Reduced mass associated with body A (μ_A).
    muB : float
        Reduced mass associated with body B (μ_B).
    rAB : float
        Separation between bodies A and B.

    Returns
    -------
    float
        r_close
    """
    if approx:
        return rAB * (epsilon * muB / muA)**(1/3)
    else:
        return rAB * (epsilon * muB / (2 * muA))**(1/3)

def r_min(muB, b, v1_prime):
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

    e1_prime = np.sqrt(1 + (b**2 * v1_prime**4) / muB**2)
    a1_prime = -b / np.sqrt(e1_prime**2 - 1)
    r_min = (np.sqrt(muB**2 + b**2 * v1_prime**4) - muB) / (v1_prime**2)

    return a1_prime * (1-e1_prime)

def r_AB_vec(rAB, lambda_1):
    """
    Separation vector between bodies A and B r_AB.

    Parameters
    ----------
    rAB : float
        Separation between bodies A and B.
    lambda_1 : float
        Orbital phase angle of Body B (0 to 2π).

    Returns
    -------
    array
        Position vector of Body B relative to Star A: [x, y, z]
    """
    rABVec =rAB* np.array([ # broadcast over grid
        np.sin(lambda_1),
        -np.cos(lambda_1),
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

def v_1_vec(v1Mag, beta_1):
    """
    Incoming velocity v1.
    Parameters
    ----------
    v1_mag : float
        Incoming velocity magnitude.
    lambda_1 : float
        Orbital phase angle of Body B (determines approach direction).
    beta_1 : float
        Out-of-plane angle (polar angle from xy-plane).
    
    Returns
    -------
    float
        v1
    """
    # v1 direction set by (β1, λ1) as in Lehmann fig. 2
    v1_vec = v1Mag * np.array([
        np.zeros_like(beta_1),
        -np.cos(beta_1),
        np.sin(beta_1),
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

def v_1_prime_vec(v1Vec, vBVec):
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

def v_2_prime_vec(v1primeVec, v1primeMag, muB, b, phi):
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

    cosPsi = cos_psi(muB, b, v1primeMag)
    sinPsi = sin_psi(cosPsi, v1primeVec)
    b_hat = b_unit_vector(v1primeMag=v1primeMag, v1primeVec=v1primeVec, phi=phi)
    # q = np.sqrt(1 + v1prime_z**2/v1primeMag_xy**2)
    # q = 1/(v1primeMag_xy * v1primeMag)

    term1 = cosPsi * v1primeVec
    term2 = -sinPsi * v1primeMag * b_hat

    # term2 = ((2 * np.sign(v1prime_y) * muB * v1primeMag * b)/(b**2 * v1primeMag**4 + muB**2)) * np.array([
    #     q * (- v1prime_x * v1prime_z * np.sin(phi) + v1primeMag * v1prime_y * np.cos(phi)),
    #     q * (v1prime_y * v1prime_z * np.sin(phi) + v1primeMag * v1prime_x * np.cos(phi)),
    #     q * (v1primeMag_xy**2 * np.sin(phi))
    # ])
    # term2 = ((2 * np.sign(v1prime_y) * muB * v1primeMag * b)/(b**2 * v1primeMag**4 + muB**2)) * np.array([
    #     q * (v1prime_x * v1prime_z * np.sin(phi) - v1primeMag * v1prime_y * np.cos(phi)),
    #     q * (v1prime_y * v1prime_z * np.sin(phi) + v1primeMag * v1prime_x * np.cos(phi)),
    #     -v1primeMag*v1primeMag_xy * np.sin(phi)
    # ])
    # term2 = ((2  * muB * v1primeMag**2 * b)/(b**2 * v1primeMag**4 + muB**2)) * np.array([
    #     q * (- v1prime_x * v1prime_z * np.sin(phi) - v1primeMag * v1prime_y * np.cos(phi)),
    #     q * (- v1prime_y * v1prime_z * np.sin(phi) + v1primeMag * v1prime_x * np.cos(phi)),
    #     q * (v1primeMag_xy**2 * np.sin(phi))
    # ])
    return term1 + term2

def v_2_prime_mag(v2primeVec):
    """
    Magnitude of outgoing velocity in Body B frame |v2'|.
    Parameters
    ----------
    v2primeMag : float
        Magnitude of outgoing velocity in Body B frame.
    muB : float
        Reduced mass associated with body B (μ_B).
    b : float
        Impact parameter. 

    Returns
    -------
    float
        |v2'|
    """
    return np.linalg.norm(v2primeVec)

def v_2_vec(v2primeVec, vBVec):
    """
    Outgoing velocity in Body B frame v2'.
    Parameters
    ----------
    v2Vec : float
        Outgoing velocity vector.
    vBVec : float
        Orbital velocity vector of body B.

    Returns
    -------
    float
        v2'
    """
    v2_vec = v2primeVec + vBVec
    return v2_vec

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

def deflection_angle(cos_psi):

    return np.arccos(np.clip(cos_psi, -1, 1))

def cos_psi(muB, b, v1primeMag):
    """
    Cosine of deflection angle ψ.

    Parameters
    ----------
    muB : float
        Reduced mass associated with body B (μ_B).
    b : float
        Impact parameter.
    v1primeMag : float
        Magnitude of incoming velocity in Body B frame.

    Returns
    -------
    float
        cos(ψ)
    """
    return (b**2 * v1primeMag**4 - muB**2)/(b**2 * v1primeMag**4 + muB**2)

def sin_psi(cos_psi, v1primeVec):
    """
    Sine of deflection angle ψ.

    Parameters
    ----------
    muB : float
        Reduced mass associated with body B (μ_B).
    b : float
        Impact parameter.
    v1primeMag : float
        Magnitude of incoming velocity in Body B frame.

    Returns
    -------
    float
        sin(ψ)
    """
    # sinPsi = (2 * b * v1primeMag**2 * muB) / (b**2 * v1primeMag**4 + muB**2)
    # cos_psi = cosPsi(muB, b, v1primeMag)
    sinPsi_alt = np.sqrt(1 - cos_psi**2)
    return  sinPsi_alt

def v_B_vec(vBMag, lambda_1):
    """
    Orbital velocity vector of body B vB.
    Parameters
    ----------
    vBMag : float
        Orbital velocity magnitude of body B.
    lambda_1 : float
        Orbital phase angle of Body B.

    Returns
    -------
    float
        vB
    """
    vBVec = vBMag * np.array([np.cos(lambda_1), np.sin(lambda_1), np.zeros_like(lambda_1)])
    return vBVec

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

def specific_E2(v2Mag, U):
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

def specific_L2(rABVec, v2Vec):
    """
    Angular momentum L2.

    Parameters
    ----------
    rABVec : float
        Separation vector between bodies A and B.
    v2Vec : float
        Outgoing velocity vector.
    muA : float
        Gravitational parameter associated with body A (μ_A).

    Returns
    -------
    float
        L2
    """
    return np.linalg.norm(np.cross(rABVec, v2Vec))


def a_e(muA, specE2, specL2):
    """
    Semi-major axis a.

    Parameters
    ----------
    muA : float
        Gravitational parameter associated with body A (μ_A).
    specE2 : float
        Final energy E2.
    L2 : float
        Angular momentum L2.

    Returns
    -------
    float
        a
    """
    a = - muA / (2*specE2)
    e = np.sqrt(1 - (specL2**2) / (muA * a))
    return a, e

def beta_2(L_vec, vBVec, rABVec):
    """
    Out-of-plane angle β2.

    Parameters
    ----------
    L2 : float
        Angular momentum L2.
    a : float
        Semi-major axis a.
    e : float
        Eccentricity e.

    Returns
    -------
    float
        β2
    """
    L_mag = np.linalg.norm(L_vec)
    # Body B's angular momentum (circular orbit in xy-plane)
    L_B = np.cross(rABVec, vBVec)
    L_B_hat = L_B / np.linalg.norm(L_B)

    # Relative inclination
    cos_rel = np.dot(L_vec, L_B_hat) / L_mag
    rel_inclination_deg = np.degrees(np.arccos(np.clip(cos_rel, -1, 1)))
    return rel_inclination_deg

def capture_cross_section(b_min, b_max):
    """
    Capture cross-section σ_cap.

    Parameters
    ----------
    b_min : float
        Minimum impact parameter.
    b_max : float
        Maximum impact parameter.

    Returns
    -------
    float
        σ_cap
    """
    return np.pi * (b_max**2 - b_min**2)

def capture_cross_section_MC(b, n_sampled):
    """
    Monte Carlo estimate of capture cross-section σ_cap.

    Parameters
    ----------
    b_au : float
        Maximum impact parameter in astronomical units.
    n_sampled : int
        Number of Monte Carlo samples.

    Returns
    -------
    float
        σ_cap
    """
    b_au = (np.array(b)*u.m).to(u.au)  # Convert b from meters to astronomical units
    return 2 * np.pi * (np.max(b_au)-np.min(b_au)) * np.sum(b_au) / n_sampled

def capture_cross_section(b_mins, b_maxs, b_cap, n_sampled):
    """
    Monte Carlo estimate of capture cross-section σ_cap.

    Parameters
    ----------
    b_mins_au : float
        Minimum impact parameter in astronomical units.
    b_maxs_au : float
        Maximum impact parameter in astronomical units.
    b_cap_au : float
        Captured impact parameter in astronomical units.
    n_sampled : int
        Number of Monte Carlo samples.

    Returns
    -------
    float
        σ_cap
    """
    b_cap_au = (np.asarray(b_cap) * u.m).to(u.au)
    b_mins_au = (np.asarray(b_mins) * u.m).to(u.au)
    b_maxs_au = (np.asarray(b_maxs) * u.m).to(u.au)

    contrib = 2 * np.pi * (b_maxs_au - b_mins_au) * b_cap_au

    sigma_cap = np.sum(contrib) / n_sampled

    return contrib / n_sampled, sigma_cap
# ...existing code...
# def capture_cross_section_MC(b_captured, n_sampled, bmin, bmax, out_unit=u.au**2):
#     """
#     Monte Carlo estimate of capture cross-section:
#         sigma_hat = (2*pi*(bmax-bmin)/N) * sum_{captured} b_i

#     Parameters
#     ----------
#     b_captured : array-like
#         Captured impact parameters [meters].
#     n_sampled : int
#         Total number of sampled trials N.
#     bmin, bmax : float
#         Sampling bounds for impact parameter [meters].
#     out_unit : astropy unit, optional
#         Output unit (default: au^2).

#     Returns
#     -------
#     float
#         Estimated capture cross-section in `out_unit`.
#     """
#     if n_sampled <= 0:
#         return 0.0

#     b_captured = np.asarray(b_captured, dtype=float)
#     if b_captured.size == 0:
#         return 0.0

#     C = 2.0 * np.pi * (bmax - bmin)  # [m]
#     sigma_m2 = C * np.sum(b_captured) / n_sampled  # [m^2]

#     return (sigma_m2 * u.m**2).to(out_unit).value
def b_unit_vector( v1primeMag, v1primeVec, phi):
    """
    v1prime_vec: 3D numpy array (v1x', v1y', v1z')
    phi: scattering plane angle
    """
    v1prime_vec = v1primeVec.copy()
    v1prime_x = v1prime_vec[0]
    v1prime_y = v1prime_vec[1]
    v1prime_z = v1prime_vec[2]
    v1prime_xy = v1prime_vec.copy()
    v1prime_xy[2] = 0.0
    v1primeMag_xy = np.linalg.norm(v1prime_xy)

    q = 1/(v1primeMag_xy * v1primeMag)

    bVec = q * np.array([
        (- v1prime_x * v1prime_z * np.sin(phi) - v1primeMag * v1prime_y * np.cos(phi)),
        (- v1prime_y * v1prime_z * np.sin(phi) + v1primeMag * v1prime_x * np.cos(phi)),
        (v1primeMag_xy**2 * np.sin(phi))
    ])
    # term2 = ((2 * np.sign(v1prime_y) * muB * v1primeMag * b)/(b**2 * v1primeMag**4 + muB**2)) * np.array([
    #     q * (- v1prime_y * v1prime_z * np.cos(phi) + v1primeMag * v1prime_x * np.sin(phi)),
    #     q * (v1prime_x * v1prime_z * np.cos(phi) + v1primeMag * v1prime_y * np.sin(phi)),
    #     q * (v1primeMag_xy**2 * np.(phi))
    # ])
    return bVec

def integrate_trapezoidal(x, y, max=None, min=None):
    """
    Perform trapezoidal integration.

    Parameters
    ----------
    x : array-like
        Independent variable values.
    y : array-like
        Dependent variable values.

    Returns
    -------
    float
        Integral of y with respect to x.
    """
    if max is not None:
        mask = (np.asarray(x) <= max)
        x = np.asarray(x)[mask]
        y = np.asarray(y)[mask]
    if min is not None:
        mask = (np.asarray(x) >= min)
        x = np.asarray(x)[mask]
        y = np.asarray(y)[mask]
    else:
        x = np.asarray(x)
        y = np.asarray(y)
    integral = np.trapezoid(y, x)
    return integral

def gauss_legendre_integral(F, vmin, vmax, n_points=16):
    # Nodes x in [-1, 1], weights w
    x, w = leggauss(n_points)

    # Map nodes to [vmin, vmax]
    v = 0.5 * (vmax - vmin) * x + 0.5 * (vmax + vmin)

    # Evaluate integrand at these v
    F_vals = F(v)

    # Jacobian factor dv/dx = (vmax - vmin)/2
    return 0.5 * (vmax - vmin) * np.sum(w * F_vals)

def dict_to_sorted_arrays(occ, key1, key2):
    pairs = [
        (d['v_inf_au_yr'], d[key])
        for d in occ.values()
        if d[key] > 0
    ]
    pairs.sort(key=lambda x: x[0])
    v_arr  = np.array([p[0] for p in pairs])
    F_arr  = np.array([p[1] for p in pairs])
    return v_arr, F_arr

def integral_from_grid(v_grid, F_grid, vmin, vmax, n_points=None):
    if n_points is None:
        n_points = len(v_grid)

    def F_interp(v):
        return np.interp(v, v_grid, F_grid)

    return gauss_legendre_integral(F_interp, vmin=vmin, vmax=vmax, n_points=n_points)

def integrate_gauss_legendre(x, y, n=100, exclude_zeros=False):
    """
    Integrate y(x) over x using Gauss-Legendre quadrature.

    - x: 1D array of sample positions (monotonic not required; will be sorted)
    - y: 1D array of function values at x
    - n: number of quadrature points (typ. 8-64)

    Implementation notes:
    - We sort x and y together and drop NaNs.
    - We linearly interpolate y(x) to evaluate at the Gauss-Legendre nodes.
    - Interval is [xmin, xmax]. If less than 2 points, returns 0.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if exclude_zeros:
        mask = (y != 0)
        x = x[mask]
        y = y[mask]
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return 0.0
    # Sort by x
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    a = x[0]
    b = x[-1]
    # Get nodes and weights on [-1,1]
    xi, wi = leggauss(n)
    # Map nodes to [a,b]
    xm = 0.5*(b + a)
    xr = 0.5*(b - a)
    x_nodes = xm + xr * xi
    # Interpolate y at nodes
    y_nodes = np.interp(x_nodes, x, y)
    # Integral over [a,b]
    return xr * np.sum(wi * y_nodes)

def equation(b, v1primeVec, v1primeMag, vBVec, muB, phi, potential_energy, rclose):
    # Returns positive when GW loss < KE (no capture)
    # Returns negative when GW loss > KE (capture possible)
    v2primeVec = v_2_prime_vec(v1primeVec, v1primeMag, muB, b, phi)
    v2Vec = v_2_vec(v2primeVec=v2primeVec, vBVec=vBVec)
    v2Mag = v_2_mag(v2Vec)

    E2_val = 0.5 * v2Mag**2 + potential_energy
    return  E2_val + muB/rclose

def compute_b_max(limit_in_m, b_min,
                   v1primeVec, v1primeMag, vBVec, muB, phi,
                   potential_energy, rclose):
    """
    Find maximum b where |GW energy loss| ≥ kinetic energy
    
    At b_min: GW loss is very large (close encounter) → equation < 0
    As b increases: GW loss decreases → equation approaches 0 then becomes positive
    
    We want the b where equation = 0 (transition point)
    """
    # Check if capture is possible at b_min
    eq_at_bmin = equation(b_min, v1primeVec, v1primeMag, vBVec, muB, phi, potential_energy, rclose)

    if eq_at_bmin > 0:
        # Even at closest approach, GW loss < KE → no capture possible
        return b_min
    
    # Find where equation crosses zero (GW loss = KE)
    b_guess_max = b_min  # Start with 10x b_min
    
    # Expand search until we bracket the root
    while b_guess_max < limit_in_m:
        eq_at_bmax = equation(b_guess_max, v1primeVec, v1primeMag, vBVec, muB, phi, potential_energy, rclose)
        # print(f'bmax guess: {b_guess_max}, equation: {eq_at_bmax}')
        if eq_at_bmax >= 0:
            # Found sign change! equation(b_min) < 0, equation(b_max) > 0
            try:
                b_max = brentq(equation, b_min, b_guess_max,
                              args=(v1primeVec, v1primeMag, vBVec, muB, phi, potential_energy, rclose),
                              xtol=1e-5, rtol=1e-5)
                # print(f'bmax found: {b_max}')
                return b_max
            except ValueError:
                return 0
        
        # Still negative - need larger b
        b_guess_max *= 1.01
        

    
    return 0

def unit(x):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x)
    if n == 0:
        raise ValueError("Zero vector can't be normalized.")
    return x / n

def rodrigues_rotate(v, k, angle):
    """
    Rotate vector v about unit axis k by 'angle' (radians).
    """
    v = np.asarray(v, float)
    k = unit(k)
    ca, sa = np.cos(angle), np.sin(angle)
    return v*ca + np.cross(k, v)*sa + k*np.dot(k, v)*(1-ca)

def exit_point_from_scatter(v1primeMag, v1primeVec, bVec, muB, rClose, b):
    """
    Returns r_exit (3-vector) in the B frame.

    Provide either impact parameter b or deflection angle psi.
    v1, v2 are asymptotic velocities (3-vectors).
    """
    a_hyp = -muB / v1primeMag**2  # negative for hyperbola
    e = np.sqrt(1.0 + (b * v1primeMag**2 / muB)**2)
    l = a_hyp * (1 - e**2)
    rp = -a_hyp * (e - 1)   # periapsis distance (positive)

    s_in = unit(v1primeVec)
    n_hat = unit(np.cross(bVec, s_in))  # normal to orbital plane
    theta = np.arccos(1.0/e)  # angle of asymptote with respect to periapsis direction
    # periapsis direction: p_hat is at angle -theta_inf from s_in
    # This is the actual geometrical periapsis direction of the hyperbolic orbit
    p_hat = rodrigues_rotate(s_in, n_hat, -theta)  # periapsis direction
    q_hat = np.cross(n_hat, p_hat)

    theta_entry = np.arccos((l/rClose - 1)/e)  # Solve for theta at r=rClose
    # entry_at_rclose = rClose * np.cos(-theta_entry) * p_hat + rClose * np.sin(-theta_entry) * q_hat
    exit_at_rclose = rClose * np.cos(theta_entry) * p_hat + rClose * np.sin(theta_entry) * q_hat

    # periapsis_pos = rp * p_hat
    # C_pos = (rp + np.abs(a_hyp)) * p_hat

    return exit_at_rclose

def cartesian_to_spherical(x):
    """
    Convert Cartesian coordinates to spherical (r, theta, phi).
    - r: radial distance
    - theta: polar angle (0 at north pole)
    - phi: azimuthal angle in xy-plane from x-axis
    """
    x = np.asarray(x, float)
    r = np.linalg.norm(x)
    if r == 0:
        return 0.0, 0.0, 0.0
    theta = np.arccos(x[2] / r)  # polar angle
    phi = np.arctan2(x[1], x[0])  # azimuthal angle
    return r, theta, phi

import numpy as np

# def capture_cross_section_and_error(b_captured, N, bmin, bmax, conf=0.95):
#     """
#     Parameters
#     ----------
#     b_captured : array-like
#         Impact parameters of captured trials only.
#     N : int
#         Total number of trials (fixed, e.g. 1000).
#     bmin, bmax : float
#         Sampling interval in impact parameter.
#     conf : float
#         Confidence level, default 0.95.

#     Returns
#     -------
#     sigma_hat : float
#         Estimated capture cross section.
#     se : float
#         Standard error of the estimator.
#     ci : tuple
#         Symmetric normal-approx confidence interval.
#     """
#     b_captured = np.asarray(b_captured, dtype=float)

#     S1 = np.sum(b_captured)
#     S2 = np.sum(b_captured**2)

#     C = 2.0 * np.pi * (bmax - bmin)

#     sigma_hat = C * S1 / N

#     if N < 2:
#         return sigma_hat, np.nan, (np.nan, np.nan)

#     se = C * np.sqrt((S2 - S1**2 / N) / (N * (N - 1)))

#     # Normal approximation
#     z = 1.0 if np.isclose(conf, 0.68) else 1.96 if np.isclose(conf, 0.95) else None
#     if z is None:
#         from scipy.stats import norm
#         z = norm.ppf(0.5 + conf / 2)

#     ci = (sigma_hat - z * se, sigma_hat + z * se)

#     return sigma_hat, se, ci
def capture_cross_section_and_error(
    b_cap, b_mins, b_maxs, n_sampled, conf=0.95
    ):
    b_cap_au = (np.asarray(b_cap) * u.m).to(u.au).value
    b_mins_au = (np.asarray(b_mins) * u.m).to(u.au).value
    b_maxs_au = (np.asarray(b_maxs) * u.m).to(u.au).value

    X_cap = 2 * np.pi * (b_maxs_au - b_mins_au) * b_cap_au  # au^2

    S1 = np.sum(X_cap)
    S2 = np.sum(X_cap**2)

    sigma_hat = S1 / n_sampled

    se = np.sqrt((S2 - S1**2 / n_sampled) / (n_sampled * (n_sampled - 1)))

    if np.isclose(conf, 0.68):
        z = 1.0
    elif np.isclose(conf, 0.95):
        z = 1.96
    else:
        from scipy.stats import norm
        z = norm.ppf(0.5 + conf / 2)

    ci = (sigma_hat - z * se, sigma_hat + z * se)

    return sigma_hat, se, ci

def rate_and_error(cap_rate, sigma_hat, sigma_se, conf=[0.68, 0.95]):
    """
    Parameters
    ----------
    v : float
        Impact parameter.
    sigma_hat : float
        Estimated capture cross section.
    conf : float
        Confidence level, default 0.95.
    Returns
    -------
    mean_rate : float
        Mean capture rate.
    se : float
        Standard error of the mean.
    ci : tuple
        Symmetric normal-approx confidence interval.
    """
    # v_inf_au_yr = np.asarray(v_inf_au_yr, dtype=float)
    # vdm_au_yr = (220e3*u.m/u.s).to(u.au/u.yr).value
    # f_approx = (np.sqrt(2/np.pi)) * (v_inf_au_yr**2 / (vdm_au_yr**3)) * np.exp(- (v_inf_au_yr)**2/(2*vdm_au_yr**2))
    mean_rate = np.asarray(cap_rate)
    sigma_hat = np.asarray(sigma_hat)
    Ak = cap_rate / sigma_hat  # Assuming cap_rate is proportional to sigma_hat, we can estimate the proportionality constant Ak
    se = Ak * sigma_se
    z = 1.0 if np.isclose(conf, 0.68) else 1.96 if np.isclose(conf, 0.95) else None
    if z is None:
        from scipy.stats import norm
        z = norm.ppf(0.5 + conf / 2)
    ci = (mean_rate - z * se, mean_rate + z * se)

    return mean_rate, se, ci

def gl_effective_weights(x, n=100, exclude_zeros=False):
    """
    Return effective linear weights alpha_k such that
    integrate_gauss_legendre(x, y, n=n, exclude_zeros=exclude_zeros)
    == sum_k alpha_k * y_k
    for any y, up to numerical precision.
    """
    x = np.asarray(x, dtype=float)
    m = len(x)
    alpha = np.zeros(m)

    for k in range(m):
        e = np.zeros(m)
        e[k] = 1.0
        alpha[k] = integrate_gauss_legendre(x, e, n=n, exclude_zeros=exclude_zeros)

    return alpha

def integrated_rate_and_error(x, rate_per_v, rate_err_per_v, n_gl=100, q_vals=None):
    x = np.asarray(x, dtype=float)
    rate_per_v = np.asarray(rate_per_v, dtype=float) 
    rate_err_per_v = np.asarray(rate_err_per_v, dtype=float)

    alpha = gl_effective_weights(x, n=n_gl, exclude_zeros=False)

    if q_vals is not None:
        # e.g. 16th, 50th, 84th percentile values of rate_per_v
        qvals = weighted_quantile(rate_per_v, q_vals, weights=alpha)

    total_rate = np.sum(alpha * rate_per_v)
    total_err = np.sqrt(np.sum((alpha * rate_err_per_v)**2))

    ci68 = (total_rate - total_err, total_rate + total_err)
    ci95 = (total_rate - 1.96 * total_err, total_rate + 1.96 * total_err)

    return total_rate, total_err, ci68, ci95, alpha, qvals if q_vals is not None else None

# ...existing code...
def weighted_quantile(values, quantiles, weights=None):
    values = np.asarray(values, dtype=float)
    q = np.asarray(quantiles, dtype=float)

    if weights is None:
        return np.quantile(values, q)

    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(w) & (w > 0)
    v = values[mask]
    w = w[mask]

    idx = np.argsort(v)
    v = v[idx]
    w = w[idx]

    cdf = np.cumsum(w) / np.sum(w)
    return np.interp(q, cdf, v)
# ...existing code...

def dm_density_profile_milkyway(r):
    """
    Einasto density profile for the Milky Way.
    

    Parameters
    ----------
    r : float or array-like
        Galactocentric radius in kpc.

    Returns
    -------
    float or array-like
        Density at radius r in units of M_sun/kpc^3.
    """
    r = np.asarray(r, dtype=float) * u.kpc
    ## From Lim 2025 best fit. local rho = 0.47
    # rho_0 = 23.5 * 1e-2 * u.M_sun / u.pc**3  # Characteristic density
    # h = 3.6 * u.kpc  # Hubble parameter in units of H_0
    # beta = 1.1
    # rho_0 = rho_0.to(u.M_sun / u.kpc**3)
    # density_in_kpc3 = rho_0 / (((r / h)**beta) * (1 + r / h)**(3-beta))
    # density_in_au3 = density_in_kpc3.to(u.M_sun / u.au**3)

    ## Ou et 2024 best fit. einasto profile with local rho = 0.447
    # rho_0 = 0.447 * u.M_sun / u.pc**3  # Characteristic density
    M0 = 0.62 * 1e11 * u.M_sun  # Normalization mass
    rs = 3.86 * u.kpc  # Scale radius
    alpha = 0.91 # Einasto shape parameter
    rho = (M0 / (4 * np.pi * rs**3)) * np.exp(-(r / rs)**alpha)

    return rho

# def baryonic_density_profile_milkyway(r):
#     """
#     Baryonic density profile for the Milky Way (disk + bulge + dust + gas).
    
#     Includes exponential disk, Hernquist bulge, warm dust, cold dust, HI gas, and H2 gas.

#     Parameters
#     ----------
#     r : float or array-like
#         Galactocentric radius in kpc.

#     Returns
#     -------
#     float or array-like
#         Baryonic density at radius r in units of M_sun/kpc^3.
#     """
    
#     def _double_exponential_density(M0, r, R_d, z,  z_d):
#         """Density for a double exponential disk."""
#         return (M0 / (4 * np.pi * R_d**2 * z_d)) * np.exp(-r / R_d - np.abs(z) / z_d)
    
#     def _hernquist_density(M_bulge, r_b, r):
#         """Density for a Hernquist bulge."""
#         return (M_bulge * r_b) / ((2 * np.pi) * r * (r_b + r)**3)

#     r = np.asarray(r, dtype=float) * u.kpc
    
#     total_density = 0
#     z = 1 * u.kpc
#     # ===== DISK (Exponential) =====
#     M_disc = 3.65e10 * u.M_sun  # Total disk mass
#     R_d = 2.35 * u.kpc  # Scale radius
#     z_d = 0.14 * u.kpc  # Scale height
#     disk_density = _double_exponential_density(M_disc, r, R_d, z=z, z_d=z_d)
    
#     # ===== BULGE (Hernquist profile) =====
#     M_bulge = 1.55e10 * u.M_sun  # Total bulge mass
#     a_bulge = 0.70 * u.kpc  # Scale radius
#     bulge_density = _hernquist_density(M_bulge, a_bulge, r)
    
#     # ===== WARM DUST (Exponential) =====
#     M_warm_dust = 2.20e5 * u.M_sun
#     R_wd = 3.30 * u.kpc
#     z_wd = 0.09 * u.kpc
#     warm_dust_density = _double_exponential_density(M_warm_dust, r, R_wd, z=z, z_d=z_wd)
    
#     # ===== COLD DUST (Exponential) =====
#     M_cold_dust = 7.00e7 * u.M_sun
#     R_cd = 5.00 * u.kpc
#     z_cd = 0.10 * u.kpc
#     cold_dust_density = _double_exponential_density(M_cold_dust, r, R_cd, z=z, z_d=z_cd)
    
#     # ===== HI GAS (Exponential) =====
#     M_HI = 8.20e9 * u.M_sun
#     R_HI = 18.24 * u.kpc
#     z_HI = 0.52 * u.kpc
#     HI_density = _double_exponential_density(M_HI, r, R_HI, z=z, z_d=z_HI)
    
#     # ===== H2 GAS (Exponential) =====
#     M_H2 = 1.30e9 * u.M_sun
#     R_H2 = 2.57 * u.kpc
#     z_H2 = 0.08 * u.kpc
#     HII_density = _double_exponential_density(M_H2, r, R_H2, z=z, z_d=z_H2)
    
#     # total_density = disk_density + bulge_density + warm_dust_density + cold_dust_density + HI_density + HII_density

#     return (disk_density, bulge_density, warm_dust_density, cold_dust_density, HI_density, HII_density)
def _hernquist_density(M_bulge, r_b, r):
    """Density for a Hernquist bulge. Returns M_sun/kpc^3."""
    M_bulge_val = M_bulge.value
    r_b_val = r_b.value
    r_val = r  # r should be unitless (kpc) already from r_array
    rho = (M_bulge_val * r_b_val) / ((2 * np.pi) * r_val * (r_b_val + r_val)**3)
    return rho * u.M_sun / u.kpc**3

def baryonic_mass_and_velocity_milkyway(r, r_in=None, return_components=False):
    """
    Baryonic mass profile for the Milky Way (disk + bulge + dust + gas).

    Includes exponential disk, Hernquist bulge, warm dust, cold dust, HI gas, and H2 gas.

    Parameters
    ----------
    r : float or array-like
        Galactocentric radius in kpc.

    Returns
    -------
    float or array-like
        Baryonic density at radius r in units of M_sun/kpc^3.
    """
    
    # r = np.asarray(r, dtype=float) * u.kpc
    
    total_density = 0
    # ===== DISK (Exponential) =====
    M_disc = 3.65e10 * u.M_sun  # Total disk mass
    R_d = 2.35 * u.kpc  # Scale radius
    z_d = 0.14 * u.kpc  # Scale height
    disk_mass = mass_enclosed_disk(r, R_d, M_disc, r_in)
    disk_velocity = circular_velocity_disk(r, M_disc, R_d)  
    
    # ===== BULGE (Hernquist profile) =====
    M_bulge = 1.55e10 * u.M_sun  # Total bulge mass
    a_bulge = 0.70 * u.kpc  # Scale radius
    from functools import partial

    bulge_density_func = partial(_hernquist_density, M_bulge, a_bulge)
    bulge_mass = mass_enclosed_spherical(r, bulge_density_func, r_in)
    bulge_velocity = circular_velocity_spherical(r, bulge_density_func)

    # ===== WARM DUST (Exponential) =====
    M_warm_dust = 2.20e5 * u.M_sun
    R_wd = 3.30 * u.kpc
    z_wd = 0.09 * u.kpc
    warm_dust_mass = mass_enclosed_disk(r, R_wd, M_warm_dust, r_in)
    warm_dust_velocity = circular_velocity_disk(r, M_warm_dust, R_wd)
    
    # ===== COLD DUST (Exponential) =====
    M_cold_dust = 7.00e7 * u.M_sun
    R_cd = 5.00 * u.kpc
    z_cd = 0.10 * u.kpc
    cold_dust_mass = mass_enclosed_disk(r, R_cd, M_cold_dust, r_in)
    cold_dust_velocity = circular_velocity_disk(r, M_cold_dust, R_cd)

    # ===== HI GAS (Exponential) =====
    M_HI = 8.20e9 * u.M_sun
    R_HI = 18.24 * u.kpc
    z_HI = 0.52 * u.kpc
    HI_mass = mass_enclosed_disk(r, R_HI, M_HI, r_in)
    HI_velocity = circular_velocity_disk(r, M_HI, R_HI)

    # ===== H2 GAS (Exponential) =====
    M_H2 = 1.30e9 * u.M_sun
    R_H2 = 2.57 * u.kpc
    z_H2 = 0.08 * u.kpc
    H2_mass = mass_enclosed_disk(r, R_H2, M_H2, r_in)
    H2_velocity = circular_velocity_disk(r, M_H2, R_H2)

    # total_density = disk_density + bulge_density + warm_dust_density + cold_dust_density + HI_density + HII_density

    total_mass = disk_mass + bulge_mass + warm_dust_mass + cold_dust_mass + HI_mass + H2_mass
    total_velocity = np.sqrt(disk_velocity**2 + bulge_velocity**2 + warm_dust_velocity**2 + cold_dust_velocity**2 + HI_velocity**2 + H2_velocity**2)
    if return_components:
        return total_mass, total_velocity, (disk_mass, bulge_mass, warm_dust_mass, cold_dust_mass, HI_mass, H2_mass)
    return total_mass, total_velocity  

def mass_enclosed_spherical(r, density_profile, r_in=None):
    """
    Mass enclosed within radius r for the Milky Way by integrating the density profile.

    Parameters
    ----------
    r : float or array-like
        Galactocentric radius in kpc (unitless).
    density_profile : callable
        Density function rho(r_val) returning density in M_sun/kpc^3

    Returns
    -------
    float
        Mass enclosed within radius r in units of M_sun.
    """
    if r_in is not None:
        r_in = r_in.value if hasattr(r_in, 'value') else r_in
        r_array = np.linspace(r_in, r, 1000)  # unitless values in kpc
    else:
        r_array = np.linspace(0.001, r, 1000)  # unitless values in kpc
    rho = density_profile(r_array)  # density in M_sun/kpc^3
    # Ensure rho has proper units
    if not hasattr(rho, 'unit'):
        rho = rho * u.M_sun / u.kpc**3
    r_array_kpc = r_array * u.kpc
    integrand = 4 * np.pi * r_array_kpc**2 * rho  # has units M_sun/kpc
    mass_enclosed = integrate_trapezoidal(r_array, integrand.value) * u.M_sun
    return mass_enclosed.value

def circular_velocity_spherical(r, density_profile):
    """
    Circular velocity profile for a spherically symmetric mass distribution.

    Parameters
    ----------
    r : float
        Galactocentric radius in kpc (unitless).
    density_profile : callable
        Density function rho(r_val) returning density in M_sun/kpc^3

    Returns
    -------
    float
        Circular velocity at radius r in km/s (unitless).
    """
    G = const.G.to(u.kpc**3 / (u.M_sun * u.yr**2))  # Keep units
    M_enclosed = mass_enclosed_spherical(r, density_profile)  # returns unitless M_sun
    v_sq = G * (M_enclosed * u.M_sun) / (r * u.kpc)
    v_circ = np.sqrt(v_sq).to(u.km / u.s)
    return v_circ.value

def circular_velocity_disk(R, M, Rd):
    """
    Circular velocity profile for the Milky Way disk component.

    Parameters
    ----------
    R : float
        Galactocentric radius in kpc (unitless).
    M : Quantity
        Disk mass with units (M_sun).
    Rd : Quantity
        Disk scale radius with units (kpc).

    Returns
    -------
    float
        Circular velocity at radius R in km/s.
    """
    from scipy.special import iv, kv
    G = const.G.to(u.kpc**3 / (u.M_sun * u.yr**2))  # Keep units
    R_kpc = R * u.kpc  # Convert R to quantity with units
    y = R_kpc / (2 * Rd)
    Sigma0 = M / (2 * np.pi * Rd**2)
    v_sq = 4 * np.pi * G * Sigma0 * Rd * y.value**2 * (
        iv(0, y.value) * kv(0, y.value)
        - iv(1, y.value) * kv(1, y.value)
    )
    return np.sqrt(v_sq).to(u.km / u.s).value


def mass_enclosed_disk(R, Rd, M, r_in=None):
    """
    Mass enclosed within radius R for the Milky Way disk component.

    Parameters
    ----------
    R : float
        Galactocentric radius in kpc (unitless).
    Rd : Quantity
        Disk scale radius with units (kpc).
    M : Quantity
        Total disk mass with units (M_sun).

    Returns
    -------
    float
        Mass enclosed in M_sun (unitless).
    """
    M_val = M.value
    Rd_val = Rd.value
    if r_in is not None:
        r_in_val = r_in.value if hasattr(r_in, 'value') else r_in
        M_val_r_in = M_val * (1 - np.exp(-r_in_val / Rd_val) * (1 + r_in_val / Rd_val))

        return M_val * (1 - np.exp(-R / Rd_val) * (1 + R / Rd_val)) - M_val_r_in
    else:
        return M_val * (1 - np.exp(-R / Rd_val) * (1 + R / Rd_val))

def dark_matter_mass_and_velocity_milkyway(r):
    """
    Dark matter mass and circular velocity profiles for the Milky Way.

    Parameters
    ----------
    r : float or array-like
        Galactocentric radius in kpc.

    Returns
    -------
    tuple of arrays
        (M_dm, v_dm) where M_dm is the dark matter mass enclosed within radius r in M_sun,
        and v_dm is the circular velocity due to dark matter at radius r in km/s.
    """

    M_dm = mass_enclosed_spherical(r, dm_density_profile_milkyway)
    v_dm = circular_velocity_spherical(r, dm_density_profile_milkyway)
    return M_dm, v_dm

def circular_velocity_milkyway(r):
    """
    Total circular velocity profile for the Milky Way, including both baryonic and dark matter contributions.

    Parameters
    ----------
    r : float or array-like
        Galactocentric radius in kpc.

    Returns
    -------
    float or array-like
        Total circular velocity at radius r in km/s.
    """
    _, v_baryon = baryonic_mass_and_velocity_milkyway(r)
    _, v_dm = dark_matter_mass_and_velocity_milkyway(r)
    v_total = np.sqrt(v_baryon**2 + v_dm**2)
    return v_total

# def v_dm_pdf(v, r):
#     """
#     Dark matter velocity distribution function at radius r.

#     Parameters
#     ----------
#     v : float or array-like
#         Velocity in km/s.
#     r : float or array-like
#         Galactocentric radius in kpc.

#     Returns
#     -------
#     float or array-like
#         Probability density of dark matter particles having velocity v at radius r.
#     """
#     v = np.asarray(v, dtype=float) * u.km / u.s
#     r = np.asarray(r, dtype=float) * u.kpc

#     v_circ = circular_velocity_milkyway(r)
#     sigma_v = v_circ / np.sqrt(2)  # Velocity dispersion for isotropic Maxwellian

#     # Maxwell-Boltzmann distribution (truncated at escape velocity)
#     f_v = (v**2 * np.exp(-v**2 / (2 * sigma_v**2))) / (sigma_v**3 * np.sqrt(2 * np.pi))
    
    # return f_v.to(1/(u.km/u.s)).value

def v_pbh_pdf(v_pbh, r, v_dm=None):
    """
    Local dark matter velocity distribution function at the Sun's position.

    Parameters
    ----------
    v : float or array-like
        Velocity in km/s with units.

    Returns
    -------
    float or array-like
        Probability density of dark matter particles having velocity v at the Sun's position.
    """
    v_inf_au_yr = (v_pbh*(u.km/u.s)).to(u.au/u.yr)
    if v_dm is not None:
        v_dm_au_yr = v_dm
    else:
        v_dm_km_s = circular_velocity_milkyway(r) * (u.km/u.s)
        v_dm_au_yr = v_dm_km_s.to(u.au / u.yr)

    sigma_dm_au_yr = v_dm_au_yr / np.sqrt(2)  # Velocity dispersion for isotropic Maxwellian
    f_approx = (np.sqrt(2/np.pi)) * (v_inf_au_yr**2 / (sigma_dm_au_yr**3)) * np.exp(- (v_inf_au_yr)**2/(2*sigma_dm_au_yr**2))
    return f_approx

def v_pbh_pdf_old(v_pbh, r, v_dm=None):
    """
    Local dark matter velocity distribution function at the Sun's position.

    Parameters
    ----------
    v : float or array-like
        Velocity in km/s with units.

    Returns
    -------
    float or array-like
        Probability density of dark matter particles having velocity v at the Sun's position.
    """
    v_inf_au_yr = (v_pbh*(u.km/u.s)).to(u.au/u.yr)
    if v_dm is not None:
        v_dm_au_yr = v_dm
    else:
        v_dm_au_yr = circular_velocity_milkyway(r) * (u.km/u.s)

    vdm_au_yr = v_dm_au_yr.to(u.au / u.yr)
    sigma_dm_au_yr = vdm_au_yr  # Velocity dispersion for isotropic Maxwellian
    f_approx = (np.sqrt(2/np.pi)) * (v_inf_au_yr**2 / (sigma_dm_au_yr**3)) * np.exp(- (v_inf_au_yr)**2/(2*sigma_dm_au_yr**2))
    return f_approx

def get_Neq_at_r_f(Mpbh, Neq, ri=None, rf=8, num_points=100):
    """
    Compute total Neq across all v bins for a range of PBH fractions.
    Returns a dict of PBH fraction to total Neq.
    """
    # if pbh_fractions is None:
    #     pbh_fractions = [1e-2, 1e-1, 1.0]
    
    # used_bounds_mode = False
    # if isinstance(pbh_fractions, str) and pbh_fractions == 'bounds':
    bound_ids = ['EGRB', '511keV-DeLaTorreLuque2024', 'CMBevap', 'EDGESevap-Mittal2021',
    'Voyager', 'INTEGRAL', 'LeoTevap', 'SuperK', 'Comptel', 'AMS', 'Xrayevap',
    'LyaEvap-Khan2025', 'OGLE-strict', 'OGLE-highcadence', 'SNe', 'M', 'HSC',
    'EROS', 'Icarus', 'Microlensing-LongDuration', 'Quasars-Xray',
    'FRB-Leung2022', 'Radio', 'CMB', 'EDGES', 'X-ray', 'LeoT', 'WideBinaries',
    'UFdwarfs', 'LIGO', 'LIGO-SGWB-O2', 'LIGO-SGWB-O3', 'LIGO-subsolar',
    'GW-Lensing', 'PBH-EMRIs', '3G-GW-1y', '3G-GW-10y', 'SIGWs', 'OGLE-hint',
    'FL-small', 'GRB-parallax', 'WhiteDwarfmicro', 'Xraylensing-eXTP-proj',
    'GECCO']
    bounds_result = get_pbh_fraction_bounds(Mpbh, bound_id=bound_ids, return_all_bounds=False)
    pbh_fractions = [bounds_result['min_fraction'], 1e0]
    # used_bounds_mode = True
    
    total_Neq = Neq
    total_Neq_kpc3 = (total_Neq * u.au**3).to(u.kpc**3)  # convert total Neq to number per kpc^3
    range_kpc3, number_density_in_range_kpc3 = get_pbh_number_density_r(Mpbh, ri=ri, rf=rf, num_points=num_points)
    d = {'total_Neq_kpc3': total_Neq_kpc3.value, 'range_kpc3': range_kpc3, 'number_density_in_range_kpc3': number_density_in_range_kpc3.value}
    
    for i, f in enumerate(pbh_fractions):
        pbh_number_density = number_density_in_range_kpc3 * f  # in number/kpc^3
        neq_range = total_Neq_kpc3 * pbh_number_density  # total Neq scaled by PBH number density
        if i == 0:
            d[f'Neq_pbh_f_bound'] = neq_range.value
            d['bound_fraction'] = f
        else:
            d[f'Neq_pbh_f_full'] = neq_range.value
            d['full_fraction'] = f
    return d
    



def get_pbh_number_density_r(Mpbh, ri, rf, num_points):
    """
    Compute PBH number density across a range of radii.
    Default DM density is in M_sun/au^3 for number density calculation.
    Returns a dict of radius to PBH number density.
    param ri: inner radius in kpc
    param rf: outer radius in kpc
    param num_points: number of points to sample across the radius range
    """
    if ri is not None:
        radii = np.linspace(ri, rf, num_points)
    else:
        radii = rf
    dm_densities = dm_density_profile_milkyway(radii)  # in M_sun/kpc^3
    mpbh = (Mpbh * u.kg).to(u.M_sun)  # default to 1e-16 M_sun if not specified
    n_pbh = dm_densities / mpbh  # in number/kpc^3
    return radii, n_pbh


def get_pbh_fraction_bounds(Mpbh, bound_id, return_all_bounds=False):
    """
    Get the PBH fraction bounds for the mC mass from PlotPBHbounds module.
    
    Parameters
    ----------
    Mpbh : float
        The PBH mass in kg
    bound_id : str or list
        Single bound ID name(s) to query (e.g., 'LIGO', 'Microlensing', etc.)
        See PBHbounds/bounds/ for available bound files.
    return_all_bounds : bool
        If True and bound_id is a list, return all bounds. If False, return minimum.
        
    Returns
    -------
    dict : Contains 'mC_Msun' and bound values 'bound_{bound_id}' for each constraint
    """
    try:
        # Dynamically import tools from PBHbounds
        import sys
        pbhbounds_path = os.path.join(REPO_ROOT, 'PBHbounds')
        if pbhbounds_path not in sys.path:
            sys.path.insert(0, pbhbounds_path)
        import tools
    except ImportError as e:
        raise ImportError(f"Could not import tools from PBHbounds: {e}")
    
    # Convert mC from kg to solar masses
    mC_kg = Mpbh
    if mC_kg is None:
        raise ValueError("Mpbh must be provided to compute mC and query bounds.")
    
    mC_Msun = (mC_kg * u.kg).to(u.M_sun).value
    
    # Ensure bound_id is a list
    if isinstance(bound_id, str):
        bound_ids = [bound_id]
    else:
        bound_ids = list(bound_id)
    
    result = {'mC_kg': mC_kg, 'mC_Msun': mC_Msun}
    bounds_values = []
    
    for bid in bound_ids:
        try:
            # Load bound data from PlotPBHbounds
            m_arr, f_arr = tools.load_bound(bid)
            
            # Interpolate to find f at mC_Msun
            # Handle case where mC is outside the bounds
            if mC_Msun < m_arr.min() or mC_Msun > m_arr.max():
                f_at_mC = np.nan  # Out of range
                status = "out_of_range"
            else:
                # Use log-log interpolation for better accuracy
                f_at_mC = np.interp(np.log10(mC_Msun), np.log10(m_arr), np.log10(f_arr), left=np.nan, right=np.nan)
                f_at_mC = 10**f_at_mC if not np.isnan(f_at_mC) else np.nan
                status = "valid"
            
            result[f'bound_{bid}'] = f_at_mC
            result[f'status_{bid}'] = status
            bounds_values.append(f_at_mC)
            
        except Exception as e:
            print(f"Warning: Could not load bound '{bid}': {e}")
            result[f'bound_{bid}'] = np.nan
            result[f'status_{bid}'] = "error"
    
    # If multiple bounds and not returning all, compute summary statistics
    if len(bounds_values) > 1 and not return_all_bounds:
        valid_bounds = [b for b in bounds_values if not np.isnan(b)]
        if valid_bounds:
            result['min_fraction'] = np.min(valid_bounds)
            result['max_fraction'] = np.max(valid_bounds)
            result['mean_fraction'] = np.mean(valid_bounds)
    
    return result

def star_number_at_r(r, dr, M_star=1.0, fraction = 1.0):
    """
    Estimate the number of stars at radius r, given a stellar mass M_star.
    
    Parameters
    ----------
    r : float
        Galactocentric radius in kpc.
    M_star : float
        Average mass of a star in solar masses (default 1.0 M_sun).
        
    Returns
    -------
    N_stars : float
        Estimated number of stars at radius r.
    """

    r_in, r_out = r - dr/2, r + dr/2
    if r_in < 0:
        r_in = 0.001
    M_disk, M_bulge = baryonic_mass_and_velocity_milkyway(r_out, r_in, return_components=True)[2][:2]
    N_bulge = (M_bulge / M_star) * fraction
    N_disk = (M_disk / M_star) * fraction
    return N_bulge + N_disk

def star_number_at_ring(r, M_star=1.0, fraction = 1.0):
    """
    Estimate the number of stars in a ring at radius r with width dr, given a stellar mass M_star.
    
    Parameters
    ----------
    r : float
        Galactocentric radius in kpc.
    M_star : float
        Average mass of a star in solar masses (default 1.0 M_sun).
        
    Returns
    -------
    N_stars : float
        Estimated number of stars in the ring at radius r.
    """
    # Stellar density profile (e.g., exponential disk)
    # For simplicity, we can use an exponential disk model: rho(r) = rho_0 * exp(-r/R_d)
    # where rho_0 is the central density and R_d is the scale length.
    
    rho_100 = 0.025  # Central stellar density in M_sun/pc^3 at solar neighborhood sphere of r=100pc Lutsenko 2025
    r_100 = 100 # pc

    M_100 = rho_100 * (4/3) * np.pi * (r_100)**3  # Mass in M_sun within 100 pc sphere
    N_100 = M_100 / M_star  # Number of stars in 100 pc sphere
    
    # Convert r to pc for density calculation
    ring_at_SN = 2 * np.pi * r * 1000  # circumference of ring at solar neighborhood in pc
    spheres_in_ring = ring_at_SN / (2 * r_100)  # number of 100 pc spheres that fit in the ring
    
    N_stars = N_100 * spheres_in_ring * fraction  # Total number of stars in the ring
    
    return N_stars

def star_number_at_disc(r_in, r_out, M_star=1.0, fraction = 1.0):
    """
    Estimate the number of stars in a disc between radii r_in and r_out, given a stellar mass M_star.
    
    Parameters
    ----------
    r_in : float
        Inner radius of the disc in kpc.
    r_out : float
        Outer radius of the disc in kpc.
    M_star : float
        Average mass of a star in solar masses (default 1.0 M_sun).
        
    Returns
    -------
    N_stars : float
        Estimated number of stars in the disc between r_in and r_out.
    """
    # Stellar density profile (e.g., exponential disk)
    # For simplicity, we can use an exponential disk model: rho(r) = rho_0 * exp(-r/R_d)
    # where rho_0 is the central density and R_d is the scale length.
    
    rho_100 = 0.025  # Central stellar density in M_sun/pc^3 at solar neighborhood sphere of r=100pc Lutsenko 2025
    r_100 = 100 # pc

    M_100 = rho_100 * (4/3) * np.pi * (r_100)**3  # Mass in M_sun within 100 pc sphere
    N_100 = M_100 / M_star  # Number of stars in 100 pc sphere
    
    # Convert radii to pc for density calculation
    r_in_pc = r_in * 1000  # kpc to pc
    r_out_pc = r_out * 1000  # kpc to pc
    
    area_disc = np.pi * (r_out_pc**2 - r_in_pc**2)  # Area of the disc in pc^2
    N_stars = N_100 * (area_disc / (np.pi * r_100**2)) * fraction  # Scale number of stars by area ratio and fraction
    return N_stars

def star_number_galactic_bulge(r, M_star=1.0, fraction = 1.0):
    """
    Estimate the number of stars in the galactic bulge within radius r, given a stellar mass M_star.
    
    Parameters
    ----------
    r : float
        Galactocentric radius in kpc.
    M_star : float
        Average mass of a star in solar masses (default 1.0 M_sun).
        
    Returns
    -------
    N_stars : float
        Estimated number of stars in the galactic bulge within radius r.
    """
    # Bulge density profile (e.g., Hernquist profile)
    # For simplicity, we can use a Hernquist profile: rho(r) = (M_bulge * r_b) / (2 * pi * r * (r_b + r)^3)
    
    M_bulge = 1.55e10  # Total bulge mass in M_sun Bland-hawthorn2016
    a_bulge = 0.70  # Scale radius in kpc
    
    # Convert r to kpc for density calculation
    r_kpc = r  # already in kpc
    
    # Calculate mass enclosed within radius r using Hernquist profile
    def hernquist_mass_enclosed(r):
        return M_bulge * (r_kpc**2 / (r_kpc + a_bulge)**2)
    
    M_enclosed = hernquist_mass_enclosed(r_kpc)  # Mass enclosed within radius r in M_sun
    N_stars = (M_enclosed / M_star) * fraction  # Number of stars based on enclosed mass and average star mass
    return N_stars


    