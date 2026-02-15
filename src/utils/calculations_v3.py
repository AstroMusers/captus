import astropy.constants as const
from math import pi
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy import linalg
from scipy.optimize import brentq, minimize_scalar


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

def v_1_vec(v1Mag, lambda_1, beta_1):
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
    e = np.sqrt(1 + (2*specE2*specL2**2)/(muA**2))
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

def integrate_gauss_legendre(x, y, n=100, exclude_zeros=True):
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
    v2Vec = v_2_vec(v1primeVec, v1primeMag, vBVec, muB, b, phi)
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
        b_guess_max *= 1.1
        

    
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

def exit_point_from_scatter(v1primeVec, v2primeVec, bVec, muB, rClose, b=None, psi=None):
    """
    Returns r_exit (3-vector) in the B frame.

    Provide either impact parameter b or deflection angle psi.
    v1, v2 are asymptotic velocities (3-vectors).
    """
    s_in  = unit(v1primeVec)
    v_inf = np.linalg.norm(v1primeVec)
    e = np.sqrt(1.0 + (b * v_inf**2 / muB)**2)


    # semi-latus rectum
    p = (b**2 * v_inf**2) / muB

    # theta_exit
    cos_th = (p / rClose - 1.0) / e
    if abs(cos_th) > 1.0:
        raise ValueError(f"Invalid geometry: cos(theta_exit)={cos_th} not in [-1,1].")
    theta_exit = np.arccos(cos_th)  # outgoing branch: + arccos

    # plane normal
    n = np.cross(bVec, s_in)
    nn = np.linalg.norm(n)
    if nn == 0:
        raise ValueError("v1 and v2 are colinear; scattering plane undefined.")
    n_hat = n / nn

    # theta_infty
    theta_inf = np.arccos(-1.0/e)

    # periapsis direction
    p_hat = rodrigues_rotate(s_in, n_hat, -theta_inf)
    q_hat = np.cross(n_hat, p_hat)

    r_hat_exit  =  np.cos(theta_exit)*p_hat + np.sin(theta_exit)*q_hat
    r_hat_enter =  np.cos(theta_exit)*p_hat - np.sin(theta_exit)*q_hat

    r_exit  = rClose * r_hat_exit
    r_enter = rClose * r_hat_enter
    
    return r_exit

