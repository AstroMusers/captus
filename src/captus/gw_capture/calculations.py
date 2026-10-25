import numpy as np
from scipy.optimize import brentq
import captus.utils.misc as misc
from astropy import constants as const
import astropy.units as u
from captus.utils.calculations import (
    get_Neq_at_r_f,
    integrate_gauss_legendre,
    orbital_period as _orbital_period_si,
    v_pbh_pdf,
)

G = const.G.value
def galactic_dm_maxwell_dist(v, r=8):
    if v <= 550:
        f_vinf = v_pbh_pdf(v, r=r)
        return f_vinf.value

    else:
        f_vinf = 0
        return f_vinf
    
def neq_r_f(Mpbh, Neq, ri=None, rf=8):
    n = get_Neq_at_r_f(Mpbh, Neq, ri=ri, rf=rf, num_points=1)
    return n

def coalescence_time(v_inf, b, b_max, M_pbh, M_star): #Radial 2025
    # b_prop = b/b_max
    # # M_pbh = M_pbh * G / const.c.value**2
    # # M_star = M_star * G / const.c.value**2
    Msun = const.M_sun.value 
    # # v_inf = v_inf / const.c.value
    # mu = (M_pbh * M_star) / (M_pbh + M_star)
    # nu = mu / (M_pbh + M_star)
    # b_max = (19*np.pi*(nu)/3)**(1/7) * (G * (M_pbh + M_star)/v_inf**2) * (const.c.value/v_inf)**(5/7) # * const.G.value**(2/7) * const.c.value**(-5/7)
    # b = b_prop * b_max
    if b_max <= 0 or b < 0:
        raise ValueError("Impact parameters must satisfy b_max > 0 and b >= 0")
    impact_ratio = b / b_max
    if impact_ratio > 1:
        raise ValueError("Coalescence time is defined only for b <= b_max")
    if impact_ratio == 1:
        return np.inf
    T = (1.08*u.day).to(u.s).value * ((M_pbh + M_star)/Msun) * ((v_inf/1e5)**-3) * np.sqrt(impact_ratio**21 / (1 - impact_ratio**7))
    M_total = M_pbh + M_star
    # return (19*np.pi/85) * (G * M_total) / (v_inf**3) * \
    #        np.sqrt((b/b_max)**21 / (1 - (b/b_max)**7))
    return T

def orbital_energy(M_pbh, M_star, v_inf, b, e):
    mu = (M_pbh * M_star) / (M_pbh + M_star)

    L, r = orbital_momentum(e, mu, b, v_inf)

    KE = 0.5 * mu * v_inf**2
    LE = 0.5 * (L**2) / (mu * r**2)
    PE = - G * mu * (M_star + M_pbh) / r
    return KE + PE + LE

def orbital_momentum(e, mu, b, v_inf):
    L = mu * b * v_inf
    return L, np.inf

def r_minimum(M_star, M_pbh, v_inf, e):
    return G * (M_star + M_pbh) * (e-1) / (v_inf**2)

def b_minimum(M_star, M_pbh, R_star, v_inf):
    R_collision = R_star + (2 * G * M_pbh / const.c.value**2)
    return np.sqrt(R_collision**2 + (2 * G * (M_star + M_pbh) * R_collision) / v_inf**2)

def p_e(e):
    return ((e + 1)**-3.5) * ((np.arccos(-1/e) * (24 + 73*e**2 + ((37/4)*e**4))) + (602 + 673*e**2) * np.sqrt(e**2 - 1)/12)

# def eccentricity(b, M_pbh, M_star, v_inf):
#     return 1 + 1e-9 * (b/(1e5))**2 * (v_inf/(2.2e5))**4 * (M_star/const.M_sun.value)**-2

def eccentricity_original(b, M_pbh, M_star, v_inf): # Hyperbolic orbit eccentricity pre GW emission
    return np.sqrt(1 + (b**2 * v_inf**4) / (G**2 * (M_star + M_pbh)**2))

def eccentricity_v2(E, L, M_pbh, M_star): # Bound orbit eccentricity post capture
    mu = (M_pbh * M_star) / (M_pbh + M_star)
    GM = G * (M_pbh + M_star)
    val = 1.0 + (2.0 * E * L**2) / (mu**3 * GM**2)
    # numerical safety
    return np.sqrt(np.clip(val, 0.0, None))

def delta_E_GW(e, b, M_pbh, M_star, v_inf):

    pre_factor = ((8/15)  * (M_pbh**2 * M_star**2)) / ((M_star+ M_pbh)**3 * const.c.value**5)
    # print('pre factor:', pre_factor)
    return pre_factor * (p_e(e) / (e - 1)**3.5) * v_inf**7

def equation(b, M_pbh, M_star, v_inf):
    # Returns positive when GW loss < KE (no capture)
    # Returns negative when GW loss > KE (capture possible)
    e = eccentricity_original(b, M_pbh, M_star, v_inf) 
    # return -delta_E_GW(e, b, M_pbh, M_star, v_inf) + orbital_energy(M_pbh, M_star, v_inf, b, e)
    # E_initial = 0.5 * (M_pbh * M_star) / (M_pbh + M_star) * v_inf**2
    mu = (M_pbh * M_star) / (M_pbh + M_star)
    E_initial = 0.5 * mu * v_inf**2
    return -delta_E_GW(e, b, M_pbh, M_star, v_inf) + orbital_energy(M_pbh, M_star, v_inf, b, e)

def compute_b_max(M_pbh, M_star, v_inf, b_min, limit_in_m):
    """
    Find maximum b where |GW energy loss| ≥ kinetic energy
    
    At b_min: GW loss is very large (close encounter) → equation < 0
    As b increases: GW loss decreases → equation approaches 0 then becomes positive
    
    We want the b where equation = 0 (transition point)
    """
    e = eccentricity_original(b_min, M_pbh, M_star, v_inf)
    # Check if capture is possible at b_min
    eq_at_bmin = equation(b_min, M_pbh, M_star, v_inf)
    
    if eq_at_bmin > 0:
        # Even at closest approach, GW loss < KE → no capture possible
        return b_min
    
    # Find where equation crosses zero (GW loss = KE)
    b_guess_max = b_min  # Start with 10x b_min
    
    # Expand search until we bracket the root
    while b_guess_max < limit_in_m:
        eq_at_bmax = equation(b_guess_max, M_pbh, M_star, v_inf)
        # print(f'bmax guess: {b_guess_max}, equation: {eq_at_bmax}')
        if eq_at_bmax >= 0:
            # Found sign change! equation(b_min) < 0, equation(b_max) > 0
            try:
                b_max = brentq(equation, b_min, b_guess_max,
                              args=(M_pbh, M_star, v_inf),
                              xtol=1e-5, rtol=1e-5)
                # print(f'bmax found: {b_max}')
                return b_max
            except ValueError:
                return 0
        
        # Still negative - need larger b
        b_guess_max += b_min/100
        

    
    return 0

def compute_capture_crossec(b_min, b_max):
    if b_max <= b_min:
        return 0
    else:
        return np.pi * (b_max**2 - b_min**2)
    
def compute_semimajor_axis(b_ave, M_pbh, M_star, v_inf):
    E = equation(b_ave, M_pbh, M_star, v_inf)  # this must be the *post-loss* orbital energy
    mu = (M_pbh * M_star) / (M_pbh + M_star)

    if not np.isfinite(E) or E >= 0:   # not bound
        return np.nan, np.nan

    a = -G * (M_pbh + M_star) * mu / (2.0 * E)

    # your choice of angular momentum model:
    L = mu * b_ave * v_inf
    e = eccentricity_v2(E, L, M_pbh, M_star)

    return a, e

def orbital_period(m1, m2, semimajax):
    """Return Keplerian periods in years for SI masses and semimajor axes in AU."""

    semimajor_axis_m = (np.asarray(semimajax) * u.au).to_value(u.m)
    period_seconds = _orbital_period_si(
        semimajor_axis_m,
        np.asarray(m1),
        np.asarray(m2),
    )
    return (period_seconds * u.s).to_value(u.yr)
