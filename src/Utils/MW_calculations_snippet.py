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

    ## Ou et 2024 best fit. einasto profile with local rho = 0.447
    # rho_0 = 0.447 * u.M_sun / u.pc**3  # Characteristic density
    M0 = 0.62 * 1e11 * u.M_sun  # Normalization mass
    rs = 3.86 * u.kpc  # Scale radius
    alpha = 0.91 # Einasto shape parameter
    rho = (M0 / (4 * np.pi * rs**3)) * np.exp(-(r / rs)**alpha)

    return rho

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
        Galactocentric radius in kpc (values).
    density_profile : callable
        Density function rho(r_val) returning density in M_sun/kpc^3

    Returns
    -------
    float
        Mass enclosed within radius r in units of M_sun (value).
    """
    if r_in is not None:
        r_in = r_in.value if hasattr(r_in, 'value') else r_in
        r_array = np.linspace(r_in, r, 1000)  # values in kpc
    else:
        r_array = np.linspace(0.001, r, 1000)  # values in kpc
    rho = density_profile(r_array)  # density in M_sun/kpc^3

    if not hasattr(rho, 'unit'):
        rho = rho * u.M_sun / u.kpc**3

    r_array_kpc = r_array * u.kpc
    integrand = 4 * np.pi * r_array_kpc**2 * rho  # M_sun/kpc
    mass_enclosed = integrate_trapezoidal(r_array, integrand.value) * u.M_sun
    return mass_enclosed.value

def circular_velocity_spherical(r, density_profile):
    """
    Circular velocity profile for a spherically symmetric mass distribution.

    Parameters
    ----------
    r : float
        Galactocentric radius in kpc (value).
    density_profile : callable
        Density function rho(r_val) returning density in M_sun/kpc^3

    Returns
    -------
    float
        Circular velocity at radius r in km/s (unitless).
    """
    G = const.G.to(u.kpc**3 / (u.M_sun * u.yr**2)) 
    M_enclosed = mass_enclosed_spherical(r, density_profile)  
    v_sq = G * (M_enclosed * u.M_sun) / (r * u.kpc)
    v_circ = np.sqrt(v_sq).to(u.km / u.s)
    return v_circ.value

def circular_velocity_disk(R, M, Rd):
    """
    Circular velocity profile for the Milky Way disk component.
    Razor thin exponential disk model from Freeman (1970).

    Parameters
    ----------
    R : float
        Galactocentric radius in kpc (value).
    M : Quantity
        Disk mass with units (M_sun).
    Rd : Quantity
        Disk scale radius with units (kpc).

    Returns
    -------
    float
        Circular velocity at radius R in km/s (value).
    """
    from scipy.special import iv, kv
    G = const.G.to(u.kpc**3 / (u.M_sun * u.yr**2))  # Keep units
    R_kpc = R * u.kpc  # Convert R to quantity with units
    y = R_kpc / (2 * Rd)
    Sigma0 = M / (2 * np.pi * Rd**2)
    # iv, kv are modified Bessel functions of the first and second kind
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
        Mass enclosed in M_sun (value).
    """
    M_val = M.value
    Rd_val = Rd.value
    # sigma_0 = M_val / (2 * np.pi * Rd_val**2)
    # M(<R) = 2 * np.pi * sigma_0 * (Rd_val**2  - Rd_val * np.exp(-R / Rd_val) * (Rd_val  + R))
    # using simplified ver. of above formula that directly uses M_val instead of sigma_0
    
    if r_in is not None: # When r_in is provided, compute mass enclosed between r_in and R (for dr calculations)
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

    sigma_dm_au_yr = v_dm_au_yr / np.sqrt(2)  # Velocity disp for isotropic Maxwellian
    f_approx = (np.sqrt(2/np.pi)) * (v_inf_au_yr**2 / (sigma_dm_au_yr**3)) * np.exp(- (v_inf_au_yr)**2/(2*sigma_dm_au_yr**2))
    return f_approx


def get_Neq_at_r_f(Mpbh, Neq, ri=None, rf=8, num_points=100):
    """
    Compute total Neq across all v bins for a range of PBH fractions.
    Returns a dict of PBH fraction to total Neq.
    """

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
    total_Neq_kpc3 = (total_Neq * u.au**3).to(u.kpc**3)  # convert Neq to number per kpc^3
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
    mpbh = (Mpbh * u.kg).to(u.M_sun) 
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

def star_number_at_dr(r, dr, M_star=1.0, fraction = 1.0):
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


