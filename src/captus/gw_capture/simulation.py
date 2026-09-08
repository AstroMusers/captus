import numpy as np
from scipy.optimize import brentq
import captus.utils.misc as misc
from astropy import constants as const
import astropy.units as u
from captus.gw_capture.calculations import *
from captus.gw_capture.plotting import plot_capture_cross_section, plot_coalescence_time, plot_n_eqs, plot_all_grids

G = const.G.value
Hubble_time = 13.8e9 

class GWCapture:
    def __init__(self, config_dict):
        self.name = config_dict.get('name', 'Typical_NS_PBH_Capture')
        self.Rs = config_dict.get('R_star', 11e3)  # Default for typical NS, in meters
        self.Ms = config_dict.get('M_star', 1.4 * const.M_sun.value)  # Default for typical NS, in kg
        self.gridsize = config_dict.get('gridsize', 100)  # Default grid size
        self.M_pbh_values = config_dict.get('M_pbh_values', np.logspace(-8, 1, self.gridsize) * const.M_sun.value)  # Default PBH mass range in kg
        self.v_inf_values = config_dict.get('v_inf_values', np.logspace(3, 6, self.gridsize))  # Default v_inf range in m/s
        self.M_grid, self.V_grid = np.meshgrid(self.M_pbh_values, self.v_inf_values)
        self.r = config_dict.get('r', 8)  # Default distance in kpc for galactic DM distribution
        self.bmax_limit = config_dict.get('bmax_limit', 1e7 * u.au.to(u.m))  # Default limit for b_max in meters
        self.v_inf_values_kms = [v.to(u.km/u.s).value for v in (self.v_inf_values * u.m/u.s)]
        self.m_pbh_values_in_Msun = self.M_pbh_values / const.M_sun.value

    def capture_cross_section(self):
        cross_section_grid = np.zeros_like(self.M_grid)

        b_max_values = np.zeros_like(self.M_grid)
        b_min_values = np.zeros_like(self.M_grid)

        for i in range(self.gridsize):
            for j in range(self.gridsize):
                b_min = b_minimum(self.Ms, self.M_pbh_values[j], self.Rs, self.v_inf_values[i])
                b_min_values[i, j] = b_min
                # print(equation(b_min_values, M_grid[i, j], M_star, V_grid[i, j]))
                b_max = compute_b_max(self.M_pbh_values[j], self.Ms, self.v_inf_values[i], b_min, self.bmax_limit)
                b_max_values[i, j] = b_max
                b_min_au = (b_min * u.m).to(u.au).value
                # print(f'bmin: {b_min_au}')
                # print(f'p(e) at bmin: {p_e(eccentricity_original(b_min, m_pbh_values[j], M_star, v_inf_values[i]))}')
                # print(f'eccentricity at bmin: {eccentricity_original(b_min, m_pbh_values[j], M_star, v_inf_values[i])}')
                # print(f'equation(b_min): {equation(b_min, m_pbh_values[j], M_star, v_inf_values[i])}')
                # print(f'equation at bmax guess: {equation(b_min*1000000, m_pbh_values[j], M_star, v_inf_values[i])}')
                b_max_au = (b_max * u.m).to(u.au).value
                cs = compute_capture_crossec(b_min_au, b_max_au)
                # cs_au = (cs * u.m**2).to(u.au**2).value
                cross_section_grid[i, j] = cs

        self.cross_section_grid = cross_section_grid
        self.b_min_values = b_min_values
        self.b_max_values = b_max_values

    def get_capture_cross_section(self, plot=False):
        if not hasattr(self, 'cross_section_grid'):
            self.capture_cross_section()
        if plot:
            plot_capture_cross_section(self.m_pbh_values_in_Msun, self.v_inf_values_kms, self.cross_section_grid, self.name, self.r)
        return self.cross_section_grid

    def _orbital_parameters(self):

        semaj_grid = np.zeros_like(self.M_grid)
        eccentricity_grid = np.zeros_like(self.M_grid)
        orbital_period_grid = np.zeros_like(self.M_grid)

        for i in range(self.gridsize):
            for j in range(self.gridsize):
                Rb_ave = (self.b_min_values[i, j] + self.b_max_values[i, j]) / 2
                a, e = compute_semimajor_axis(Rb_ave, self.M_grid[i, j], self.Ms, self.V_grid[i, j])
                semaj_grid[i, j] = (a*u.m).to(u.au).value
                eccentricity_grid[i, j] = e
                orbital_period_grid[i, j] = orbital_period(self.M_grid[i, j], self.Ms, semaj_grid[i, j])

        self.semimajor_axis_grid = semaj_grid
        self.eccentricity_grid = eccentricity_grid
        self.orbital_period_grid = orbital_period_grid

    def get_orbital_parameters(self):
        if not hasattr(self, 'semimajor_axis_grid'):
            self._orbital_parameters()
        return self.semimajor_axis_grid, self.eccentricity_grid, self.orbital_period_grid

    def _coalescence_time(self):
        coaltime = np.full_like(self.M_grid, np.nan, dtype=float)

        for j in range(self.gridsize):
            for i in range(self.gridsize):
                v = self.v_inf_values[i]
                b_ave = (self.b_min_values[i, j] + self.b_max_values[i, j]) / 2
                a0, e0 = compute_semimajor_axis(
                    (b_ave), 
                    self.M_grid[i, j],
                    self.Ms,
                    self.V_grid[i, j],
                )
                # Coalescence formula requires a bound ellipse: 0 <= e0 < 1 and a0 > 0
                if (not np.isfinite(a0)) or (not np.isfinite(e0)) or (a0 <= 0) or (e0 < 0) or (e0 >= 1):
                    continue

                t_sec = coalescence_time(self.V_grid[i, j], b_ave, self.b_max_values[i, j], self.M_grid[i, j], self.Ms)  # returns seconds (SI)
                t_yr = (t_sec * u.s).to(u.yr).value
                coaltime[i, j] = t_yr

        self.coalescence_time = coaltime

    def get_coalescence_time(self, plot=False):
        if not hasattr(self, 'coalescence_time'):
            self._coalescence_time()
        if plot:
            plot_coalescence_time(self.m_pbh_values_in_Msun, self.v_inf_values_kms, self.coalescence_time, self.name, self.r)
        return self.coalescence_time

    def _systems_in_eq(self):
        gridsize = self.M_grid.shape[0]
        n_eqs_perpbh = []
        n_eqs_grid = np.zeros_like(self.M_grid)
        neq_grid = np.zeros_like(self.M_grid)
        for j in range(gridsize):
            x_vals = []
            y_vals = []
            for i in range(gridsize):
                # print(m, M_star, R_star, v)
                v = self.v_inf_values[i]
                lifetime = self.coalescence_time[i, j]
                term_rate = 1 / lifetime
                v_km_s = (v * u.m / u.s).to(u.km / u.s).value
                f = galactic_dm_maxwell_dist(v=v_km_s, r=self.r)
                v_inf_auyr = (v * u.m / u.s).to(u.au / u.yr).value
                rate =  f * self.cross_section_grid[i, j] * v_inf_auyr / term_rate
                y_vals.append(rate)
                x_vals.append(v)
                n_eqs_grid[i, j] = rate
                neq_rf = neq_r_f(self.M_grid[i, j], rate, rf=self.r)
                neq_grid[i, j] = neq_rf['Neq_pbh_f_bound']
            n_eq = integrate_gauss_legendre(x_vals, y_vals, n=100)
            # print(f'Capture rate for mPBH = {m_pbh_values[j]}:', n_eq)
            n_eqs_perpbh.append(n_eq)

        self.n_eqs_per_pbh = n_eqs_perpbh
        self.n_eqs = neq_grid

    def get_systems_in_eq(self, plot=False):
        if not hasattr(self, 'n_eqs'):
            self._systems_in_eq()
        if plot:
            plot_n_eqs(self.m_pbh_values_in_Msun, self.v_inf_values_kms, self.n_eqs, self.name, self.r)
        return self.n_eqs

    def get_systems_in_eq_per_pbh(self):
        if not hasattr(self, 'n_eqs_per_pbh'):
            self._systems_in_eq()
        return self.n_eqs_per_pbh

    def get_all_plots(self):
        if not hasattr(self, 'cross_section_grid'):
            self.capture_cross_section()
        if not hasattr(self, 'semimajor_axis_grid'):
            self._orbital_parameters()
        if not hasattr(self, 'coalescence_time'):
            self._coalescence_time()
        if not hasattr(self, 'n_eqs'):
            self._systems_in_eq()



        plot_all_grids(
            self.m_pbh_values_in_Msun,
            self.v_inf_values_kms,
            self.cross_section_grid,
            self.n_eqs,
            self.coalescence_time,
            self.name,
            self.r,
        )


        


