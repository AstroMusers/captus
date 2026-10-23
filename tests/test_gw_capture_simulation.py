import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u

from captus.gw_capture.simulation import GWCapture
from captus.gw_capture.calculations import orbital_period as gw_orbital_period
from captus.utils.calculations import orbital_period as shared_orbital_period
import captus.gw_capture.simulation as gw_sim


def _make_config(gridsize=3):
    return {
        "name": "test_capture",
        "R_star": 11e3,
        "M_star": 1.4 * 1.98847e30,
        "gridsize": gridsize,
        "M_pbh_values": np.logspace(-8, -6, gridsize) * 1.98847e30,
        "v_inf_values": np.logspace(3, 4, gridsize),
        "r": 8,
        "bmax_limit": 1e5,
    }


def test_minimum_impact_parameter_includes_pbh_horizon():
    stellar_mass = 1.4 * const.M_sun.value  # [kg]
    pbh_mass = 10.0 * const.M_sun.value  # [kg]
    stellar_radius = 11e3  # [m]
    velocity = 100e3  # [m s^-1]
    collision_radius = stellar_radius + 2 * const.G.value * pbh_mass / const.c.value**2  # [m]
    expected = np.sqrt(
        collision_radius**2
        + 2 * const.G.value * (stellar_mass + pbh_mass) * collision_radius / velocity**2
    )  # [m]
    from captus.gw_capture.calculations import b_minimum

    assert b_minimum(stellar_mass, pbh_mass, stellar_radius, velocity) == pytest.approx(expected)


def test_initialization_builds_grids():
    capture = GWCapture(_make_config(gridsize=4))

    assert capture.name == "test_capture"
    assert capture.M_grid.shape == (4, 4)
    assert capture.V_grid.shape == (4, 4)
    assert capture.v_inf_values_kms[0] > 0
    assert capture.m_pbh_values_in_Msun.shape == (4,)


def test_gw_orbital_period_matches_shared_kepler_utility():
    semimajor_axes_au = np.array([0.2, 1.0, 3.0])
    mass_pbh_kg = 2.0 * const.M_sun.value
    mass_star_kg = 1.4 * const.M_sun.value
    shared_period = shared_orbital_period(
        (semimajor_axes_au * u.au).to_value(u.m),
        mass_pbh_kg,
        mass_star_kg,
    )
    expected_period_years = (shared_period * u.s).to_value(u.yr)

    actual_period_years = gw_orbital_period(
        mass_pbh_kg,
        mass_star_kg,
        semimajor_axes_au,
    )

    np.testing.assert_allclose(actual_period_years, expected_period_years)


def test_capture_cross_section_getter_caches_result(monkeypatch):
    called = {}
    monkeypatch.setattr(
        gw_sim,
        "plot_capture_cross_section",
        lambda *args, **kwargs: called.setdefault("plot_capture_cross_section", True),
    )

    capture = GWCapture(_make_config(gridsize=3))
    grid = capture.get_capture_cross_section(plot=True)

    assert grid.shape == (3, 3)
    assert hasattr(capture, "cross_section_grid")
    assert hasattr(capture, "b_min_values")
    assert hasattr(capture, "b_max_values")
    assert called.get("plot_capture_cross_section") is True
    assert np.all(np.isfinite(grid))


def test_orbital_parameters_and_coalescence_time(monkeypatch):
    monkeypatch.setattr(
        gw_sim,
        "plot_coalescence_time",
        lambda *args, **kwargs: None,
    )

    capture = GWCapture(_make_config(gridsize=3))
    capture.get_capture_cross_section()

    a_grid, e_grid, p_grid = capture.get_orbital_parameters()
    t_grid = capture.get_coalescence_time(plot=True)

    assert a_grid.shape == (3, 3)
    assert e_grid.shape == (3, 3)
    assert p_grid.shape == (3, 3)
    assert t_grid.shape == (3, 3)
    assert np.all(np.isfinite(a_grid) | np.isnan(a_grid))
    assert np.all(np.isfinite(e_grid) | np.isnan(e_grid))
    assert np.all(np.isfinite(p_grid) | np.isnan(p_grid))


def test_systems_in_eq_and_all_plots(monkeypatch):
    monkeypatch.setattr(
        gw_sim,
        "plot_n_eqs",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        gw_sim,
        "plot_all_grids",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        gw_sim,
        "neq_r_f",
        lambda *args, **kwargs: {"Neq_pbh_f_bound": 1.0},
    )

    capture = GWCapture(_make_config(gridsize=3))
    capture.get_capture_cross_section()
    capture.get_coalescence_time()

    n_eqs = capture.get_systems_in_eq(plot=True)
    n_eqs_per_pbh = capture.get_systems_in_eq_per_pbh()

    assert n_eqs.shape == (3, 3)
    assert len(n_eqs_per_pbh) == 3
    assert np.all(np.isfinite(n_eqs) | np.isnan(n_eqs))

    capture.get_all_plots()
