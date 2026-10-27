import numpy as np
import pytest

from captus.utils.calculations import (
    U,
    baryonic_mass_and_velocity_milkyway,
    collision_cross_section,
    dict_to_sorted_arrays,
    integrate_gauss_legendre,
    integrated_rate_and_error,
    potential_energy,
    rate_and_error,
    star_number_galactic_bulge,
    star_number_at_disc,
    star_number_at_ring,
    weighted_quantile,
)


def test_collision_cross_section_reuses_circle_area():
    radii = np.array([2.0, 3.0])
    expected = np.pi * radii**2 * (1.0 + (4.0 / 5.0)**2)

    np.testing.assert_array_equal(
        collision_cross_section(radii, 4.0, 5.0), expected
    )


def test_stellar_count_helpers_share_reference_density_without_changing_estimates():
    stellar_mass = 2.0  # [M_sun]
    fraction = 0.5
    reference_stars = 0.025 * (4.0 / 3.0) * np.pi * 100.0**3 / stellar_mass
    ring_radius = 8.0  # [kpc]
    inner_radius = 0.1  # [kpc]
    outer_radius = 0.2  # [kpc]
    expected_ring = reference_stars * (2.0 * np.pi * ring_radius * 1000.0 / 200.0) * fraction
    expected_disc = reference_stars * ((outer_radius**2 - inner_radius**2) / 0.1**2) * fraction

    assert star_number_at_ring(ring_radius, stellar_mass, fraction) == pytest.approx(expected_ring)
    assert star_number_at_disc(inner_radius, outer_radius, stellar_mass, fraction) == pytest.approx(expected_disc)


def test_galactic_bulge_count_matches_shared_hernquist_mass_parameters():
    radius = 3.0  # [kpc]
    stellar_mass = 2.0  # [M_sun]
    expected_mass = 1.55e10 * (radius / (radius + 0.70))**2  # [M_sun]
    _, _, components = baryonic_mass_and_velocity_milkyway(radius, return_components=True)

    assert components[1] == pytest.approx(expected_mass, rel=1e-5)
    assert star_number_galactic_bulge(radius, stellar_mass) == pytest.approx(
        expected_mass / stellar_mass, rel=1e-12
    )


@pytest.mark.parametrize('case', ['mixed', 'nonpositive', 'empty'])
def test_dict_to_sorted_arrays_uses_requested_keys_and_preserves_input(case):
    from copy import deepcopy

    records = {
        'late': {'coordinate': 3.0, 'density': 2.0, 'v_inf_au_yr': -10.0},
        'early': {'coordinate': 1.0, 'density': 5.0, 'v_inf_au_yr': 20.0},
        'same': {'coordinate': 1.0, 'density': 3.0, 'v_inf_au_yr': 10.0},
        'zero': {'coordinate': 2.0, 'density': 0.0, 'v_inf_au_yr': 30.0},
        'negative': {'coordinate': 4.0, 'density': -1.0, 'v_inf_au_yr': 40.0},
    }
    if case == 'nonpositive':
        records = {name: records[name] for name in ('zero', 'negative')}
    elif case == 'empty':
        records = {}
    original = deepcopy(records)

    coordinates, densities = dict_to_sorted_arrays(records, 'coordinate', 'density')

    expected_coordinates = [1.0, 1.0, 3.0] if case == 'mixed' else []
    expected_densities = [5.0, 3.0, 2.0] if case == 'mixed' else []
    np.testing.assert_array_equal(coordinates, expected_coordinates)
    np.testing.assert_array_equal(densities, expected_densities)
    assert coordinates.shape == densities.shape
    assert coordinates.ndim == densities.ndim == 1
    assert records == original


def test_weighted_quantile_omits_nonfinite_values_and_nonpositive_weights():
    values = np.array([1.0, 10.0, 3.0, np.nan, 9.0])
    weights = np.array([1.0, 0.0, 3.0, 7.0, np.inf])
    quantiles = np.array([-1.0, 0.0, 0.25, 0.5, 1.0, 2.0])

    result = weighted_quantile(values, quantiles, weights=weights)

    np.testing.assert_allclose(
        result, [1.0, 1.0, 1.0, 1.6666666666666665, 3.0, 3.0]
    )
    np.testing.assert_array_equal(values, [1.0, 10.0, 3.0, np.nan, 9.0])
    np.testing.assert_array_equal(weights, [1.0, 0.0, 3.0, 7.0, np.inf])


def test_weighted_quantile_without_weights_matches_numpy_for_multidimensional_values():
    values = np.array([[1.0, 3.0], [5.0, 7.0]])
    quantiles = np.array([0.25, 0.75])

    np.testing.assert_array_equal(
        weighted_quantile(values, quantiles), np.quantile(values, quantiles)
    )


def test_integrate_gauss_legendre_reuses_shared_rule_and_preserves_filters():
    coordinates = np.array([3.0, 0.0, 2.0, 1.0, np.nan])
    values = np.array([7.0, 0.0, 5.0, 3.0, 99.0])

    assert integrate_gauss_legendre(
        coordinates, values, n=6, exclude_zeros=True
    ) == pytest.approx(10.0)
    assert integrate_gauss_legendre([1.0, np.nan], [3.0, 4.0]) == 0.0


def test_integrated_rate_uses_shared_confidence_multipliers():
    result = integrated_rate_and_error([0.0, 1.0], [1.0, 1.0], [0.1, 0.2], n_gl=4)
    total_rate, total_error, ci68, ci95, _, _ = result

    assert total_rate == pytest.approx(1.0)
    assert ci68 == pytest.approx((total_rate - total_error, total_rate + total_error))
    assert ci95 == pytest.approx((total_rate - 1.96 * total_error, total_rate + 1.96 * total_error))


def test_legacy_potential_energy_name_preserves_argument_order():
    first_gravitational_parameter = np.array([4.0, 9.0])
    second_gravitational_parameter = np.array([2.0, 5.0])
    first_separation = np.array([2.0, 3.0])
    second_separation = np.array([4.0, 10.0])

    np.testing.assert_array_equal(
        U(
            first_gravitational_parameter,
            second_gravitational_parameter,
            first_separation,
            second_separation,
        ),
        potential_energy(
            first_gravitational_parameter,
            first_separation,
            second_gravitational_parameter,
            second_separation,
        ),
    )


@pytest.mark.parametrize(
    ('confidence', 'expected_bounds'),
    [(0.95, [1.804, 2.196]), (0.68, [1.9, 2.1])],
)
def test_rate_and_error_uses_scalar_confidence(confidence, expected_bounds):
    arguments = ([2.0], [4.0], [0.2])
    result = rate_and_error(*arguments) if confidence == 0.95 else rate_and_error(*arguments, conf=confidence)

    mean_rate, rate_error, interval = result
    np.testing.assert_allclose(mean_rate, [2.0])
    np.testing.assert_allclose(rate_error, [0.1])
    np.testing.assert_allclose([interval[0][0], interval[1][0]], expected_bounds)