from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u

from captus.three_body_capture.analysis.analysis import Analysis
from captus.utils import calculations as calc

class FakeNPZ:
    """Minimal np.load(...)-like object used by Analysis (has .files and __getitem__)."""
    def __init__(self, mapping):
        # store arrays so .item()/np.asarray(...) works like real NpzFile
        self._d = {k: (np.array(v) if not isinstance(v, np.ndarray) else v) for k, v in mapping.items()}
        self.files = list(self._d.keys())

    def __getitem__(self, key):
        return self._d[key]

    def keys(self):
        return self.files

def _make_analysis():
    # construct minimal Analysis instance without running __init__
    analysis = Analysis.__new__(Analysis)
    analysis.name = "demo"
    analysis.system_param_dict = {"seed_base": 1, "mA": 1.0, "mB": 2.0, "mC": 3.0, "aB": 4.0}
    analysis.simulation_param_dict = {"sample_size": 1, "importance_sampling": False, "trials": 1, "vDM": 1.0}
    # mc_results can be plain dict-like for this test
    analysis.mc_results = {"V1": {"v_inf": np.array(1000.0)}}
    analysis.sampled_mc_results = {
        "V1": {
            "sample_number": 1,
            "n_captured": 1,
            "idx": np.array([0]),
            "cap_b": np.array([1.0]),
            "cap_bmin": np.array([0.5]),
            "cap_bmax": np.array([1.5]),
            "checks": np.array([1, 0, 1, 0, 0, 0, 0, 1]),
            "epsilon": 0.1,
            "v_inf": np.array(1000.0),
        }
    }

    # Use FakeNPZ to emulate npz files (provides .files and indexing)
    analysis.rebound_results = {"V1": [FakeNPZ({"i": np.array([0])})]}
    analysis.results_dictionary = {"stale": True}
    return analysis

def test_get_combined_dictionary_recomputes_when_cache_disabled(monkeypatch):
    analysis = _make_analysis()

    monkeypatch.setattr(
        analysis,
        "_get_masks",
        lambda rebound_list: {
            "ejection": np.array([0]),
            "collision": np.array([0]),
            "termination": np.array([0]),
            "completed": np.array([1]),
        },
    )
    monkeypatch.setattr(
        analysis,
        "_get_occurrences",
        lambda *args, **kwargs: {
            "n_sampled": 1,
            "n_captured": 1,
            "capture_rate": 2.0,
            "total_systems_neq": 3.0,
            "collided_systems_neq": 0.0,
            "ejected_systems_neq": 0.0,
            "terminated_systems_neq": 0.0,
            "capture_cross_section_total": 1.0,
        },
    )
    monkeypatch.setattr(analysis, "_get_termination_counts", lambda rebound_list: {"completed_count": 1})
    monkeypatch.setattr(
        analysis,
        "_total_occurrences_trapz",
        lambda catalog: {
            "total_systems_neq_gl": 3.0,
            "capture_rate_gl": 2.0,
            "v_bins_used": 1,
            "collided_systems_neq_gl": 0.0,
            "ejected_systems_neq_gl": 0.0,
            "terminated_systems_neq_gl": 0.0,
        },
    )
    monkeypatch.setattr(
        analysis,
        "_total_occurrences_gl",
        lambda catalog: {
            "total_systems_neq_gl": 3.0,
            "capture_rate_gl": 2.0,
            "v_bins_used": 1,
            "collided_systems_neq_gl": 0.0,
            "ejected_systems_neq_gl": 0.0,
            "terminated_systems_neq_gl": 0.0,
        },
    )
    monkeypatch.setattr(
        analysis,
        "_get_Neq_at_r_f",
        lambda catalog, rf=8, ri=None, num_points=100: {"Neq_pbh_f_bound": 1.0, "Neq_pbh_f_full": 2.0},
    )
    monkeypatch.setattr(analysis, "_total_termination_counts", lambda catalog: {"completed_count": 1})
    monkeypatch.setattr(analysis, "_get_errors", lambda catalog: {"ok": True})

    catalog = analysis.get_combined_dictionary(update=True, use_cached_data=False)

    assert "stale" not in catalog
    assert catalog["V1"]["occurrences"]["capture_rate"] == 2.0
    assert catalog["V1"]["sample_number"] == 1
    assert catalog["v_keys"] == ["V1"]
    assert catalog["errors"] == {"ok": True}


@pytest.mark.parametrize("return_all_bounds", [False, True])
@pytest.mark.parametrize("mass_solar", [0.1, 1.0, 10.0, 100.0, 1000.0])
def test_pbh_bounds_entry_points_preserve_results_and_warnings(monkeypatch, capsys, mass_solar, return_all_bounds):
    def load_bound(bound_id):
        if bound_id == "missing":
            raise FileNotFoundError("missing bound")
        scale = 1.0 if bound_id == "first" else 2.0
        return np.array([1.0, 100.0]), scale * np.array([0.01, 1.0])

    monkeypatch.setenv("PBHBOUNDS_PATH", "fixture")
    monkeypatch.setattr(calc, "_load_pbhbounds_tools", lambda path: SimpleNamespace(load_bound=load_bound))
    mass_kg = (mass_solar * u.M_sun).to_value(u.kg)  # [kg]
    bound_ids = ["first", "second", "missing"]
    standalone = calc.get_pbh_fraction_bounds(mass_kg, bound_ids, return_all_bounds)
    assert capsys.readouterr().out == ""
    analysis = Analysis.__new__(Analysis)
    result = analysis.get_pbh_fraction_bounds({"mC": mass_kg}, bound_ids, return_all_bounds)
    assert "Could not load bound 'missing': missing bound" in capsys.readouterr().out
    assert result.keys() == standalone.keys()
    for key in result:
        if key.startswith("status_"):
            assert result[key] == standalone[key]
        else:
            np.testing.assert_allclose(result[key], standalone[key], equal_nan=True)
    assert result["status_missing"] == "error"
    assert np.isnan(result["bound_missing"])
    assert result["mC_Msun"] == (mass_kg * u.kg).to_value(u.M_sun)
    if 1.0 <= result["mC_Msun"] <= 100.0:
        assert result["status_first"] == "valid"
        assert result["bound_first"] == pytest.approx(0.01 * mass_solar)
        assert result["bound_second"] == pytest.approx(0.02 * mass_solar)
        assert ("min_fraction" in result) == (not return_all_bounds)
        if not return_all_bounds:
            assert result["min_fraction"] == pytest.approx(0.01 * mass_solar)
            assert result["max_fraction"] == pytest.approx(0.02 * mass_solar)
            assert result["mean_fraction"] == pytest.approx(0.015 * mass_solar)
    else:
        assert result["status_first"] == "out_of_range"
        assert np.isnan(result["bound_first"])
        assert "min_fraction" not in result
    single = analysis.get_pbh_fraction_bounds({"mC": mass_kg}, "first")
    assert "min_fraction" not in single
    assert "status_second" not in single


def test_pbh_bounds_preserve_missing_input_errors(monkeypatch):
    analysis = Analysis.__new__(Analysis)
    monkeypatch.delenv("PBHBOUNDS_PATH", raising=False)
    with pytest.raises(ValueError, match="PBHbounds path not provided"):
        calc.get_pbh_fraction_bounds(None, "first")
    with pytest.raises(ValueError, match="PBHbounds path not provided"):
        analysis.get_pbh_fraction_bounds({}, "first")
    monkeypatch.setenv("PBHBOUNDS_PATH", "fixture")
    monkeypatch.setattr(calc, "_load_pbhbounds_tools", lambda path: SimpleNamespace())
    with pytest.raises(ValueError, match="Mpbh must be provided"):
        calc.get_pbh_fraction_bounds(None, "first")
    with pytest.raises(ValueError, match="catalog must contain 'mC'"):
        analysis.get_pbh_fraction_bounds({}, "first")


def test_pbh_tools_loader_reads_configured_checkout(tmp_path, monkeypatch, capsys):
    path = tmp_path / "tools.py"
    with pytest.raises(FileNotFoundError, match="PBHbounds tools.py not found"):
        calc._load_pbhbounds_tools(tmp_path)
    print("Writing to %s..." % path)
    path.write_text("def load_bound(bound_id):\n    return bound_id\n")
    tools = calc._load_pbhbounds_tools(str(tmp_path))
    assert tools.load_bound("first") == "first"
    assert "Reading from %s..." % path in capsys.readouterr().out
    monkeypatch.setattr(calc.importlib.util, "spec_from_file_location", lambda *args: None)
    with pytest.raises(ImportError, match="Could not create import spec"):
        calc._load_pbhbounds_tools(tmp_path)


@pytest.mark.parametrize('paired', [False, True])
@pytest.mark.parametrize('number_shells', [1, 3])
def test_shell_sum_uses_fresh_merged_analyses_at_each_radius(monkeypatch, paired, number_shells):
    from captus.utils.misc import get_shell_sum_Neq_and_params

    class ShellAnalysis:
        def __init__(self, amplitude):
            self.amplitude = amplitude
            self.results_dictionary = {'original': amplitude}

        def merged_copy_for_error_analysis(self, other):
            return ShellAnalysis(self.amplitude + other.amplitude)

        def set_bootstrap(self, enabled):
            assert enabled is False

        def update_results_dictionary(self, radius):
            self.results_dictionary = {'total_pbh_neq_n': {
                'Neq_pbh_f_full': self.amplitude * radius,
                'Neq_pbh_f_bound': 0.5 * self.amplitude * radius,
                'number_density_in_range_kpc3': self.amplitude + radius,
            }}

    def star_number(radius, width, M_star, fraction=1.0):
        return 10.0 * radius * width * fraction / M_star

    monkeypatch.setattr(calc, 'star_number_at_r', star_number)
    left = [ShellAnalysis(1.0), ShellAnalysis(2.0)]
    right = [ShellAnalysis(3.0), ShellAnalysis(5.0)]
    amplitudes = np.array([4.0, 7.0])
    analysis = (left, right) if paired else [ShellAnalysis(value) for value in amplitudes]
    edges = np.array([1.0, 2.0, 4.0, 7.0])[:number_shells + 1]  # [kpc]
    original_edges = edges.copy()
    centers = 0.5 * (edges[:-1] + edges[1:])  # [kpc]
    widths = np.diff(edges)  # [kpc]
    mass = 2.0  # [solar mass]
    fraction = 0.25

    all_results, integrated = get_shell_sum_Neq_and_params(
        analysis, edges, M_star=mass, fraction=fraction,
    )

    star_counts = 10.0 * centers * widths / mass
    assert set(all_results) == set(centers)
    assert set(integrated) == {0, 1}
    for radius, count in zip(centers, star_counts):
        shell = all_results[radius]
        np.testing.assert_array_equal(shell['N_eq_full'], amplitudes * radius)
        np.testing.assert_array_equal(shell['N_eq_pbh_bound'], 0.5 * amplitudes * radius)
        np.testing.assert_array_equal(shell['n_pbh_density'], amplitudes + radius)
        np.testing.assert_array_equal(shell['N_eq_full_stars'], amplitudes * radius * count)
    for index, amplitude in enumerate(amplitudes):
        expected = np.sum(amplitude * centers * star_counts)
        assert integrated[index]['N_eq_full_stars_integrated'] == expected
        assert integrated[index]['Neq_pbh_bound_stars_integrated'] == 0.5 * expected
        assert integrated[index]['N_eq_full_alike_integrated'] == fraction * expected
        assert integrated[index]['Neq_pbh_bound_alike_integrated'] == 0.5 * fraction * expected
        assert integrated[index]['N_stars_total_integrated'] == np.sum(star_counts)
    if paired:
        for original in left + right:
            assert original.results_dictionary == {'original': original.amplitude}
    np.testing.assert_array_equal(edges, original_edges)