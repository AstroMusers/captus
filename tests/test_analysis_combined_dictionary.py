import numpy as np

from captus.three_body_capture.analysis.analysis import Analysis


def _make_analysis():
    analysis = Analysis.__new__(Analysis)
    analysis.name = "demo"
    analysis.system_param_dict = {"seed_base": 1, "mA": 1.0, "mB": 2.0, "mC": 3.0, "aB": 4.0}
    analysis.simulation_param_dict = {"sample_size": 1, "importance_sampling": False, "trials": 1, "vDM": 1.0}
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
        }
    }
    analysis.rebound_results = {"V1": [object()]}
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
