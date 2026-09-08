from captus.three_body_capture.simulations.threebodyevolution import ThreeBodyEvolution
import captus.three_body_capture.simulations.threebodyevolution as tb_mod


def test_iter_runs_accepts_dict_and_list(dummy_configuration_class, sampled_mc_block):
    run_a = {"name": "run_a", "configuration": dummy_configuration_class(), "seed": 11, "sampled_mc": sampled_mc_block()}
    run_b = {"name": "run_b", "configuration": dummy_configuration_class(), "seed": 22, "sampled_mc": sampled_mc_block()}

    evo_dict = ThreeBodyEvolution({"a": run_a, "b": run_b})
    evo_list = ThreeBodyEvolution([run_a, run_b])

    assert len(evo_dict._iter_runs()) == 2
    assert len(evo_list._iter_runs()) == 2


def test_build_tasks_flattens_runs(monkeypatch, dummy_configuration_class, sampled_mc_block):
    run = {"name": "run_a", "configuration": dummy_configuration_class(), "seed": 123, "sampled_mc": sampled_mc_block(v_inf=1500.0, n=2)}
    evo = ThreeBodyEvolution([run])

    monkeypatch.setattr(tb_mod, "check_result_exists", lambda *args, **kwargs: False)

    all_pars, run_info = evo._build_tasks()

    assert len(all_pars) == 2
    assert len(run_info) == 1
    assert run_info[0]["name"] == "run_a"
    assert run_info[0]["n_sims"] == 2
    assert run_info[0]["skipped"] == 0
    assert all(hasattr(task[1], "entropy") for task in all_pars)
    assert [task[3] for task in all_pars] == [1500.0, 1500.0]


def test_run_simulation_with_fake_pool(monkeypatch, dummy_configuration_class, dummy_pool, sampled_mc_block):
    run = {"name": "run_a", "configuration": dummy_configuration_class(), "seed": 456, "sampled_mc": sampled_mc_block(v_inf=2000.0, n=1)}
    evo = ThreeBodyEvolution([run])

    monkeypatch.setattr(tb_mod, "check_result_exists", lambda *args, **kwargs: False)
    monkeypatch.setattr(tb_mod.multiprocessing, "Pool", dummy_pool)
    monkeypatch.setattr(
        tb_mod,
        "rebound_worker",
        lambda pars: {"i": pars[2], "v_inf": pars[3]},
    )

    result = evo.run_simulation(num_cores=1)

    assert result is None
    assert hasattr(evo, "results") is False or evo.results is None
