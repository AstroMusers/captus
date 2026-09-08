from captus.three_body_capture.simulations.threebodycapture import ThreeBodyCapture
import captus.three_body_capture.simulations.threebodycapture as tb_mod


def test_iter_runs_accepts_dict_and_list(dummy_configuration_class, sampled_mc_block):
    config_a = dummy_configuration_class(seed_base=111, v_inf_grid=(1.0, 2.0))
    config_b = dummy_configuration_class(seed_base=222, v_inf_grid=(3.0,))

    capture_dict = ThreeBodyCapture(
        {
            "run_a": {"configuration": config_a, "seed": 111, "v_inf_grid": [1.0, 2.0]},
            "run_b": {"configuration": config_b, "seed": 222, "v_inf_grid": [3.0]},
        }
    )
    capture_list = ThreeBodyCapture([config_a, config_b])

    assert len(capture_dict._iter_runs()) == 2
    assert len(capture_list._iter_runs()) == 2


def test_build_tasks_expands_v_inf_grid(dummy_configuration_class):
    config = dummy_configuration_class(seed_base=4321, v_inf_grid=(5.0, 6.0, 7.0))
    capture = ThreeBodyCapture([config])

    tasks = capture._build_tasks()

    assert len(tasks) == 3
    assert [task[2] for task in tasks] == [5.0, 6.0, 7.0]
    assert all(hasattr(task[1], "entropy") for task in tasks)
    assert all(task[0] is config for task in tasks)


def test_run_simulation_with_fake_pool(monkeypatch, dummy_configuration_class, dummy_pool):
    capture = ThreeBodyCapture([dummy_configuration_class(seed_base=999, v_inf_grid=(11.0, 22.0))])

    monkeypatch.setattr(tb_mod.multiprocessing, "cpu_count", lambda: 2)
    monkeypatch.setattr(tb_mod.multiprocessing, "Pool", dummy_pool)
    monkeypatch.setattr(
        tb_mod,
        "_mc_worker",
        lambda pars: {"v_inf": pars[2], "seed_state": pars[1].entropy},
    )

    result = capture.run_simulation(num_cores=1)

    assert result is None
