from captus.three_body_capture.configurations.configuration import Configuration
from pathlib import Path

import pytest


def test_configuration_import():
    assert Configuration is not None


@pytest.mark.parametrize(
    ("name", "grid_size"),
    [
        ("Sun_Jupiter", 10),
        ("CygnusX1", 20),
        ("Intermediate", 20),
        ("OmegaCenturi", 10),
        ("BetaPictoris", 10),
    ],
)
def test_loads_packaged_system_configuration(name, grid_size):
    configuration = Configuration(name=name, seed=3)

    assert configuration.system_param_dict["name"] == name
    assert len(configuration.simulation_param_dict["v_inf_grid"]) == grid_size


def test_sun_pbh_variants_select_their_system_parameters():
    light_pbh = Configuration(name="Sun_Jupiter_lightPBH", seed=3)
    massive_pbh = Configuration(name="Sun_Jupiter_massivePBH", seed=3)

    assert light_pbh.system_param_dict["mC"] < massive_pbh.system_param_dict["mC"]


def test_unknown_system_requires_a_configuration_file():
    with pytest.raises(ValueError, match="supply config_file"):
        Configuration(name="UnknownSystem", seed=3)


def test_loads_custom_configuration_file():
    config_file = (
        Path(__file__).parents[1]
        / "src/captus/three_body_capture/configurations/SunJupiter.yaml"
    )
    configuration = Configuration(name="CustomSystem", seed=3, config_file=config_file)

    assert configuration.system_param_dict["name"] == "CustomSystem"
    assert len(configuration.simulation_param_dict["v_inf_grid"]) == 10


def test_save_directories_create_defaults_and_overrides(monkeypatch, tmp_path):
    repository_root = tmp_path / "captus"
    repository_root.mkdir()
    monkeypatch.chdir(repository_root)
    configuration = Configuration.__new__(Configuration)
    configuration.name = "test_run"

    configuration.set_save_dir(mc_dir="custom/mc", plots_dir="custom/plots")

    expected_paths = {
        "save_dir_mc": tmp_path / "custom/mc",
        "save_dir_rebound": tmp_path / "runs/test_run/Rebound_Simulation_Results",
        "save_dir_plots": tmp_path / "custom/plots",
    }
    for attribute, expected_path in expected_paths.items():
        actual_path = Path(getattr(configuration, attribute)).resolve()
        assert actual_path == expected_path.resolve()
        assert actual_path.is_dir()