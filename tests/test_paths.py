import importlib
import os
from pathlib import Path

import captus.paths as paths


def test_repository_path_defaults_to_checkout(monkeypatch):
    monkeypatch.delenv("CAPTUS_PATH", raising=False)
    monkeypatch.delenv("CAPTUS_REPO_ROOT", raising=False)
    module = importlib.reload(paths)

    expected_path = Path(module.__file__).resolve().parents[2]
    assert os.environ["CAPTUS_PATH"] == str(expected_path)
    assert module.get_repository_path() == expected_path
    assert module.get_data_path() == expected_path / "data"
    assert module.get_visuals_path() == expected_path / "visuals"


def test_repository_path_honors_environment_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPTUS_PATH", str(tmp_path))

    assert paths.get_repository_path() == tmp_path.resolve()


def test_legacy_repository_path_remains_a_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("CAPTUS_PATH", raising=False)
    monkeypatch.setenv("CAPTUS_REPO_ROOT", str(tmp_path))
    module = importlib.reload(paths)

    assert module.get_repository_path() == tmp_path.resolve()
    assert Path(os.environ["CAPTUS_PATH"]).resolve() == tmp_path.resolve()


def test_empty_primary_path_uses_legacy_path_after_import(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPTUS_PATH", "")
    monkeypatch.setenv("CAPTUS_REPO_ROOT", str(tmp_path))

    assert paths.get_repository_path() == tmp_path.resolve()
    assert paths.get_data_path() == tmp_path.resolve() / "data"