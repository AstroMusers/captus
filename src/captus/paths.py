import os
from pathlib import Path

from tdpy.paths import RepositoryPaths


PATH_ENV_VAR = "CAPTUS_PATH"
LEGACY_PATH_ENV_VAR = "CAPTUS_REPO_ROOT"
DEFAULT_REPOSITORY_PATH = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    PATH_ENV_VAR,
    os.environ.get(LEGACY_PATH_ENV_VAR, str(DEFAULT_REPOSITORY_PATH)),
)

_REPOSITORY_PATHS = RepositoryPaths(PATH_ENV_VAR, DEFAULT_REPOSITORY_PATH)


def get_repository_path() -> Path:
    if not os.environ.get(PATH_ENV_VAR) and os.environ.get(LEGACY_PATH_ENV_VAR):
        os.environ[PATH_ENV_VAR] = os.environ[LEGACY_PATH_ENV_VAR]
    return _REPOSITORY_PATHS.get_repository_path()


def get_data_path() -> Path:
    return get_repository_path() / "data"


def get_visuals_path() -> Path:
    return get_repository_path() / "visuals"