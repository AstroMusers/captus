import os
from pathlib import Path


PATH_ENV_VAR = "CAPTUS_PATH"
LEGACY_PATH_ENV_VAR = "CAPTUS_REPO_ROOT"
DEFAULT_REPOSITORY_PATH = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    PATH_ENV_VAR,
    os.environ.get(LEGACY_PATH_ENV_VAR, str(DEFAULT_REPOSITORY_PATH)),
)


def get_repository_path() -> Path:
    path_value = os.environ.get(PATH_ENV_VAR) or os.environ.get(LEGACY_PATH_ENV_VAR)
    if not path_value or not path_value.strip():
        path_value = str(DEFAULT_REPOSITORY_PATH)
        os.environ[PATH_ENV_VAR] = path_value
    return Path(path_value).expanduser().resolve()


def get_data_path() -> Path:
    return get_repository_path() / "data"


def get_visuals_path() -> Path:
    return get_repository_path() / "visuals"