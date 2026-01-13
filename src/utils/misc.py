import os


def _resolve_repo_root() -> str:
    # Optional override for admins/users
    env_root = os.getenv("CAPTUS_REPO_ROOT")
    if env_root:
        return os.path.abspath(env_root)
    else:
        REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        return REPO_ROOT