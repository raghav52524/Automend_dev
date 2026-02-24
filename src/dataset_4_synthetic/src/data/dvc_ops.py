"""DVC CLI wrappers using subprocess.run."""
import subprocess


def dvc_pull() -> subprocess.CompletedProcess:
    """Run dvc pull."""
    return subprocess.run(["dvc", "pull"], check=True)


def dvc_add(path: str) -> subprocess.CompletedProcess:
    """Run dvc add on the given path."""
    return subprocess.run(["dvc", "add", path], check=True)


def dvc_push() -> subprocess.CompletedProcess:
    """Run dvc push."""
    return subprocess.run(["dvc", "push"], check=True)
