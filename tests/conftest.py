import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    """Tests read profile.example.yaml etc. by relative path; anchor cwd."""
    monkeypatch.chdir(REPO_ROOT)
