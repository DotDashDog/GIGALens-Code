"""Unit tests for gigalens_research.paths result/archive root resolution."""
import os

import pytest

from gigalens_research import paths

_ENV_VARS = [
    "GIGALENS_RESULTS_ROOT",
    "PSCRATCH",
    "SCRATCH",
    "GIGALENS_ARCHIVE_ROOT",
    "CFS",
    "GIGALENS_CFS_PROJECT",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start every test from a known-empty environment for the path vars."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("USER", "tester")
    yield


# -- results_root ----------------------------------------------------------

def test_results_root_explicit_override_wins(monkeypatch):
    monkeypatch.setenv("PSCRATCH", "/pscratch/sd/x/xx")
    monkeypatch.setenv("GIGALENS_RESULTS_ROOT", "~/somewhere/custom")
    assert paths.results_root() == os.path.expanduser("~/somewhere/custom")


def test_results_root_prefers_pscratch(monkeypatch):
    monkeypatch.setenv("PSCRATCH", "/pscratch/sd/x/xx")
    monkeypatch.setenv("SCRATCH", "/some/other/scratch")
    assert paths.results_root() == "/pscratch/sd/x/xx/gigalens"


def test_results_root_falls_back_to_scratch(monkeypatch):
    monkeypatch.setenv("SCRATCH", "/some/scratch")
    assert paths.results_root() == "/some/scratch/gigalens"


def test_results_root_last_resort_home(monkeypatch):
    assert paths.results_root() == os.path.expanduser("~/GIGALens-Code")


# -- resolve_out_dir -------------------------------------------------------

def test_resolve_out_dir_none_passthrough():
    assert paths.resolve_out_dir(None) is None


def test_resolve_out_dir_absolute_unchanged(monkeypatch):
    monkeypatch.setenv("GIGALENS_RESULTS_ROOT", "/results/root")
    assert paths.resolve_out_dir("/abs/path/run7") == "/abs/path/run7"


def test_resolve_out_dir_relative_joined(monkeypatch):
    monkeypatch.setenv("GIGALENS_RESULTS_ROOT", "/results/root")
    assert paths.resolve_out_dir("results/run7") == "/results/root/results/run7"


def test_resolve_out_dir_expands_user_to_absolute(monkeypatch):
    monkeypatch.setenv("GIGALENS_RESULTS_ROOT", "/results/root")
    # ``~`` expands to an absolute path, so it is returned verbatim (not joined).
    assert paths.resolve_out_dir("~/x/run7") == os.path.expanduser("~/x/run7")


# -- cfs_archive_root ------------------------------------------------------

def test_cfs_archive_root_explicit_override_wins(monkeypatch):
    monkeypatch.setenv("CFS", "/global/cfs/cdirs")
    monkeypatch.setenv("GIGALENS_CFS_PROJECT", "m5362")
    monkeypatch.setenv("GIGALENS_ARCHIVE_ROOT", "/explicit/archive")
    assert paths.cfs_archive_root() == "/explicit/archive"


def test_cfs_archive_root_from_project(monkeypatch):
    monkeypatch.setenv("CFS", "/global/cfs/cdirs")
    monkeypatch.setenv("GIGALENS_CFS_PROJECT", "m5362")
    assert paths.cfs_archive_root() == "/global/cfs/cdirs/m5362/tester/gigalens"


def test_cfs_archive_root_last_resort(monkeypatch):
    assert paths.cfs_archive_root() == os.path.expanduser("~/gigalens_archive")
