"""Central resolution of *where* results are written and archived.

Historically every script hard-coded its output directory, almost always under
``$HOME/GIGALens-Code/...``. On NERSC that fills the 40 GiB home quota fast (and
backs the data up needlessly). This module provides a single, env-overridable
knob so sampling output lands on fast, roomy scratch by default, while still
allowing an explicit override (a laptop, or a shared project directory).

Resolution order for the *results root* (:func:`results_root`):

1. ``$GIGALENS_RESULTS_ROOT`` if set — the explicit override; wins over all else.
2. ``$PSCRATCH/gigalens`` (or ``$SCRATCH/gigalens``) if a scratch space is
   defined — the NERSC default. Fast and ~20 TiB, but **purged** after ~180 days
   of no access, so anything worth keeping must be archived to durable storage
   (see :func:`cfs_archive_root`).
3. ``~/GIGALens-Code`` as a last resort (laptop / no scratch).

Nothing here creates directories or touches the filesystem; callers do that.
The functions are intentionally tiny and dependency-free (only ``os``) so they
can be imported from anywhere without risking an import cycle.
"""
from __future__ import annotations

import os
from typing import Optional


def results_root() -> str:
    """Absolute base directory under which results should be written.

    See the module docstring for the resolution order.
    """
    override = os.environ.get("GIGALENS_RESULTS_ROOT")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    scratch = os.environ.get("PSCRATCH") or os.environ.get("SCRATCH")
    if scratch:
        return os.path.join(scratch, "gigalens")
    return os.path.expanduser("~/GIGALens-Code")


def resolve_out_dir(out_dir: Optional[str]) -> Optional[str]:
    """Resolve a stage/campaign output directory against :func:`results_root`.

    - ``None`` -> ``None`` (disk I/O disabled; unchanged).
    - an absolute path (after ``~`` expansion) -> returned unchanged, so any
      existing caller that already builds an absolute path keeps its exact
      behavior.
    - a *relative* path -> joined onto :func:`results_root`.

    Keeping absolute paths untouched means no current caller changes behavior
    (they all pass absolute paths today), while new code can simply pass a
    relative path like ``resolve_out_dir("results/my_run")`` and land in the
    right place regardless of the machine.
    """
    if out_dir is None:
        return None
    expanded = os.path.expanduser(str(out_dir))
    if os.path.isabs(expanded):
        return expanded
    return os.path.join(results_root(), expanded)


def cfs_archive_root() -> str:
    """Absolute base directory for durable (non-purged, backed-up) archival.

    Resolution order:

    1. ``$GIGALENS_ARCHIVE_ROOT`` if set — the explicit override.
    2. ``$CFS/<project>/<user>/gigalens`` when both ``$CFS`` and
       ``$GIGALENS_CFS_PROJECT`` are set (NERSC Community File System — backed
       up, not purged). ``$USER`` is appended so multiple people sharing a
       project directory do not collide.
    3. ``~/gigalens_archive`` as a last resort.

    Deliberately not hard-coding a project name: set ``GIGALENS_CFS_PROJECT``
    (e.g. ``m5362``) or ``GIGALENS_ARCHIVE_ROOT`` in your environment.
    """
    override = os.environ.get("GIGALENS_ARCHIVE_ROOT")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    cfs = os.environ.get("CFS")
    project = os.environ.get("GIGALENS_CFS_PROJECT")
    if cfs and project:
        return os.path.join(cfs, project, os.environ.get("USER", ""), "gigalens")
    return os.path.expanduser("~/gigalens_archive")


if __name__ == "__main__":  # pragma: no cover - human debugging aid
    print("results_root():    ", results_root())
    print("cfs_archive_root():", cfs_archive_root())
