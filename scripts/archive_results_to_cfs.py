#!/usr/bin/env python3
"""Archive result subtrees from the (purge-eligible) results root to durable CFS.

The results root ($PSCRATCH/gigalens on NERSC) is fast and large but is purged
after ~180 days of no access. Anything worth keeping — certified runs, plots for
a paper, reference posteriors — should be copied to the Community File System,
which is backed up and never purged.

Usage
-----
    # Preview (dry-run is the default — nothing is copied):
    python scripts/archive_results_to_cfs.py results/sample_cosmology/dspl_cosmology_newapi

    # Actually copy:
    python scripts/archive_results_to_cfs.py results/sample_cosmology/dspl_cosmology_newapi --execute

Each PATH is interpreted relative to the results root (see
``gigalens_research.paths.results_root``); an absolute path that lives under the
results root is accepted too. The same relative sub-structure is recreated under
the archive root (``gigalens_research.paths.cfs_archive_root``), so
``results/sample_cosmology/foo`` archives to ``<archive_root>/results/sample_cosmology/foo``.

Set the archive destination first, e.g.::

    export GIGALENS_CFS_PROJECT=m5362     # -> $CFS/m5362/$USER/gigalens
    # or, fully explicit:
    export GIGALENS_ARCHIVE_ROOT=/global/cfs/cdirs/m5362/$USER/gigalens

rsync is used with ``-a`` (archive mode: preserves times/perms, recursive), so
re-running only transfers changed files and it is safe to run repeatedly.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from gigalens_research.paths import cfs_archive_root, results_root


def _rel_under_root(path: str, root: str) -> str:
    """Return ``path`` expressed relative to ``root``, or raise if it is outside."""
    abspath = os.path.abspath(os.path.expanduser(path))
    if os.path.isabs(path) or path.startswith("~"):
        rel = os.path.relpath(abspath, root)
        if rel.startswith(os.pardir):
            raise SystemExit(
                f"error: {path!r} is not under the results root {root!r}; "
                f"pass a path inside the results root, or a relative subpath."
            )
        return rel
    return os.path.normpath(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help="Result subtrees to archive (relative to the results root).")
    ap.add_argument("--execute", action="store_true",
                    help="Actually copy. Without this flag the script only previews (rsync --dry-run).")
    args = ap.parse_args(argv)

    src_root = results_root()
    dst_root = cfs_archive_root()
    dry = not args.execute

    print(f"results root: {src_root}")
    print(f"archive root: {dst_root}")
    print(f"mode:         {'DRY-RUN (no copy; pass --execute to copy)' if dry else 'EXECUTE'}\n", flush=True)

    rc = 0
    for p in args.paths:
        rel = _rel_under_root(p, src_root)
        src = os.path.join(src_root, rel)
        dst = os.path.join(dst_root, rel)
        if not os.path.exists(src):
            print(f"!! skip: source does not exist: {src}")
            rc = 1
            continue
        # rsync SRC/ DST/ mirrors the *contents* of SRC into DST.
        os.makedirs(os.path.dirname(dst), exist_ok=True) if not dry else None
        cmd = ["rsync", "-a", "--info=stats2"]
        if dry:
            cmd.append("--dry-run")
        cmd += [src.rstrip("/") + "/", dst.rstrip("/") + "/"]
        print(f"$ {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            rc = result.returncode
        print()
    if dry:
        print("Dry-run only. Re-run with --execute to perform the copy.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
