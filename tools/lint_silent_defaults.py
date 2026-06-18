"""Lint: no silent *scientific* defaults in model-construction code.

A missing or empty model input — PSF, noise model, mask, units, regularization,
priors, pixel grid — must FAIL loudly, never silently default to a
degraded-but-plausible model. A flexible model then fits the data to a fine
chi^2 and the misspecification is invisible. This is the pattern that caused a
run to be fit with no PSF::

    psf = np.load(p) if os.path.exists(p) else None   # <- silent misspecification

We flag risky fallback constructs (``X if ... else None``, ``os.path.exists(...)
else ...``, ``dict.get(...)``, ``getattr(obj, name, default)``) that appear on a
line mentioning a physics-sensitive name, in the model-construction modules
below. A flagged line is allowed only if it carries an explicit annotation::

    psf = ... if ... else None   # physics-default-ok: <reason>

This is a deliberately simple heuristic (high signal, easy to read/extend), not
a substitute for review. Run::

    python tools/lint_silent_defaults.py            # audit, prints findings
    python tools/lint_silent_defaults.py --quiet    # exit code only

Exit code is non-zero if any unannotated violation is found, so it can run in
CI or as a pytest test (see tests/test_no_silent_scientific_defaults.py).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from typing import List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model-construction / asset-loading surface: where a silent physics default is
# most damaging. Globs are relative to the repo root.
TARGET_GLOBS = [
    "src/gigalens_research/simtests/system.py",
    "src/gigalens_research/simtests/pipelines.py",
    "src/gigalens_research/simtests/experiments/*.py",
    "src/gigalens_research/simulations/*.py",
]

# Names whose silent default changes the *physics* of the forward model.
# ``psf``/``kernel`` are matched as substrings (so ``psf_path``, ``psf_srcdir``,
# ``kernel_fn`` etc. are caught — a word boundary would miss them); the rest are
# word-bounded to keep noise down.
SENSITIVE = re.compile(
    r"(?:psf|kernel)"
    r"|\b("
    r"noise|background_rms|exp_time|err_?map|mask|"
    r"regulari[sz]|ridge|prior|beta|n_max|supersample|delta_pix|num_pix|"
    r"pixel_scale|photfnu|units?|deflection|gamma|theta_e|sigma_bkg"
    r")\b",
    re.IGNORECASE,
)

# Fallback constructs that can turn "missing input" into "silent default".
RISKY = [
    ("exists-ternary", re.compile(r"os\.path\.(?:exists|isfile|isdir)\s*\([^)]*\)\s*(?:else|if)")),
    ("else-None", re.compile(r"\belse\s+None\b")),
    ("dict-get-default", re.compile(r"\.get\s*\([^)]*,[^)]*\)")),
    ("getattr-default", re.compile(r"\bgetattr\s*\([^,]+,[^,]+,[^)]+\)")),
]

ALLOW = "physics-default-ok"


BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "silent_defaults_baseline.txt")


@dataclass
class Violation:
    path: str
    lineno: int
    kind: str
    text: str

    @property
    def signature(self) -> str:
        """Stable identity, independent of line number (which drifts as files
        change): the file plus the offending source text."""
        rel = os.path.relpath(self.path, REPO_ROOT)
        return f"{rel} :: {self.text.strip()}"

    def __str__(self) -> str:
        rel = os.path.relpath(self.path, REPO_ROOT)
        return f"{rel}:{self.lineno}: [{self.kind}] {self.text.strip()}"


def load_baseline(path: str = BASELINE_PATH) -> set:
    """Set of accepted (known-debt) violation signatures, if a baseline exists."""
    if not os.path.exists(path):
        return set()
    sigs = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                sigs.add(line)
    return sigs


def write_baseline(violations: List["Violation"], path: str = BASELINE_PATH) -> None:
    header = (
        "# Baseline of accepted silent-default findings (known debt).\n"
        "# Triage each: make the missing input RAISE, or annotate the source line\n"
        "# `# physics-default-ok: <reason>` and remove it here. New violations not\n"
        "# in this list fail the lint (tools/lint_silent_defaults.py).\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for sig in sorted({v.signature for v in violations}):
            f.write(sig + "\n")


def _iter_target_files(root: str = REPO_ROOT) -> List[str]:
    files: List[str] = []
    for pat in TARGET_GLOBS:
        files.extend(glob.glob(os.path.join(root, pat)))
    return sorted(set(files))


def find_violations(root: str = REPO_ROOT) -> List[Violation]:
    out: List[Violation] = []
    for path in _iter_target_files(root):
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                if ALLOW in line:                 # explicitly justified
                    continue
                if not SENSITIVE.search(line):
                    continue
                for kind, rx in RISKY:
                    if rx.search(line):
                        out.append(Violation(path, i, kind, line))
                        break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true", help="exit code only, no listing")
    ap.add_argument("--audit", action="store_true",
                    help="list ALL findings, ignoring the baseline")
    ap.add_argument("--write-baseline", action="store_true",
                    help="record current findings as accepted known-debt and exit 0")
    args = ap.parse_args()

    violations = find_violations()

    if args.write_baseline:
        write_baseline(violations)
        print(f"Wrote {len({v.signature for v in violations})} finding(s) to {BASELINE_PATH}")
        return 0

    baseline = set() if args.audit else load_baseline()
    new = [v for v in violations if v.signature not in baseline]

    if not args.quiet:
        files = _iter_target_files()
        print(f"Scanned {len(files)} model-construction file(s) for silent physics defaults.")
        if args.audit:
            print(f"AUDIT: {len(violations)} total finding(s) (baseline ignored):\n")
            for v in violations:
                print("  " + str(v))
        elif new:
            print(f"\n{len(new)} NEW unannotated risky fallback(s) "
                  f"(not in baseline of {len(baseline)}):\n")
            for v in new:
                print("  " + str(v))
            print(
                "\nEach must either RAISE on the missing input, or be annotated "
                f"`# {ALLOW}: <reason>` if the default is genuinely safe. "
                "(To accept deliberately: re-run with --write-baseline.)"
            )
        else:
            print(f"No new silent physics defaults (baseline: {len(baseline)} known).")
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())
