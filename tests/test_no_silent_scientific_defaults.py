"""Gate: no NEW silent scientific defaults in model-construction code.

Wraps tools/lint_silent_defaults.py. Fails if a risky fallback (silently
defaulting a physics input like PSF/noise/grid/n_max) is introduced that is not
in tools/silent_defaults_baseline.txt. Burn the baseline down over time by
fixing each entry (raise) or annotating the source line `# physics-default-ok:`.

This is stdlib-only and does not import gigalens/jax, so it runs anywhere.
"""
import os
import sys

_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
sys.path.insert(0, _TOOLS)

import lint_silent_defaults as lint  # noqa: E402


def test_no_new_silent_scientific_defaults():
    baseline = lint.load_baseline()
    new = [v for v in lint.find_violations() if v.signature not in baseline]
    assert not new, (
        "New silent scientific default(s) introduced:\n  "
        + "\n  ".join(str(v) for v in new)
        + "\n\nMake the missing input RAISE, or annotate the line "
        "`# physics-default-ok: <reason>`. To deliberately accept, run "
        "`python tools/lint_silent_defaults.py --write-baseline`."
    )
