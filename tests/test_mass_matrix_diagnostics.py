"""The inverse-mass-matrix history in MCLMC/MAMS debug diagnostics.

gigalens no longer records the metric once per chain per step. It records a COMPACT
history — the starting metric plus the one installed at each adaptation-window
boundary, ``(n_windows + 1, dim, dim)`` — alongside ``inverse_mass_matrix_steps``, the
step index at which each row became active. The metric only ever changed at those
boundaries, so the per-step array was one matrix repeated thousands of times.

Two layouts therefore have to be read: the compact one, and the legacy
``(n_chains, n_steps, dim, dim)`` still sitting in runs saved on disk. These tests pin
both, on both samplers, and pin the property that makes the capture side dangerous —
``hist.inverse_mass_matrix[:1]`` is correct for the dense layout and silently
destructive for the compact one, since it returns a well-shaped array either way.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib

matplotlib.use("Agg")

import types

import numpy as np
import pytest
from matplotlib.figure import Figure

from gigalens_research.inference_utils.pipeline import (
    StageDiagnostics,
    _mass_matrix_diagnostics,
)
from gigalens_research.plotting.diagnostics import (
    _mass_matrix_history,
    plot_stage_diagnostics,
)

DIM = 4
N_CHAINS = 3
N_STEPS = 40
N_WINDOWS = 5


def _spd(scale: float, dim: int = DIM) -> np.ndarray:
    """A symmetric positive-definite matrix whose eigenvalues scale with ``scale``."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(dim, dim))
    return scale * (a @ a.T + dim * np.eye(dim))


def _compact_history():
    """``(n_windows + 1, dim, dim)`` with a distinct, shrinking scale per window."""
    scales = np.geomspace(1.0, 1e-3, N_WINDOWS + 1)
    mats = np.stack([_spd(s) for s in scales])
    # Window boundaries: row 0 active from step 0, then one per expanding window.
    steps = np.array([0, 4, 8, 16, 24, 32], dtype=np.int32)
    return mats, steps


def _dense_history():
    """Legacy ``(1, n_steps, dim, dim)`` — chain 0 of a per-step-per-chain array."""
    mats = np.stack([_spd(s) for s in np.geomspace(1.0, 1e-3, N_STEPS)])
    return mats[None, ...]


def _base_arrays():
    """The per-step traces every plotter shares."""
    rng = np.random.default_rng(1)
    return {
        "step_size": np.abs(rng.normal(size=(N_CHAINS, N_STEPS))) + 0.1,
        "L": np.abs(rng.normal(size=(N_CHAINS, N_STEPS))) + 1.0,
        "nonan": np.ones((N_CHAINS, N_STEPS)),
        "samples_cov": _spd(1e-3),
        "samples_z": rng.normal(size=(N_CHAINS, N_STEPS, DIM)),
    }


def _mclmc_arrays(compact: bool):
    arr = _base_arrays()
    arr["xi"] = np.abs(np.random.default_rng(2).normal(size=(N_CHAINS, N_STEPS))) + 0.1
    if compact:
        mats, steps = _compact_history()
        arr["inverse_mass_matrix"] = mats
        arr["inverse_mass_matrix_steps"] = steps
    else:
        arr["inverse_mass_matrix"] = _dense_history()
    return arr


def _mams_arrays(compact: bool):
    arr = _mclmc_arrays(compact)
    arr.pop("xi")
    rng = np.random.default_rng(3)
    arr["acceptance_rate"] = rng.uniform(0.5, 0.9, size=(N_CHAINS, N_STEPS))
    arr["num_integration_steps"] = rng.integers(1, 20, size=(N_CHAINS, N_STEPS))
    return arr


def _diag(stage_class: str, arr):
    return StageDiagnostics(
        stage_name="s", stage_class=stage_class, arrays=arr,
        config={"num_burnin_steps": N_STEPS, "frac_tune1": 0.2,
                "frac_tune2": 0.6, "frac_tune3": 0.2,
                "target_acceptance": 0.8},
        ctx=None,
    )


# --- reading the history ---------------------------------------------------------


def test_compact_history_is_read_whole():
    mats, steps = _compact_history()
    got_mats, got_steps = _mass_matrix_history(
        {"inverse_mass_matrix": mats, "inverse_mass_matrix_steps": steps})
    assert got_mats.shape == (N_WINDOWS + 1, DIM, DIM)
    np.testing.assert_array_equal(got_mats, mats)
    np.testing.assert_array_equal(got_steps, steps)


def test_dense_history_drops_the_chain_axis():
    dense = _dense_history()
    mats, steps = _mass_matrix_history({"inverse_mass_matrix": dense})
    assert mats.shape == (N_STEPS, DIM, DIM)
    np.testing.assert_array_equal(mats, dense[0])
    # No recorded steps for the legacy layout: one row per step, so the row index
    # IS the step index.
    np.testing.assert_array_equal(steps, np.arange(N_STEPS))


def test_final_metric_is_row_minus_one_under_both_layouts():
    """What the surrogate corner depends on: [-1] is the metric carried into
    sampling, whichever layout is in hand."""
    mats, steps = _compact_history()
    compact, _ = _mass_matrix_history(
        {"inverse_mass_matrix": mats, "inverse_mass_matrix_steps": steps})
    dense_raw = _dense_history()
    dense, _ = _mass_matrix_history({"inverse_mass_matrix": dense_raw})
    np.testing.assert_array_equal(compact[-1], mats[-1])
    np.testing.assert_array_equal(dense[-1], dense_raw[0, -1])
    assert compact[-1].shape == dense[-1].shape == (DIM, DIM)


def test_compact_history_without_steps_reports_only_the_final_metric():
    """Rows cannot be placed on the step axis without their step indices, so the
    plotter is told to show the final metric alone rather than invent positions."""
    mats, _ = _compact_history()
    got, steps = _mass_matrix_history({"inverse_mass_matrix": mats})
    assert steps is None
    assert got.shape == (1, DIM, DIM)
    np.testing.assert_array_equal(got[0], mats[-1])


def test_absent_history_is_not_an_error():
    assert _mass_matrix_history({}) is None


def test_unexpected_rank_is_rejected_loudly():
    with pytest.raises(ValueError, match="rank 2"):
        _mass_matrix_history({"inverse_mass_matrix": np.eye(DIM)})


# --- capture ---------------------------------------------------------------------


def test_capture_keeps_the_whole_compact_history():
    """The regression guard for the silent one.

    ``hist.inverse_mass_matrix[:1]`` was right for the dense layout and is quietly
    destructive for the compact one: it returns a valid (1, dim, dim) array holding
    only the STARTING metric, so the adaptation history vanishes with nothing raising
    and the plot still drawing.
    """
    mats, steps = _compact_history()
    hist = types.SimpleNamespace(
        inverse_mass_matrix=mats, inverse_mass_matrix_steps=steps)
    out = _mass_matrix_diagnostics(hist)
    assert out["inverse_mass_matrix"].shape == (N_WINDOWS + 1, DIM, DIM)
    np.testing.assert_array_equal(out["inverse_mass_matrix"], mats)
    np.testing.assert_array_equal(out["inverse_mass_matrix_steps"], steps)
    # The distinguishing check: the FINAL metric must survive, not just the first.
    np.testing.assert_array_equal(out["inverse_mass_matrix"][-1], mats[-1])
    assert not np.allclose(out["inverse_mass_matrix"][0], mats[-1]), \
        "fixture too weak: first and last metrics must differ for this to bite"


def test_capture_drops_replicated_chains_from_a_dense_history():
    dense = np.broadcast_to(_dense_history(), (N_CHAINS, N_STEPS, DIM, DIM))
    hist = types.SimpleNamespace(inverse_mass_matrix=dense)
    out = _mass_matrix_diagnostics(hist)
    assert out["inverse_mass_matrix"].shape == (1, N_STEPS, DIM, DIM)
    assert "inverse_mass_matrix_steps" not in out


# --- plotting --------------------------------------------------------------------


@pytest.mark.parametrize("stage_class,builder", [
    ("MCLMCStage", _mclmc_arrays),
    ("MAMSStage", _mams_arrays),
])
@pytest.mark.parametrize("compact", [True, False], ids=["compact", "legacy_dense"])
def test_plotter_renders_under_both_layouts(stage_class, builder, compact):
    fig = plot_stage_diagnostics(_diag(stage_class, builder(compact)), chain=1)
    assert isinstance(fig, Figure)


@pytest.mark.parametrize("stage_class,builder", [
    ("MCLMCStage", _mclmc_arrays),
    ("MAMSStage", _mams_arrays),
])
def test_mass_matrix_panel_spans_the_whole_run(stage_class, builder):
    """The final window stays active through sampling, so its segment must reach the
    end of the run. Stopping at the last update would read as the preconditioner
    ending there."""
    fig = plot_stage_diagnostics(_diag(stage_class, builder(True)))
    ax = fig.axes[2]                                   # panel 3 in both layouts
    xs = np.concatenate([ln.get_xdata() for ln in ax.lines if len(ln.get_xdata())])
    assert xs.max() >= N_STEPS - 1


@pytest.mark.parametrize("stage_class,builder", [
    ("MCLMCStage", _mclmc_arrays),
    ("MAMSStage", _mams_arrays),
])
def test_mass_matrix_panel_tracks_the_recorded_eigenvalues(stage_class, builder):
    """The plotted levels are the history's own eigenvalues, not a resampling of
    them: a wrong layout read would still draw, so the values are what pins it."""
    arr = builder(True)
    fig = plot_stage_diagnostics(_diag(stage_class, arr))
    ax = fig.axes[2]
    eig = np.linalg.eigvalsh(arr["inverse_mass_matrix"])
    plotted = np.concatenate([ln.get_ydata() for ln in ax.lines if len(ln.get_ydata())])
    for expected in (eig.min(axis=1)[0], eig.max(axis=1)[0], eig.min(axis=1)[-1]):
        assert np.any(np.isclose(plotted, expected)), \
            f"{expected} missing from the panel — layout misread?"


def test_chain_kwarg_is_honoured():
    """`diagnostics("mclmc", chain=3)` — the reported call — with more chains than 3."""
    arr = _mclmc_arrays(True)
    n = 6
    rng = np.random.default_rng(4)
    for k in ("step_size", "L", "nonan", "xi"):
        arr[k] = np.abs(rng.normal(size=(n, N_STEPS))) + 0.1
    arr["samples_z"] = rng.normal(size=(n, N_STEPS, DIM))
    fig = plot_stage_diagnostics(_diag("MCLMCStage", arr), chain=3)
    ax_xi = fig.axes[3]
    assert "chain 3" in ax_xi.get_ylabel()


# --- surrogate corner ------------------------------------------------------------


@pytest.mark.parametrize("compact", [True, False], ids=["compact", "legacy_dense"])
@pytest.mark.parametrize("stage_class", ["MCLMCStage", "MAMSStage"])
def test_surrogate_corner_renders_under_both_layouts(stage_class, compact):
    pytest.importorskip("corner")
    from gigalens_research.plotting.diagnostics import plot_mclmc_surrogate_corner

    arr = _mclmc_arrays(compact)
    fig = plot_mclmc_surrogate_corner(_diag(stage_class, arr), max_samples=200)
    assert isinstance(fig, Figure)
    # The title must name the stage that produced the draws: MAMS is routed through
    # this same function, and a hard-coded "MCLMC" would mislabel its figure.
    assert stage_class.removesuffix("Stage") in fig._suptitle.get_text()


def test_surrogate_uses_the_final_metric_not_the_first(monkeypatch):
    """The surrogate covariance is the metric carried into sampling.

    Reading row 0 of the compact history instead would silently draw the surrogate
    from the STARTING metric — a plausible-looking cloud built from the wrong matrix,
    which is why this asserts on the covariance handed to the sampler rather than on
    the figure rendering at all.
    """
    pytest.importorskip("corner")
    from gigalens_research.plotting import diagnostics as D

    arr = _mclmc_arrays(True)
    seen = {}
    orig_default_rng = np.random.default_rng

    class _SpyRng:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def multivariate_normal(self, mean, cov, size):
            seen["cov"] = np.asarray(cov)
            return self._inner.multivariate_normal(mean, cov, size)

    monkeypatch.setattr(D.np.random, "default_rng",
                        lambda *a, **k: _SpyRng(orig_default_rng(*a, **k)))
    D.plot_mclmc_surrogate_corner(_diag("MCLMCStage", arr), max_samples=100)

    imm = arr["inverse_mass_matrix"]
    np.testing.assert_array_equal(seen["cov"], imm[-1])
    assert not np.allclose(imm[0], imm[-1]), \
        "fixture too weak: first and last metrics must differ for this to bite"
