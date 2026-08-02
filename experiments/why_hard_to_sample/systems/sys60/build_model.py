"""Shim for t2_zspace_diagnostics.py's bijector-dependent parts (C-8-safe name
recovery + theta-space bound-crowding), which do `from build_model import
prob_model`. Delegates to this dir's system.load_target(). Heavy at import
(builds the full model). The T0/T1 harness never sees this file: the
common.load_target dispatcher checks for system.py FIRST."""
from system import load_target as _load_target

prob_model, qz, z_center, dim, param_names = _load_target()
