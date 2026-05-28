"""
gigalens_research.simulations

Utilities for constructing simulated lensing source planes.

Currently exposes :class:`ImageBasedLight` — a :mod:`gigalens` light profile
backed by a continuous (linearly-interpolated) 2-D image — and the
:func:`load_vela_source` helper that builds one from a Vela-catalog source
directory in the conventional layout used by
``experiments/vela_sim_systems/lens_vela_system.ipynb``.
"""
from .image_based_light import ImageBasedLight
from .vela import VelaSource, load_vela_source

__all__ = [
    "ImageBasedLight",
    "VelaSource",
    "load_vela_source",
]
