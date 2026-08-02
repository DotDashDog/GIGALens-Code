"""Single source of truth for the MCLMC sampler settings shared by every
arm of experiments T0 and T1 (docs/logs/why-hard-to-sample.md).

The whole point of pre-registration T0/T1 is that the seed-variance arm (T0),
the real-run arm, and the Gaussian-clone arm run BYTE-IDENTICAL sampler
settings, differing only in seed and target. So there is exactly ONE config
object; every driver imports it and every output manifest serializes it. If you
need to change a knob, change it here once and re-run all arms.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StandardMCLMCConfig:
    """The standard MCLMC config for this investigation.

    Fields map 1:1 onto MCLMC_JIT kwargs (src/gigalens_research/inference/mclmc.py:39).
    Everything NOT listed here (init_L, init_step_size, frac_tune1/2/3, ...) is
    deliberately left at MCLMC_JIT's own defaults so the sampler is identical to
    the reference driver_h1h2.py invocation except for burn-in/results length.
    """

    n_hmc: int = 8
    num_burnin_steps: int = 2000
    num_results: int = 2000
    desired_energy_variance: float = 5e-4
    regularize_mass_matrix: bool = True
    progress_bar: bool = False

    def mclmc_kwargs(self, *, seed: int, debug_output: bool = True) -> dict:
        """Exact kwargs to splat into MCLMC_JIT (minus prob_model / qz)."""
        return dict(
            n_hmc=self.n_hmc,
            num_burnin_steps=self.num_burnin_steps,
            num_results=self.num_results,
            desired_energy_variance=self.desired_energy_variance,
            regularize_mass_matrix=self.regularize_mass_matrix,
            progress_bar=self.progress_bar,
            seed=int(seed),
            debug_output=debug_output,
        )

    def to_dict(self) -> dict:
        return asdict(self)


# The one instance every driver uses. Import this, do not build your own.
STANDARD = StandardMCLMCConfig()
