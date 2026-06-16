"""Built-in experiment modules.

Importing this package registers:
- GL2 Sérsic and Vela shapelets inference/generator builders
- Standard pipeline builders (map_svi_hmc, map_bootstrap_mclmc)
- All built-in metrics

The CLI imports this automatically; user scripts can do::

    import gigalens_research.simtests.experiments  # registers all built-ins
"""
# Pipeline builders + PartialTruthBootstrapQzStage must be imported first so that
# map_svi_hmc and map_bootstrap_mclmc are in the registry before experiments
# try to reference them (vela_sersic_shapelets imports the stage from pipelines).
from .. import pipelines  # noqa: F401
from .. import metrics    # noqa: F401
from . import gl2_sersic      # noqa: F401
from . import vela_shapelets  # noqa: F401
from . import vela_sersic_shapelets  # noqa: F401
