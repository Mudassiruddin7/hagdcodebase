"""Baseline circuit-discovery methods (paper Section IV-D).

All baselines search the same vertex space as HAGD (the candidate features of the
attribution graph) and are scored by the same held-out protocol
(``hagd.extract.finalize_circuit``), so the comparison isolates the search strategy.
"""

from .acdc import run_acdc  # noqa: F401
from .attribution_patching import run_attribution_patching  # noqa: F401
from .exhaustive import run_exhaustive  # noqa: F401
from .sparse_probing import run_sparse_probing  # noqa: F401

RUNNERS = {
    "acdc": run_acdc,
    "attribution_patching": run_attribution_patching,
    "exhaustive": run_exhaustive,
    "sparse_probing": run_sparse_probing,
}
