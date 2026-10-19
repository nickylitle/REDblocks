"""simblocks: building blocks for fixed-time-step explicit dynamics sims.

    import simblocks as sb
    sb.find(domain="structures", category="beams")   # query the catalog
    print(sb.catalog_markdown())

Layout = domain / category / geometry / load (see registry.py). Builders run before the loop,
kernels run inside it (njit-safe), analytic blocks are closed forms for sizing and checks.
"""
from . import _cacheguard
_cacheguard.check()     # numba caches each compiled function by its own file only: clear them all when any source changed

from .registry import REGISTRY, block, find, catalog_markdown
from . import materials, structures, contact, constitutive, friction, actuators, rigid_body, geometry, viz, \
    aero, thermal, integrators, diagnostics

from .registry import load_env_blocks as _load_env_blocks  # noqa: E402
_load_env_blocks()      # SIMBLOCKS_BLOCKS folders of your own blocks (desktop app sets it; study workers inherit it)
