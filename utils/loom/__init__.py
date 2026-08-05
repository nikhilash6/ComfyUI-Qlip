"""Thin redirect: the licensed implementation lives in qlip.inference.loom."""
from qlip.inference.loom import (  # noqa: F401
    GraphedFn, attach, dequantize_linears, detach, find_regions,
    freeze_scales, install_chain, memoize_by_shape, numeric_diff,
    quantize_linears, release_masters, start_calibration, stats,
    uninstall_chain, unmemoize)
