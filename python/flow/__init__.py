# =============================================================================
# Bud Flow Lang - High-Performance SIMD DSL for Python
# =============================================================================
#
# Flow provides a tiered API for SIMD programming:
#
# - User Tier: NumPy-like API with automatic optimization (default)
# - Developer Tier: Pattern, Pipeline, Memory hints for explicit control
# - Expert Tier: Direct Highway SIMD access via flow.highway
#
# Usage:
#   from flow import Bunch, Pattern, Pipeline, Memory
#   from flow.highway import TagF32, get_simd_target
#
# =============================================================================

import bud_flow_lang_py as _core

# Core types
Bunch = _core.Bunch
ScalarType = _core.ScalarType

# Developer tier
Pattern = _core.Pattern
Pipeline = _core.Pipeline
Memory = _core.Memory
Prefetch = _core.Prefetch
BlockIterator = _core.BlockIterator

# Memory constants
CACHE_LINE_SIZE = _core.CACHE_LINE_SIZE
AVX_ALIGNMENT = _core.AVX_ALIGNMENT
AVX512_ALIGNMENT = _core.AVX512_ALIGNMENT
SSE_ALIGNMENT = _core.SSE_ALIGNMENT

# Expert tier (available as flow.highway)
from . import highway

# Helper functions
def select(bunch, pattern):
    """Select elements from bunch using pattern"""
    return _core.select(bunch, pattern)

def scatter(bunch, values, pattern):
    """Scatter values into bunch using pattern"""
    return _core.scatter(bunch, values, pattern)

# Version
__version__ = "0.1.0"

__all__ = [
    # Core
    "Bunch",
    "ScalarType",
    # Developer tier
    "Pattern",
    "Pipeline",
    "Memory",
    "Prefetch",
    "BlockIterator",
    "select",
    "scatter",
    # Constants
    "CACHE_LINE_SIZE",
    "AVX_ALIGNMENT",
    "AVX512_ALIGNMENT",
    "SSE_ALIGNMENT",
    # Expert tier
    "highway",
]
