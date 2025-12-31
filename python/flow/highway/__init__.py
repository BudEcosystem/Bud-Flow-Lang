# =============================================================================
# Bud Flow Lang - Highway Expert Tier
# =============================================================================
#
# Direct access to Highway SIMD operations for expert users.
#
# Usage:
#   from flow.highway import TagF32, get_simd_target, add, mul
#
#   # Query SIMD capabilities
#   target = get_simd_target()  # e.g., "AVX2", "AVX-512"
#   lanes = TagF32.lanes        # Number of float32 lanes (e.g., 8 for AVX2)
#
#   # SIMD operations
#   result = add(a, b)          # Vectorized addition
#   result = mul_add(a, b, c)   # Fused multiply-add
#
# =============================================================================

import bud_flow_lang_py as _core

# Get the highway submodule
_highway = _core.highway

# Tag types for type information
TagF32 = _highway.TagF32
TagF64 = _highway.TagF64
TagI32 = _highway.TagI32

# Scalar type constants
float32 = _highway.float32
float64 = _highway.float64
int32 = _highway.int32
int64 = _highway.int64
uint8 = _highway.uint8

# SIMD info functions
get_simd_lanes_f32 = _highway.get_simd_lanes_f32
get_simd_lanes_f64 = _highway.get_simd_lanes_f64
get_simd_lanes_i32 = _highway.get_simd_lanes_i32
get_simd_target = _highway.get_simd_target

# Arithmetic operations
add = _highway.add
sub = _highway.sub
mul = _highway.mul
div = _highway.div
neg = _highway.neg
abs = _highway.abs

# Math operations
sqrt = _highway.sqrt
rsqrt = _highway.rsqrt
exp = _highway.exp
log = _highway.log
sin = _highway.sin
cos = _highway.cos
tanh = _highway.tanh

# FMA
mul_add = _highway.mul_add

# Reductions
reduce_sum = _highway.reduce_sum
reduce_min = _highway.reduce_min
reduce_max = _highway.reduce_max
dot_product = _highway.dot_product

# Comparisons
eq = _highway.eq
lt = _highway.lt
le = _highway.le
gt = _highway.gt
ge = _highway.ge

# Conditional
where = _highway.where
clamp = _highway.clamp

# Raw pointer access (expert only)
data_ptr = _highway.data_ptr
mutable_data_ptr = _highway.mutable_data_ptr

__all__ = [
    # Tag types
    "TagF32",
    "TagF64",
    "TagI32",
    # Scalar types
    "float32",
    "float64",
    "int32",
    "int64",
    "uint8",
    # SIMD info
    "get_simd_lanes_f32",
    "get_simd_lanes_f64",
    "get_simd_lanes_i32",
    "get_simd_target",
    # Arithmetic
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "abs",
    # Math
    "sqrt",
    "rsqrt",
    "exp",
    "log",
    "sin",
    "cos",
    "tanh",
    # FMA
    "mul_add",
    # Reductions
    "reduce_sum",
    "reduce_min",
    "reduce_max",
    "dot_product",
    # Comparisons
    "eq",
    "lt",
    "le",
    "gt",
    "ge",
    # Conditional
    "where",
    "clamp",
    # Raw pointers
    "data_ptr",
    "mutable_data_ptr",
]
