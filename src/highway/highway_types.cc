// =============================================================================
// Bud Flow Lang - Highway Type Wrappers Implementation
// =============================================================================
//
// Implementation of runtime SIMD information functions using Highway's
// multi-target dispatch mechanism.
//
// =============================================================================

#include "bud_flow_lang/highway/highway_types.h"

// Highway includes for ScalableTag and Lanes
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "highway_types-inl.h"
#include <hwy/foreach_target.h>  // IWYU pragma: keep
#include <hwy/highway.h>

#include "highway_types-inl.h"  // For the static target

// =============================================================================
// HWY_EXPORT declarations - must be in hwy namespace or global scope
// =============================================================================
namespace hwy {
HWY_EXPORT(GetSimdLanesF32Impl);
HWY_EXPORT(GetSimdLanesF64Impl);
HWY_EXPORT(GetSimdLanesI32Impl);
HWY_EXPORT(GetSimdTargetImpl);
}  // namespace hwy

// =============================================================================
// Bring dispatch tables into scope
// =============================================================================
using hwy::GetSimdLanesF32ImplHighwayDispatchTable;
using hwy::GetSimdLanesF64ImplHighwayDispatchTable;
using hwy::GetSimdLanesI32ImplHighwayDispatchTable;
using hwy::GetSimdTargetImplHighwayDispatchTable;

namespace bud {
namespace highway {

size_t getSimdLanesF32() {
    return HWY_DYNAMIC_DISPATCH(GetSimdLanesF32Impl)();
}

size_t getSimdLanesF64() {
    return HWY_DYNAMIC_DISPATCH(GetSimdLanesF64Impl)();
}

size_t getSimdLanesI32() {
    return HWY_DYNAMIC_DISPATCH(GetSimdLanesI32Impl)();
}

const char* getSimdTarget() {
    return HWY_DYNAMIC_DISPATCH(GetSimdTargetImpl)();
}

// TagInfo specializations
template <>
size_t TagInfo<float>::lanes() {
    return getSimdLanesF32();
}
template <>
size_t TagInfo<double>::lanes() {
    return getSimdLanesF64();
}
template <>
size_t TagInfo<int32_t>::lanes() {
    return getSimdLanesI32();
}
template <>
size_t TagInfo<int64_t>::lanes() {
    return getSimdLanesF64();
}  // Same as double
template <>
size_t TagInfo<uint8_t>::lanes() {
    return getSimdLanesF32() * 4;
}  // 4x float32

template <>
ScalarType TagInfo<float>::dtype() {
    return ScalarType::kFloat32;
}
template <>
ScalarType TagInfo<double>::dtype() {
    return ScalarType::kFloat64;
}
template <>
ScalarType TagInfo<int32_t>::dtype() {
    return ScalarType::kInt32;
}
template <>
ScalarType TagInfo<int64_t>::dtype() {
    return ScalarType::kInt64;
}
template <>
ScalarType TagInfo<uint8_t>::dtype() {
    return ScalarType::kUint8;
}

template <>
const char* TagInfo<float>::typeName() {
    return "float32";
}
template <>
const char* TagInfo<double>::typeName() {
    return "float64";
}
template <>
const char* TagInfo<int32_t>::typeName() {
    return "int32";
}
template <>
const char* TagInfo<int64_t>::typeName() {
    return "int64";
}
template <>
const char* TagInfo<uint8_t>::typeName() {
    return "uint8";
}

}  // namespace highway
}  // namespace bud
