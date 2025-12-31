// =============================================================================
// Bud Flow Lang - Highway Type Wrappers Inline Implementation
// =============================================================================
//
// This file is included multiple times by foreach_target.h, once per SIMD
// target. Each inclusion generates target-specific implementations.
//
// =============================================================================

// Highway toggle guard pattern - allows this file to be included multiple times
// foreach_target.h toggles HWY_TARGET_TOGGLE, we match that with our guard
#if defined(HIGHWAY_HWY_BUD_TYPES_INL_H_) == defined(HWY_TARGET_TOGGLE)
    #ifdef HIGHWAY_HWY_BUD_TYPES_INL_H_
        #undef HIGHWAY_HWY_BUD_TYPES_INL_H_
    #else
        #define HIGHWAY_HWY_BUD_TYPES_INL_H_
    #endif

    #include <hwy/highway.h>

    #include <cstddef>
    #include <cstdint>

// Highway requires functions used with HWY_EXPORT to be in the hwy namespace
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Get number of float32 lanes for this target
HWY_ATTR size_t GetSimdLanesF32Impl() {
    const hn::ScalableTag<float> d;
    return hn::Lanes(d);
}

// Get number of float64 lanes for this target
HWY_ATTR size_t GetSimdLanesF64Impl() {
    const hn::ScalableTag<double> d;
    return hn::Lanes(d);
}

// Get number of int32 lanes for this target
HWY_ATTR size_t GetSimdLanesI32Impl() {
    const hn::ScalableTag<int32_t> d;
    return hn::Lanes(d);
}

// Get the SIMD target name
HWY_ATTR const char* GetSimdTargetImpl() {
    return hwy::TargetName(HWY_TARGET);
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // guard
