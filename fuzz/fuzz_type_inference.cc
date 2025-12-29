// =============================================================================
// Bud Flow Lang - Type Inference Fuzz Target
// =============================================================================

#include "bud_flow_lang/type_system.h"

#include <cstdint>
#include <cstddef>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) return 0;

    // Use fuzz data to create shapes
    size_t rank1 = data[0] % 5;
    size_t rank2 = data[1] % 5;

    std::vector<size_t> dims1, dims2;
    for (size_t i = 0; i < rank1 && i + 2 < size; ++i) {
        dims1.push_back((data[i + 2] % 100) + 1);
    }
    for (size_t i = 0; i < rank2 && i + rank1 + 2 < size; ++i) {
        dims2.push_back((data[i + rank1 + 2] % 100) + 1);
    }

    bud::Shape s1(dims1);
    bud::Shape s2(dims2);

    // Try broadcasting
    auto result = bud::Shape::broadcast(s1, s2);
    (void)result;

    // Try type inference
    bud::TypeDesc t1(bud::ScalarType::kFloat32, s1);
    bud::TypeDesc t2(bud::ScalarType::kFloat32, s2);

    auto infer_result = bud::TypeInferrer::inferBinaryOp(t1, t2);
    (void)infer_result;

    return 0;
}
