// =============================================================================
// Bud Flow Lang - Auto-Vectorization Example
// =============================================================================

#include "bud_flow_lang/bud_flow_lang.h"

#include <iostream>

int main() {
    bud::initialize();

    const size_t N = 1000000;

    // Create vectors
    auto a_result = bud::Bunch::fill(N, 2.0f);
    auto b_result = bud::Bunch::fill(N, 3.0f);

    if (!a_result || !b_result) {
        std::cerr << "Failed to create vectors\n";
        return 1;
    }

    bud::Bunch a = *a_result;
    bud::Bunch b = *b_result;

    // Dot product: sum(a * b)
    // This should be auto-vectorized using SIMD
    float dot = bud::dot(a, b);

    std::cout << "Dot product of " << N << " elements: " << dot << "\n";
    std::cout << "Expected: " << (2.0f * 3.0f * N) << "\n";

    bud::shutdown();
    return 0;
}
