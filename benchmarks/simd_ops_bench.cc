// =============================================================================
// Bud Flow Lang - SIMD Operations Benchmarks
// =============================================================================

#include "bud_flow_lang/bunch.h"

#include <nanobench.h>

namespace bud {

void benchSimdOps() {
    ankerl::nanobench::Bench bench;
    bench.title("SIMD Operations");

    const size_t N = 1000000;

    auto a_result = Bunch::ones(N);
    auto b_result = Bunch::fill(N, 2.0f);

    if (!a_result || !b_result)
        return;

    Bunch a = *a_result;
    Bunch b = *b_result;

    bench.run("Sum reduction (1M floats)", [&] {
        float sum = a.sum();
        ankerl::nanobench::doNotOptimizeAway(sum);
    });

    bench.run("Dot product (1M floats)", [&] {
        float dot = a.dot(b);
        ankerl::nanobench::doNotOptimizeAway(dot);
    });

    bench.run("Min/Max (1M floats)", [&] {
        float min_val = a.min();
        float max_val = a.max();
        ankerl::nanobench::doNotOptimizeAway(min_val);
        ankerl::nanobench::doNotOptimizeAway(max_val);
    });
}

}  // namespace bud
