// =============================================================================
// Bud Flow Lang - JIT Compilation Benchmarks
// =============================================================================

#include "bud_flow_lang/ir.h"

#include <nanobench.h>

namespace bud {

void benchJitCompile() {
    ankerl::nanobench::Bench bench;
    bench.title("JIT Compilation");

    bench.run("IR Build: Simple expression", [&] {
        ir::IRModule module("bench");
        auto& builder = module.builder();

        auto a = builder.constant(1.0f);
        auto b = builder.constant(2.0f);
        auto sum = builder.add(a, b);
        module.setOutput(sum);

        ankerl::nanobench::doNotOptimizeAway(sum.id);
    });

    bench.run("IR Build: Complex expression (10 ops)", [&] {
        ir::IRModule module("bench");
        auto& builder = module.builder();

        auto x = builder.constant(1.0f);
        for (int i = 0; i < 10; ++i) {
            x = builder.mul(x, builder.constant(1.1f));
            x = builder.add(x, builder.constant(0.1f));
        }
        module.setOutput(x);

        ankerl::nanobench::doNotOptimizeAway(x.id);
    });
}

}  // namespace bud
