// =============================================================================
// Bud Flow Lang - Arena Allocator Benchmarks
// =============================================================================

#include "bud_flow_lang/arena.h"

#include <nanobench.h>

namespace bud {

void benchArena() {
    ankerl::nanobench::Bench bench;
    bench.title("Arena Allocator");

    bench.run("Arena: 1000 x 64-byte allocations", [&] {
        Arena arena;
        for (int i = 0; i < 1000; ++i) {
            void* ptr = arena.allocate(64);
            ankerl::nanobench::doNotOptimizeAway(ptr);
        }
    });

    bench.run("Arena: Create 1000 objects", [&] {
        Arena arena;
        struct Obj {
            int x;
            float y;
        };
        for (int i = 0; i < 1000; ++i) {
            auto* obj = arena.create<Obj>();
            ankerl::nanobench::doNotOptimizeAway(obj);
        }
    });
}

}  // namespace bud
