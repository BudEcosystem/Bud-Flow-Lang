// =============================================================================
// Bud Flow Lang - Benchmark Entry Point
// =============================================================================

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include "bud_flow_lang/bud_flow_lang.h"

int main(int argc, char** argv) {
    bud::initialize();
    // Benchmarks are run from other files
    bud::shutdown();
    return 0;
}
