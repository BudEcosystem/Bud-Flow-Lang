// =============================================================================
// Bud Flow Lang - IR Parser Fuzz Target
// =============================================================================

#include "bud_flow_lang/ir.h"

#include <cstddef>
#include <cstdint>
#include <string>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size == 0)
        return 0;

    // Try to parse as JSON IR
    std::string json(reinterpret_cast<const char*>(data), size);

    try {
        auto result = bud::ir::IRModule::fromJson(json);
        // Result doesn't matter, we're testing for crashes
        (void)result;
    } catch (...) {
        // Exceptions are expected for invalid input
    }

    return 0;
}
