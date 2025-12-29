// =============================================================================
// Bud Flow Lang - Basic Usage Example
// =============================================================================

#include "bud_flow_lang/bud_flow_lang.h"

#include <iostream>
#include <vector>

int main() {
    // Initialize runtime
    bud::RuntimeConfig config;
    config.enable_debug_output = true;

    auto init_result = bud::initialize(config);
    if (!init_result) {
        std::cerr << "Failed to initialize: " << init_result.error().toString() << "\n";
        return 1;
    }

    // Print hardware info
    const auto& hw = bud::getHardwareInfo();
    std::cout << "SIMD width: " << hw.simd_width << " bytes\n";

    // Create some data
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto bunch_result = bud::Bunch::fromData(data.data(), data.size());
    if (!bunch_result) {
        std::cerr << "Failed to create Bunch\n";
        return 1;
    }

    bud::Bunch b = *bunch_result;

    // Compute statistics
    std::cout << "Bunch: " << b.toString() << "\n";
    std::cout << "Sum: " << b.sum() << "\n";
    std::cout << "Mean: " << b.mean() << "\n";
    std::cout << "Min: " << b.min() << "\n";
    std::cout << "Max: " << b.max() << "\n";

    // Cleanup
    bud::shutdown();

    return 0;
}
