// =============================================================================
// Bud Flow Lang - Test Main Entry Point
// =============================================================================

#include <spdlog/spdlog.h>

#include <gtest/gtest.h>

int main(int argc, char** argv) {
    // Initialize logging for tests
    spdlog::set_level(spdlog::level::warn);

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
