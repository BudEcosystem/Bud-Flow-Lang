#!/usr/bin/env python3
# =============================================================================
# Bud Flow Lang - Pytest Configuration
# =============================================================================

import pytest
import sys
import os

# Add build directory to path for the Python module
BUILD_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BUILD_DIR, "build"))

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add skip marker for tests that need specific hardware
    skip_avx512 = pytest.mark.skip(reason="requires AVX-512 support")
    skip_neon = pytest.mark.skip(reason="requires ARM NEON support")

    for item in items:
        if "avx512" in item.keywords:
            # Check if AVX-512 is available
            try:
                import bud_flow_lang_py as flow
                info = flow.get_hardware_info()
                if not info.get("has_avx512", False):
                    item.add_marker(skip_avx512)
            except ImportError:
                pass

        if "neon" in item.keywords:
            # Check if NEON is available
            try:
                import bud_flow_lang_py as flow
                info = flow.get_hardware_info()
                if not info.get("has_neon", False):
                    item.add_marker(skip_neon)
            except ImportError:
                pass


@pytest.fixture(scope="session")
def flow_module():
    """Provide the flow module for tests that need it."""
    try:
        import bud_flow_lang_py as flow
        return flow
    except ImportError as e:
        pytest.skip(f"bud_flow_lang_py module not available: {e}")
