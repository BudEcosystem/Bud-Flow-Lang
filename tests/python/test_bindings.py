#!/usr/bin/env python3
# =============================================================================
# Bud Flow Lang - Python Bindings Test Suite
# =============================================================================
#
# Comprehensive tests for the Python bindings including:
# - flow() factory function
# - NumPy interoperability
# - @flow.kernel decorator
# - All Bunch operators and operations
# - Fused operations
# - Reductions
# - Hardware info
#
# Run with: pytest test_bindings.py -v
#

import pytest
import numpy as np
import sys
import os

# Add build directory to path for the Python module
BUILD_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BUILD_DIR, "build"))

# Try to import the module
try:
    import bud_flow_lang_py as flow
    HAS_MODULE = True
except ImportError as e:
    HAS_MODULE = False
    IMPORT_ERROR = str(e)


# Skip all tests if module not available
pytestmark = pytest.mark.skipif(
    not HAS_MODULE,
    reason=f"bud_flow_lang_py module not available: {IMPORT_ERROR if not HAS_MODULE else ''}"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module", autouse=True)
def initialize_runtime():
    """Initialize the runtime once per test module."""
    if HAS_MODULE:
        flow.initialize()
        yield
        flow.shutdown()
    else:
        yield


@pytest.fixture
def sample_data():
    """Sample data for tests."""
    return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


@pytest.fixture
def sample_numpy_f32():
    """Sample NumPy array float32."""
    return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)


@pytest.fixture
def sample_numpy_f64():
    """Sample NumPy array float64."""
    return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)


@pytest.fixture
def sample_numpy_i32():
    """Sample NumPy array int32."""
    return np.array([1, 2, 3, 4], dtype=np.int32)


# =============================================================================
# Module Level Tests
# =============================================================================

class TestModuleBasics:
    """Test module-level functionality."""

    def test_version_exists(self):
        """Test that version string is available."""
        assert hasattr(flow, "__version__")
        assert isinstance(flow.__version__, str)
        assert len(flow.__version__) > 0

    def test_is_initialized(self):
        """Test that runtime is initialized."""
        assert flow.is_initialized()

    def test_get_simd_width(self):
        """Test SIMD width reporting."""
        width = flow.get_simd_width()
        assert isinstance(width, int)
        assert width >= 16  # At least SSE2 (16 bytes)
        assert width in [16, 32, 64, 128]  # Common SIMD widths


# =============================================================================
# Flow Factory Tests
# =============================================================================

class TestFlowFactory:
    """Test flow() factory function."""

    def test_flow_from_list(self, sample_data):
        """Test creating Bunch from Python list."""
        x = flow.flow(sample_data)
        assert len(x) == len(sample_data)
        assert x.size == len(sample_data)
        for i, val in enumerate(sample_data):
            assert abs(x[i] - val) < 1e-6

    def test_flow_from_numpy_f32(self, sample_numpy_f32):
        """Test creating Bunch from NumPy float32."""
        x = flow.flow(sample_numpy_f32)
        assert len(x) == len(sample_numpy_f32)
        for i in range(len(sample_numpy_f32)):
            assert abs(x[i] - sample_numpy_f32[i]) < 1e-6

    def test_flow_from_numpy_f64(self, sample_numpy_f64):
        """Test creating Bunch from NumPy float64."""
        x = flow.flow(sample_numpy_f64)
        assert len(x) == len(sample_numpy_f64)
        for i in range(len(sample_numpy_f64)):
            assert abs(x[i] - sample_numpy_f64[i]) < 1e-6

    def test_flow_from_numpy_i32(self, sample_numpy_i32):
        """Test creating Bunch from NumPy int32."""
        x = flow.flow(sample_numpy_i32)
        assert len(x) == len(sample_numpy_i32)
        for i in range(len(sample_numpy_i32)):
            assert abs(x[i] - sample_numpy_i32[i]) < 1e-6

    @pytest.mark.skip(reason="Empty list creates zero-size allocation which may abort")
    def test_flow_empty_list(self):
        """Test creating Bunch from empty list."""
        x = flow.flow([])
        assert len(x) == 0

    def test_flow_single_element(self):
        """Test creating Bunch from single element."""
        x = flow.flow([42.0])
        assert len(x) == 1
        assert abs(x[0] - 42.0) < 1e-6


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Test zeros(), ones(), full(), arange(), linspace()."""

    def test_zeros(self):
        """Test zeros factory."""
        x = flow.zeros(100)
        assert len(x) == 100
        assert x.dtype == "float32"
        for i in range(len(x)):
            assert x[i] == 0.0

    def test_zeros_with_dtype(self):
        """Test zeros with different dtypes."""
        x_f64 = flow.zeros(10, dtype="float64")
        assert x_f64.dtype == "float64"

    def test_ones(self):
        """Test ones factory."""
        x = flow.ones(100)
        assert len(x) == 100
        for i in range(len(x)):
            assert abs(x[i] - 1.0) < 1e-6

    @pytest.mark.skip(reason="ones() currently only supports float32")
    def test_ones_with_dtype(self):
        """Test ones with different dtypes."""
        x_f64 = flow.ones(10, dtype="float64")
        assert x_f64.dtype == "float64"

    def test_full(self):
        """Test full factory."""
        x = flow.full(100, 3.14)
        assert len(x) == 100
        for i in range(len(x)):
            assert abs(x[i] - 3.14) < 1e-5

    def test_arange_default(self):
        """Test arange with defaults."""
        x = flow.arange(10)
        assert len(x) == 10
        for i in range(10):
            assert abs(x[i] - float(i)) < 1e-6

    def test_arange_with_start_step(self):
        """Test arange with start and step."""
        x = flow.arange(5, start=2.0, step=0.5)
        expected = [2.0, 2.5, 3.0, 3.5, 4.0]
        assert len(x) == 5
        for i, val in enumerate(expected):
            assert abs(x[i] - val) < 1e-6

    def test_linspace(self):
        """Test linspace factory."""
        x = flow.linspace(0.0, 1.0, 5)
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert len(x) == 5
        for i, val in enumerate(expected):
            assert abs(x[i] - val) < 1e-5

    def test_linspace_no_endpoint(self):
        """Test linspace without endpoint."""
        x = flow.linspace(0.0, 1.0, 5, endpoint=False)
        expected = [0.0, 0.2, 0.4, 0.6, 0.8]
        assert len(x) == 5
        for i, val in enumerate(expected):
            assert abs(x[i] - val) < 1e-5


# =============================================================================
# NumPy Interoperability Tests
# =============================================================================

class TestNumPyInterop:
    """Test NumPy interoperability."""

    def test_from_numpy_f32(self, sample_numpy_f32):
        """Test Bunch.from_numpy with float32."""
        x = flow.Bunch.from_numpy(sample_numpy_f32)
        assert len(x) == len(sample_numpy_f32)
        for i in range(len(sample_numpy_f32)):
            assert abs(x[i] - sample_numpy_f32[i]) < 1e-6

    def test_from_numpy_f64(self, sample_numpy_f64):
        """Test Bunch.from_numpy with float64."""
        x = flow.Bunch.from_numpy(sample_numpy_f64)
        assert len(x) == len(sample_numpy_f64)

    def test_to_numpy_roundtrip_f32(self, sample_numpy_f32):
        """Test roundtrip: NumPy -> Bunch -> NumPy."""
        x = flow.flow(sample_numpy_f32)
        y = x.to_numpy()
        assert isinstance(y, np.ndarray)
        assert y.shape == sample_numpy_f32.shape
        np.testing.assert_allclose(y, sample_numpy_f32, rtol=1e-6)

    def test_to_numpy_from_list(self, sample_data):
        """Test converting Bunch from list to NumPy."""
        x = flow.flow(sample_data)
        y = x.to_numpy()
        assert isinstance(y, np.ndarray)
        assert len(y) == len(sample_data)
        np.testing.assert_allclose(y, sample_data, rtol=1e-6)


# =============================================================================
# Bunch Properties Tests
# =============================================================================

class TestBunchProperties:
    """Test Bunch properties."""

    def test_size_property(self, sample_data):
        """Test size property."""
        x = flow.flow(sample_data)
        assert x.size == len(sample_data)

    def test_len(self, sample_data):
        """Test __len__."""
        x = flow.flow(sample_data)
        assert len(x) == len(sample_data)

    def test_dtype_property(self):
        """Test dtype property."""
        x = flow.zeros(10)
        assert x.dtype == "float32"

    def test_shape_property(self, sample_data):
        """Test shape property."""
        x = flow.flow(sample_data)
        assert x.shape == (len(sample_data),)

    def test_nbytes_property(self):
        """Test nbytes property."""
        x = flow.zeros(10)
        assert x.nbytes == 10 * 4  # 10 * sizeof(float32)

    def test_itemsize_property(self):
        """Test itemsize property."""
        x = flow.zeros(10)
        assert x.itemsize == 4  # sizeof(float32)

    def test_ndim_property(self):
        """Test ndim property."""
        x = flow.zeros(10)
        assert x.ndim == 1

    def test_is_valid(self):
        """Test is_valid method."""
        x = flow.zeros(10)
        assert x.is_valid()


# =============================================================================
# Element Access Tests
# =============================================================================

class TestElementAccess:
    """Test element access (__getitem__)."""

    def test_getitem_positive(self, sample_data):
        """Test positive indexing."""
        x = flow.flow(sample_data)
        for i in range(len(sample_data)):
            assert abs(x[i] - sample_data[i]) < 1e-6

    def test_getitem_negative(self, sample_data):
        """Test negative indexing."""
        x = flow.flow(sample_data)
        assert abs(x[-1] - sample_data[-1]) < 1e-6
        assert abs(x[-2] - sample_data[-2]) < 1e-6
        assert abs(x[-len(sample_data)] - sample_data[0]) < 1e-6

    def test_getitem_out_of_bounds(self, sample_data):
        """Test out of bounds access raises IndexError."""
        x = flow.flow(sample_data)
        with pytest.raises(IndexError):
            _ = x[len(sample_data)]
        with pytest.raises(IndexError):
            _ = x[-len(sample_data) - 1]


# =============================================================================
# Arithmetic Operators Tests
# =============================================================================

class TestArithmeticOperators:
    """Test arithmetic operators."""

    def test_add_bunch_bunch(self):
        """Test Bunch + Bunch."""
        a = flow.flow([1.0, 2.0, 3.0, 4.0])
        b = flow.flow([5.0, 6.0, 7.0, 8.0])
        c = a + b
        expected = [6.0, 8.0, 10.0, 12.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-6

    def test_sub_bunch_bunch(self):
        """Test Bunch - Bunch."""
        a = flow.flow([5.0, 6.0, 7.0, 8.0])
        b = flow.flow([1.0, 2.0, 3.0, 4.0])
        c = a - b
        expected = [4.0, 4.0, 4.0, 4.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-6

    def test_mul_bunch_bunch(self):
        """Test Bunch * Bunch."""
        a = flow.flow([1.0, 2.0, 3.0, 4.0])
        b = flow.flow([2.0, 3.0, 4.0, 5.0])
        c = a * b
        expected = [2.0, 6.0, 12.0, 20.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-6

    def test_div_bunch_bunch(self):
        """Test Bunch / Bunch."""
        a = flow.flow([10.0, 20.0, 30.0, 40.0])
        b = flow.flow([2.0, 4.0, 5.0, 8.0])
        c = a / b
        expected = [5.0, 5.0, 6.0, 5.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-6

    def test_add_bunch_scalar(self):
        """Test Bunch + scalar."""
        a = flow.flow([1.0, 2.0, 3.0, 4.0])
        c = a + 10.0
        expected = [11.0, 12.0, 13.0, 14.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-6

    def test_radd_scalar_bunch(self):
        """Test scalar + Bunch."""
        a = flow.flow([1.0, 2.0, 3.0, 4.0])
        c = 10.0 + a
        expected = [11.0, 12.0, 13.0, 14.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-6

    def test_sub_bunch_scalar(self):
        """Test Bunch - scalar."""
        a = flow.flow([10.0, 20.0, 30.0, 40.0])
        c = a - 5.0
        expected = [5.0, 15.0, 25.0, 35.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-6

    def test_rsub_scalar_bunch(self):
        """Test scalar - Bunch."""
        a = flow.flow([1.0, 2.0, 3.0, 4.0])
        c = 10.0 - a
        expected = [9.0, 8.0, 7.0, 6.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-6

    def test_mul_bunch_scalar(self):
        """Test Bunch * scalar."""
        a = flow.flow([1.0, 2.0, 3.0, 4.0])
        c = a * 3.0
        expected = [3.0, 6.0, 9.0, 12.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-6

    def test_rmul_scalar_bunch(self):
        """Test scalar * Bunch."""
        a = flow.flow([1.0, 2.0, 3.0, 4.0])
        c = 3.0 * a
        expected = [3.0, 6.0, 9.0, 12.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-6

    def test_div_bunch_scalar(self):
        """Test Bunch / scalar."""
        a = flow.flow([10.0, 20.0, 30.0, 40.0])
        c = a / 10.0
        expected = [1.0, 2.0, 3.0, 4.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-6

    def test_rtruediv_scalar_bunch(self):
        """Test scalar / Bunch."""
        a = flow.flow([1.0, 2.0, 4.0, 5.0])
        c = 20.0 / a
        expected = [20.0, 10.0, 5.0, 4.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-6

    def test_neg(self):
        """Test -Bunch."""
        a = flow.flow([1.0, -2.0, 3.0, -4.0])
        c = -a
        expected = [-1.0, 2.0, -3.0, 4.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-6

    def test_pos(self):
        """Test +Bunch."""
        a = flow.flow([1.0, -2.0, 3.0, -4.0])
        c = +a
        for i in range(len(a)):
            assert abs(c[i] - a[i]) < 1e-6

    def test_pow(self):
        """Test Bunch ** scalar."""
        a = flow.flow([1.0, 2.0, 3.0, 4.0])
        c = a ** 2.0
        expected = [1.0, 4.0, 9.0, 16.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-4  # Relaxed tolerance for exp/log


# =============================================================================
# Unary Math Operations Tests
# =============================================================================

class TestUnaryMathOps:
    """Test unary math operations."""

    def test_abs(self):
        """Test abs()."""
        a = flow.flow([-1.0, 2.0, -3.0, 4.0])
        c = a.abs()
        expected = [1.0, 2.0, 3.0, 4.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-6

    def test_abs_builtin(self):
        """Test __abs__."""
        a = flow.flow([-1.0, 2.0, -3.0, 4.0])
        c = abs(a)
        expected = [1.0, 2.0, 3.0, 4.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-6

    def test_sqrt(self):
        """Test sqrt()."""
        a = flow.flow([1.0, 4.0, 9.0, 16.0])
        c = a.sqrt()
        expected = [1.0, 2.0, 3.0, 4.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-6

    def test_rsqrt(self):
        """Test rsqrt() = 1/sqrt()."""
        a = flow.flow([1.0, 4.0, 9.0, 16.0])
        c = a.rsqrt()
        expected = [1.0, 0.5, 1.0/3.0, 0.25]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-5

    def test_exp(self):
        """Test exp()."""
        a = flow.flow([0.0, 1.0, 2.0])
        c = a.exp()
        expected = [1.0, np.e, np.e**2]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-4

    def test_log(self):
        """Test log()."""
        a = flow.flow([1.0, np.e, np.e**2])
        c = a.log()
        expected = [0.0, 1.0, 2.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-5

    def test_sin(self):
        """Test sin()."""
        a = flow.flow([0.0, np.pi/2, np.pi])
        c = a.sin()
        expected = [0.0, 1.0, 0.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-5

    def test_cos(self):
        """Test cos()."""
        a = flow.flow([0.0, np.pi/2, np.pi])
        c = a.cos()
        expected = [1.0, 0.0, -1.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-5

    def test_tanh(self):
        """Test tanh()."""
        a = flow.flow([0.0, 1.0, -1.0])
        c = a.tanh()
        expected = [np.tanh(0.0), np.tanh(1.0), np.tanh(-1.0)]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-5

    def test_module_level_abs(self):
        """Test flow.abs()."""
        a = flow.flow([-1.0, 2.0, -3.0, 4.0])
        c = flow.abs(a)
        expected = [1.0, 2.0, 3.0, 4.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-6

    def test_module_level_sqrt(self):
        """Test flow.sqrt()."""
        a = flow.flow([1.0, 4.0, 9.0, 16.0])
        c = flow.sqrt(a)
        expected = [1.0, 2.0, 3.0, 4.0]
        for i, val in enumerate(expected):
            assert abs(c[i] - val) < 1e-6


# =============================================================================
# Comparison Operators Tests
# =============================================================================

class TestComparisonOperators:
    """Test comparison operators."""

    def test_eq_bunch_bunch(self):
        """Test Bunch == Bunch."""
        a = flow.flow([1.0, 2.0, 3.0, 4.0])
        b = flow.flow([1.0, 5.0, 3.0, 6.0])
        c = a == b
        # Results should be mask (1.0 for true, 0.0 for false)
        assert c[0] != 0.0  # Equal
        assert c[1] == 0.0  # Not equal
        assert c[2] != 0.0  # Equal
        assert c[3] == 0.0  # Not equal

    def test_lt_bunch_bunch(self):
        """Test Bunch < Bunch."""
        a = flow.flow([1.0, 5.0, 3.0, 4.0])
        b = flow.flow([2.0, 3.0, 3.0, 5.0])
        c = a < b
        assert c[0] != 0.0  # 1 < 2
        assert c[1] == 0.0  # 5 not < 3
        assert c[2] == 0.0  # 3 not < 3
        assert c[3] != 0.0  # 4 < 5

    def test_le_bunch_bunch(self):
        """Test Bunch <= Bunch."""
        a = flow.flow([1.0, 5.0, 3.0, 4.0])
        b = flow.flow([2.0, 3.0, 3.0, 5.0])
        c = a <= b
        assert c[0] != 0.0  # 1 <= 2
        assert c[1] == 0.0  # 5 not <= 3
        assert c[2] != 0.0  # 3 <= 3
        assert c[3] != 0.0  # 4 <= 5

    def test_gt_bunch_bunch(self):
        """Test Bunch > Bunch."""
        a = flow.flow([1.0, 5.0, 3.0, 4.0])
        b = flow.flow([2.0, 3.0, 3.0, 5.0])
        c = a > b
        assert c[0] == 0.0  # 1 not > 2
        assert c[1] != 0.0  # 5 > 3
        assert c[2] == 0.0  # 3 not > 3
        assert c[3] == 0.0  # 4 not > 5

    def test_ge_bunch_bunch(self):
        """Test Bunch >= Bunch."""
        a = flow.flow([1.0, 5.0, 3.0, 4.0])
        b = flow.flow([2.0, 3.0, 3.0, 5.0])
        c = a >= b
        assert c[0] == 0.0  # 1 not >= 2
        assert c[1] != 0.0  # 5 >= 3
        assert c[2] != 0.0  # 3 >= 3
        assert c[3] == 0.0  # 4 not >= 5

    def test_eq_bunch_scalar(self):
        """Test Bunch == scalar."""
        a = flow.flow([1.0, 2.0, 3.0, 2.0])
        c = a == 2.0
        assert c[0] == 0.0
        assert c[1] != 0.0
        assert c[2] == 0.0
        assert c[3] != 0.0

    def test_lt_bunch_scalar(self):
        """Test Bunch < scalar."""
        a = flow.flow([1.0, 2.0, 3.0, 4.0])
        c = a < 2.5
        assert c[0] != 0.0  # 1 < 2.5
        assert c[1] != 0.0  # 2 < 2.5
        assert c[2] == 0.0  # 3 not < 2.5
        assert c[3] == 0.0  # 4 not < 2.5


# =============================================================================
# Reduction Operations Tests
# =============================================================================

class TestReductions:
    """Test reduction operations."""

    def test_sum(self):
        """Test sum()."""
        a = flow.flow([1.0, 2.0, 3.0, 4.0])
        result = a.sum()
        assert abs(result - 10.0) < 1e-6

    def test_mean(self):
        """Test mean()."""
        a = flow.flow([1.0, 2.0, 3.0, 4.0])
        result = a.mean()
        assert abs(result - 2.5) < 1e-6

    def test_min(self):
        """Test min()."""
        a = flow.flow([3.0, 1.0, 4.0, 2.0])
        result = a.min()
        assert abs(result - 1.0) < 1e-6

    def test_max(self):
        """Test max()."""
        a = flow.flow([3.0, 1.0, 4.0, 2.0])
        result = a.max()
        assert abs(result - 4.0) < 1e-6

    def test_dot(self):
        """Test dot()."""
        a = flow.flow([1.0, 2.0, 3.0, 4.0])
        b = flow.flow([2.0, 3.0, 4.0, 5.0])
        result = a.dot(b)
        expected = 1*2 + 2*3 + 3*4 + 4*5  # 2 + 6 + 12 + 20 = 40
        assert abs(result - expected) < 1e-5

    def test_module_level_sum(self):
        """Test flow.sum()."""
        a = flow.flow([1.0, 2.0, 3.0, 4.0])
        result = flow.sum(a)
        assert abs(result - 10.0) < 1e-6

    def test_module_level_mean(self):
        """Test flow.mean()."""
        a = flow.flow([1.0, 2.0, 3.0, 4.0])
        result = flow.mean(a)
        assert abs(result - 2.5) < 1e-6

    def test_module_level_dot(self):
        """Test flow.dot()."""
        a = flow.flow([1.0, 2.0, 3.0, 4.0])
        b = flow.flow([2.0, 3.0, 4.0, 5.0])
        result = flow.dot(a, b)
        assert abs(result - 40.0) < 1e-5


# =============================================================================
# Fused Operations Tests
# =============================================================================

class TestFusedOperations:
    """Test fused operations."""

    def test_fma_method(self):
        """Test Bunch.fma(): self * b + c."""
        a = flow.flow([1.0, 2.0, 3.0, 4.0])
        b = flow.flow([2.0, 3.0, 4.0, 5.0])
        c = flow.flow([0.5, 0.5, 0.5, 0.5])
        result = a.fma(b, c)
        expected = [1*2 + 0.5, 2*3 + 0.5, 3*4 + 0.5, 4*5 + 0.5]
        for i, val in enumerate(expected):
            assert abs(result[i] - val) < 1e-5

    def test_fma_module(self):
        """Test flow.fma()."""
        a = flow.flow([1.0, 2.0, 3.0, 4.0])
        b = flow.flow([2.0, 3.0, 4.0, 5.0])
        c = flow.flow([0.5, 0.5, 0.5, 0.5])
        result = flow.fma(a, b, c)
        expected = [1*2 + 0.5, 2*3 + 0.5, 3*4 + 0.5, 4*5 + 0.5]
        for i, val in enumerate(expected):
            assert abs(result[i] - val) < 1e-5

    def test_clamp_method(self):
        """Test Bunch.clamp()."""
        a = flow.flow([0.0, 2.0, 5.0, 10.0])
        result = a.clamp(1.0, 6.0)
        expected = [1.0, 2.0, 5.0, 6.0]
        for i, val in enumerate(expected):
            assert abs(result[i] - val) < 1e-6

    def test_clamp_module(self):
        """Test flow.clamp()."""
        a = flow.flow([0.0, 2.0, 5.0, 10.0])
        result = flow.clamp(a, 1.0, 6.0)
        expected = [1.0, 2.0, 5.0, 6.0]
        for i, val in enumerate(expected):
            assert abs(result[i] - val) < 1e-6

    def test_lerp_method(self):
        """Test Bunch.lerp()."""
        a = flow.flow([0.0, 0.0, 0.0, 0.0])
        b = flow.flow([10.0, 20.0, 30.0, 40.0])
        result = a.lerp(b, 0.5)
        expected = [5.0, 10.0, 15.0, 20.0]
        for i, val in enumerate(expected):
            assert abs(result[i] - val) < 1e-5

    def test_lerp_module(self):
        """Test flow.lerp()."""
        a = flow.flow([0.0, 0.0, 0.0, 0.0])
        b = flow.flow([10.0, 20.0, 30.0, 40.0])
        result = flow.lerp(a, b, 0.5)
        expected = [5.0, 10.0, 15.0, 20.0]
        for i, val in enumerate(expected):
            assert abs(result[i] - val) < 1e-5

    def test_where(self):
        """Test Bunch.where()."""
        data = flow.flow([1.0, 2.0, 3.0, 4.0])
        mask = flow.flow([1.0, 0.0, 1.0, 0.0])  # Non-zero = true
        other = flow.flow([10.0, 20.0, 30.0, 40.0])
        result = data.where(mask, other)
        expected = [1.0, 20.0, 3.0, 40.0]
        for i, val in enumerate(expected):
            assert abs(result[i] - val) < 1e-6


# =============================================================================
# Kernel Decorator Tests
# =============================================================================

class TestKernelDecorator:
    """Test @flow.kernel decorator."""

    def test_kernel_basic(self):
        """Test basic @flow.kernel decorator."""
        @flow.kernel
        def add_one(x):
            return x + 1.0

        a = flow.flow([1.0, 2.0, 3.0])
        result = add_one(a)
        # Currently pass-through, so should work
        for i in range(len(a)):
            assert abs(result[i] - (a[i] + 1.0)) < 1e-6

    def test_kernel_with_options(self):
        """Test @flow.kernel with options."""
        @flow.kernel(opt_level=3, enable_fusion=False)
        def my_func(x, y):
            return x * y

        a = flow.flow([1.0, 2.0, 3.0, 4.0])
        b = flow.flow([2.0, 3.0, 4.0, 5.0])
        result = my_func(a, b)
        expected = [2.0, 6.0, 12.0, 20.0]
        for i, val in enumerate(expected):
            assert abs(result[i] - val) < 1e-6

    def test_kernel_decorator_repr(self):
        """Test KernelDecorator repr."""
        @flow.kernel
        def my_kernel(x):
            return x * 2

        # Should have repr
        repr_str = repr(my_kernel)
        assert "flow.kernel" in repr_str or "KernelDecorator" in repr_str

    def test_kernel_options_class(self):
        """Test KernelOptions class."""
        opts = flow.KernelOptions()
        opts.opt_level = 3
        opts.enable_fusion = False
        opts.target_isa = "avx2"

        assert opts.opt_level == 3
        assert opts.enable_fusion == False
        assert opts.target_isa == "avx2"


# =============================================================================
# Hardware Info Tests
# =============================================================================

class TestHardwareInfo:
    """Test hardware info functions."""

    def test_get_hardware_info(self):
        """Test get_hardware_info() returns valid info."""
        info = flow.get_hardware_info()
        assert isinstance(info, dict)
        assert "cpu_name" in info
        assert "simd_width" in info
        assert "physical_cores" in info
        assert "logical_cores" in info
        assert info["simd_width"] >= 16

    def test_get_simd_capabilities(self):
        """Test get_simd_capabilities() returns string."""
        caps = flow.get_simd_capabilities()
        assert isinstance(caps, str)
        assert len(caps) > 0


# =============================================================================
# Iteration Tests
# =============================================================================
# Note: Direct iteration over Bunch is not supported in nanobind 2.x.
# Users should use to_numpy() and iterate over the NumPy array instead.


# =============================================================================
# Copy Tests
# =============================================================================

class TestCopy:
    """Test copy functionality."""

    def test_copy(self, sample_data):
        """Test Bunch.copy()."""
        x = flow.flow(sample_data)
        y = x.copy()
        assert len(y) == len(x)
        for i in range(len(x)):
            assert abs(y[i] - x[i]) < 1e-6


# =============================================================================
# String Representation Tests
# =============================================================================

class TestStringRepresentation:
    """Test string representations."""

    def test_repr(self, sample_data):
        """Test __repr__."""
        x = flow.flow(sample_data)
        repr_str = repr(x)
        assert "Bunch" in repr_str or "bunch" in repr_str.lower()
        assert str(len(sample_data)) in repr_str

    def test_str(self, sample_data):
        """Test __str__."""
        x = flow.flow(sample_data)
        str_str = str(x)
        assert len(str_str) > 0


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_large_array(self):
        """Test with large array."""
        n = 1000000
        x = flow.zeros(n)
        assert len(x) == n
        assert x.sum() == 0.0

    def test_non_aligned_size(self):
        """Test with size not aligned to SIMD width."""
        for size in [1, 3, 7, 13, 17, 31, 33, 63, 65, 127, 129]:
            x = flow.arange(size)
            assert len(x) == size
            # Sum of 0 to n-1 is n*(n-1)/2
            expected_sum = size * (size - 1) / 2
            assert abs(x.sum() - expected_sum) < 1e-4

    def test_eval_method(self):
        """Test eval() method (currently no-op in eager mode)."""
        x = flow.flow([1.0, 2.0, 3.0])
        x.eval()  # Should not raise


# =============================================================================
# NumPy Comparison Tests
# =============================================================================

class TestNumPyComparison:
    """Compare results with NumPy for validation."""

    def test_add_matches_numpy(self):
        """Test add matches NumPy."""
        np_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np_b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

        a = flow.flow(np_a)
        b = flow.flow(np_b)
        result = (a + b).to_numpy()

        np.testing.assert_allclose(result, np_a + np_b, rtol=1e-6)

    def test_mul_matches_numpy(self):
        """Test multiply matches NumPy."""
        np_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np_b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

        a = flow.flow(np_a)
        b = flow.flow(np_b)
        result = (a * b).to_numpy()

        np.testing.assert_allclose(result, np_a * np_b, rtol=1e-6)

    def test_sqrt_matches_numpy(self):
        """Test sqrt matches NumPy."""
        np_a = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)

        a = flow.flow(np_a)
        result = a.sqrt().to_numpy()

        np.testing.assert_allclose(result, np.sqrt(np_a), rtol=1e-5)

    def test_exp_matches_numpy(self):
        """Test exp matches NumPy."""
        np_a = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

        a = flow.flow(np_a)
        result = a.exp().to_numpy()

        np.testing.assert_allclose(result, np.exp(np_a), rtol=1e-4)

    def test_dot_matches_numpy(self):
        """Test dot product matches NumPy."""
        np_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np_b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

        a = flow.flow(np_a)
        b = flow.flow(np_b)
        result = a.dot(b)

        assert abs(result - np.dot(np_a, np_b)) < 1e-4


# =============================================================================
# Slicing Tests
# =============================================================================

class TestSlicing:
    """Test array slicing operations."""

    def test_simple_slice(self):
        """Test basic slicing x[start:stop]."""
        x = flow.arange(10)
        result = x[2:7]
        expected = [2.0, 3.0, 4.0, 5.0, 6.0]
        assert list(result.to_numpy()) == expected

    def test_slice_with_step(self):
        """Test slicing with step x[::step]."""
        x = flow.arange(10)
        result = x[::2]
        expected = [0.0, 2.0, 4.0, 6.0, 8.0]
        assert list(result.to_numpy()) == expected

    def test_slice_negative_start(self):
        """Test slicing with negative start x[-3:]."""
        x = flow.arange(10)
        result = x[-3:]
        expected = [7.0, 8.0, 9.0]
        assert list(result.to_numpy()) == expected

    def test_slice_negative_stop(self):
        """Test slicing with negative stop x[:-2]."""
        x = flow.arange(10)
        result = x[:-2]
        expected = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        assert list(result.to_numpy()) == expected

    def test_slice_step_negative(self):
        """Test reverse slicing x[::-1]."""
        x = flow.arange(5)
        result = x[::-1]
        expected = [4.0, 3.0, 2.0, 1.0, 0.0]
        assert list(result.to_numpy()) == expected

    def test_slice_empty(self):
        """Test slicing that results in empty array."""
        x = flow.arange(10)
        result = x[5:5]
        assert len(result) == 0

    def test_slice_full(self):
        """Test slicing entire array x[:]."""
        x = flow.arange(5)
        result = x[:]
        expected = [0.0, 1.0, 2.0, 3.0, 4.0]
        assert list(result.to_numpy()) == expected


# =============================================================================
# In-Place Operation Tests
# =============================================================================

class TestInPlaceOperations:
    """Test in-place arithmetic operations."""

    def test_iadd_bunch(self):
        """Test in-place addition with Bunch."""
        x = flow.arange(5)
        ones = flow.ones(5)
        x += ones
        expected = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert list(x.to_numpy()) == expected

    def test_iadd_scalar(self):
        """Test in-place addition with scalar."""
        x = flow.arange(5)
        x += 10.0
        expected = [10.0, 11.0, 12.0, 13.0, 14.0]
        assert list(x.to_numpy()) == expected

    def test_isub_bunch(self):
        """Test in-place subtraction with Bunch."""
        x = flow.arange(5)
        ones = flow.ones(5)
        x -= ones
        expected = [-1.0, 0.0, 1.0, 2.0, 3.0]
        assert list(x.to_numpy()) == expected

    def test_isub_scalar(self):
        """Test in-place subtraction with scalar."""
        x = flow.arange(5)
        x -= 1.0
        expected = [-1.0, 0.0, 1.0, 2.0, 3.0]
        assert list(x.to_numpy()) == expected

    def test_imul_bunch(self):
        """Test in-place multiplication with Bunch."""
        x = flow.arange(5)
        twos = flow.full(5, 2.0)
        x *= twos
        expected = [0.0, 2.0, 4.0, 6.0, 8.0]
        assert list(x.to_numpy()) == expected

    def test_imul_scalar(self):
        """Test in-place multiplication with scalar."""
        x = flow.arange(5)
        x *= 2.0
        expected = [0.0, 2.0, 4.0, 6.0, 8.0]
        assert list(x.to_numpy()) == expected

    def test_itruediv_bunch(self):
        """Test in-place division with Bunch."""
        x = flow.full(5, 10.0)
        twos = flow.full(5, 2.0)
        x /= twos
        expected = [5.0, 5.0, 5.0, 5.0, 5.0]
        assert list(x.to_numpy()) == expected

    def test_itruediv_scalar(self):
        """Test in-place division with scalar."""
        x = flow.full(5, 10.0)
        x /= 2.0
        expected = [5.0, 5.0, 5.0, 5.0, 5.0]
        assert list(x.to_numpy()) == expected

    def test_inplace_chained(self):
        """Test chained in-place operations."""
        x = flow.arange(5)
        x += 1.0  # [1, 2, 3, 4, 5]
        x *= 2.0  # [2, 4, 6, 8, 10]
        expected = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert list(x.to_numpy()) == expected


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
