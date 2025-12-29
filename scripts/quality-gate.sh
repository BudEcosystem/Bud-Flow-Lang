#!/bin/bash
# =============================================================================
# Bud Flow Lang - Quality Gate Script
# =============================================================================
#
# This script runs ALL quality checks required before committing code.
# It is designed to be STRICT and COMPREHENSIVE to ensure production-ready code.
#
# Exit codes:
#   0 - All checks passed
#   1 - One or more checks failed
#
# Usage:
#   ./scripts/quality-gate.sh [--quick] [--fix]
#
# Options:
#   --quick  Skip full rebuild (use existing build)
#   --fix    Auto-fix formatting issues
#
# =============================================================================

set -e  # Exit on first error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"

# Parse arguments
QUICK_MODE=false
FIX_MODE=false
for arg in "$@"; do
    case $arg in
        --quick) QUICK_MODE=true ;;
        --fix) FIX_MODE=true ;;
        --help|-h)
            echo "Usage: $0 [--quick] [--fix]"
            echo "  --quick  Skip full rebuild"
            echo "  --fix    Auto-fix formatting issues"
            exit 0
            ;;
    esac
done

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNINGS=0

# Timing
START_TIME=$(date +%s)

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
}

print_section() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

check_pass() {
    echo -e "  ${GREEN}✓${NC} $1"
    ((PASSED_CHECKS++))
    ((TOTAL_CHECKS++))
}

check_fail() {
    echo -e "  ${RED}✗${NC} $1"
    ((FAILED_CHECKS++))
    ((TOTAL_CHECKS++))
}

check_warn() {
    echo -e "  ${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

check_info() {
    echo -e "  ${BLUE}ℹ${NC} $1"
}

run_check() {
    local name="$1"
    local cmd="$2"

    if eval "$cmd" > /tmp/quality_gate_output.txt 2>&1; then
        check_pass "$name"
        return 0
    else
        check_fail "$name"
        echo -e "${RED}    Output:${NC}"
        head -50 /tmp/quality_gate_output.txt | sed 's/^/    /'
        return 1
    fi
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

print_header "BUD FLOW LANG - QUALITY GATE"
echo -e "  ${BLUE}Project:${NC} ${PROJECT_ROOT}"
echo -e "  ${BLUE}Time:${NC}    $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "  ${BLUE}Mode:${NC}    $([ "$QUICK_MODE" = true ] && echo "Quick" || echo "Full")"

cd "$PROJECT_ROOT"

# Check required tools
print_section "Checking Required Tools"

MISSING_TOOLS=()

command -v cmake >/dev/null 2>&1 && check_pass "cmake found" || { check_fail "cmake not found"; MISSING_TOOLS+=("cmake"); }
command -v ninja >/dev/null 2>&1 && check_pass "ninja found" || { check_fail "ninja not found"; MISSING_TOOLS+=("ninja"); }
command -v clang-format >/dev/null 2>&1 && check_pass "clang-format found" || { check_fail "clang-format not found"; MISSING_TOOLS+=("clang-format"); }
command -v clang-tidy >/dev/null 2>&1 && check_pass "clang-tidy found" || { check_warn "clang-tidy not found (some checks will be skipped)"; }
command -v cppcheck >/dev/null 2>&1 && check_pass "cppcheck found" || { check_warn "cppcheck not found (some checks will be skipped)"; }

if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
    echo -e "\n${RED}ERROR: Required tools missing: ${MISSING_TOOLS[*]}${NC}"
    echo "Please install them before continuing."
    exit 1
fi

# =============================================================================
# Source File Discovery
# =============================================================================

print_section "Discovering Source Files"

# Find all C++ source and header files (excluding build and extern directories)
CPP_FILES=$(find "$PROJECT_ROOT" \
    -type f \( -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" -o -name "*.h" -o -name "*.hpp" -o -name "*.hxx" \) \
    ! -path "*/build/*" \
    ! -path "*/_deps/*" \
    ! -path "*/extern/*" \
    ! -path "*/.git/*" \
    2>/dev/null | sort)

FILE_COUNT=$(echo "$CPP_FILES" | wc -l)
check_info "Found $FILE_COUNT C++ files to check"

# =============================================================================
# Stage 1: Code Formatting Check
# =============================================================================

print_section "Stage 1: Code Formatting (clang-format)"

FORMAT_ERRORS=0
if [ "$FIX_MODE" = true ]; then
    check_info "Auto-fixing formatting issues..."
    echo "$CPP_FILES" | xargs -P $(nproc) clang-format -i --style=file 2>/dev/null || true
    check_pass "Formatting auto-fixed"
else
    check_info "Checking code formatting..."

    for file in $CPP_FILES; do
        if ! clang-format --style=file --dry-run --Werror "$file" 2>/dev/null; then
            check_fail "Formatting error: $file"
            ((FORMAT_ERRORS++))
            if [ $FORMAT_ERRORS -ge 5 ]; then
                check_warn "Showing first 5 formatting errors only..."
                break
            fi
        fi
    done

    if [ $FORMAT_ERRORS -eq 0 ]; then
        check_pass "All files properly formatted"
    else
        echo -e "\n${RED}TIP: Run with --fix to auto-fix formatting issues${NC}"
    fi
fi

# =============================================================================
# Stage 2: Build Verification
# =============================================================================

print_section "Stage 2: Build Verification"

if [ "$QUICK_MODE" = false ] || [ ! -f "$BUILD_DIR/Makefile" ] && [ ! -f "$BUILD_DIR/build.ninja" ]; then
    check_info "Configuring CMake..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    if cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DBUD_ENABLE_TESTS=ON \
        -DBUD_ENABLE_BENCHMARKS=ON \
        > /tmp/cmake_output.txt 2>&1; then
        check_pass "CMake configuration successful"
    else
        check_fail "CMake configuration failed"
        cat /tmp/cmake_output.txt | tail -30
        exit 1
    fi
    cd "$PROJECT_ROOT"
fi

check_info "Building project..."
cd "$BUILD_DIR"

if ninja -j$(nproc) > /tmp/build_output.txt 2>&1; then
    check_pass "Build successful (Release)"
else
    check_fail "Build failed"
    echo -e "${RED}Build errors:${NC}"
    grep -E "error:" /tmp/build_output.txt | head -20 | sed 's/^/    /'
    exit 1
fi

# Check for compiler warnings (treat as errors in strict mode)
WARNING_COUNT=$(grep -c "warning:" /tmp/build_output.txt 2>/dev/null || echo "0")
if [ "$WARNING_COUNT" -gt 0 ]; then
    check_warn "Build produced $WARNING_COUNT compiler warnings"
    grep "warning:" /tmp/build_output.txt | head -10 | sed 's/^/    /'
else
    check_pass "Build produced no warnings"
fi

cd "$PROJECT_ROOT"

# =============================================================================
# Stage 3: Static Analysis
# =============================================================================

print_section "Stage 3: Static Analysis"

# clang-tidy (if available and compile_commands.json exists)
if command -v clang-tidy >/dev/null 2>&1 && [ -f "$BUILD_DIR/compile_commands.json" ]; then
    check_info "Running clang-tidy..."

    TIDY_ERRORS=0
    # Check only source files (not headers, not tests for speed)
    SRC_FILES=$(find "$PROJECT_ROOT/src" -type f \( -name "*.cpp" -o -name "*.cc" \) 2>/dev/null | head -20)

    for file in $SRC_FILES; do
        if ! clang-tidy -p "$BUILD_DIR" "$file" --quiet 2>/dev/null | grep -q "error:"; then
            : # No errors
        else
            ((TIDY_ERRORS++))
            if [ $TIDY_ERRORS -le 3 ]; then
                clang-tidy -p "$BUILD_DIR" "$file" 2>/dev/null | grep -E "error:|warning:" | head -5 | sed 's/^/    /'
            fi
        fi
    done

    if [ $TIDY_ERRORS -eq 0 ]; then
        check_pass "clang-tidy passed"
    else
        check_warn "clang-tidy found $TIDY_ERRORS files with issues"
    fi
else
    check_warn "clang-tidy skipped (not available or no compile_commands.json)"
fi

# cppcheck (if available)
if command -v cppcheck >/dev/null 2>&1; then
    check_info "Running cppcheck..."

    CPPCHECK_OUTPUT=$(cppcheck \
        --enable=warning,style,performance,portability \
        --suppress=missingIncludeSystem \
        --suppress=unmatchedSuppression \
        --suppress=unusedFunction \
        --error-exitcode=1 \
        --quiet \
        -I"$PROJECT_ROOT/include" \
        "$PROJECT_ROOT/src" \
        "$PROJECT_ROOT/include" \
        2>&1)

    if [ $? -eq 0 ]; then
        check_pass "cppcheck passed"
    else
        check_warn "cppcheck found issues"
        echo "$CPPCHECK_OUTPUT" | head -10 | sed 's/^/    /'
    fi
else
    check_warn "cppcheck skipped (not available)"
fi

# =============================================================================
# Stage 4: Unit Tests
# =============================================================================

print_section "Stage 4: Unit Tests"

cd "$BUILD_DIR"

if [ -f "./bud_tests" ]; then
    check_info "Running unit tests..."

    if ./bud_tests --gtest_brief=1 > /tmp/test_output.txt 2>&1; then
        PASSED=$(grep -oP '\[\s*PASSED\s*\]\s*\K\d+' /tmp/test_output.txt || echo "?")
        check_pass "All unit tests passed ($PASSED tests)"
    else
        check_fail "Unit tests failed"
        grep -E "^\[.*FAILED.*\]|FAILED" /tmp/test_output.txt | head -10 | sed 's/^/    /'
        exit 1
    fi
else
    check_fail "Test binary not found"
    exit 1
fi

cd "$PROJECT_ROOT"

# =============================================================================
# Stage 5: Integration Tests
# =============================================================================

print_section "Stage 5: Integration Tests"

cd "$BUILD_DIR"

# Run examples as integration tests
if [ -f "./example_basic" ]; then
    if ./example_basic > /tmp/example_output.txt 2>&1; then
        check_pass "example_basic runs successfully"
    else
        check_fail "example_basic failed"
        cat /tmp/example_output.txt | tail -10 | sed 's/^/    /'
    fi
fi

if [ -f "./example_vectorize" ]; then
    if ./example_vectorize > /tmp/example_output.txt 2>&1; then
        check_pass "example_vectorize runs successfully"
    else
        check_fail "example_vectorize failed"
        cat /tmp/example_output.txt | tail -10 | sed 's/^/    /'
    fi
fi

cd "$PROJECT_ROOT"

# =============================================================================
# Stage 6: Memory Safety (AddressSanitizer build - optional)
# =============================================================================

print_section "Stage 6: Memory Safety Checks"

if [ "$QUICK_MODE" = false ]; then
    check_info "Building with AddressSanitizer..."

    ASAN_BUILD_DIR="${BUILD_DIR}_asan"
    mkdir -p "$ASAN_BUILD_DIR"
    cd "$ASAN_BUILD_DIR"

    if cmake "$PROJECT_ROOT" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
        -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
        -DBUD_ENABLE_TESTS=ON \
        > /tmp/asan_cmake.txt 2>&1; then

        if ninja -j$(nproc) bud_tests > /tmp/asan_build.txt 2>&1; then
            check_pass "ASan build successful"

            check_info "Running tests with ASan..."
            export ASAN_OPTIONS="detect_leaks=1:abort_on_error=1"

            if timeout 60 ./bud_tests --gtest_brief=1 > /tmp/asan_test.txt 2>&1; then
                check_pass "No memory errors detected"
            else
                check_fail "Memory errors detected"
                grep -E "ERROR:|SUMMARY:" /tmp/asan_test.txt | head -10 | sed 's/^/    /'
            fi
        else
            check_warn "ASan build failed (non-critical)"
        fi
    else
        check_warn "ASan CMake configuration failed (non-critical)"
    fi

    cd "$PROJECT_ROOT"
else
    check_warn "Memory safety checks skipped (quick mode)"
fi

# =============================================================================
# Stage 7: Code Coverage (optional)
# =============================================================================

print_section "Stage 7: Code Coverage Analysis"

if [ "$QUICK_MODE" = false ] && command -v gcov >/dev/null 2>&1; then
    check_info "Building with coverage instrumentation..."

    COV_BUILD_DIR="${BUILD_DIR}_coverage"
    mkdir -p "$COV_BUILD_DIR"
    cd "$COV_BUILD_DIR"

    if cmake "$PROJECT_ROOT" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_FLAGS="--coverage -fprofile-arcs -ftest-coverage" \
        -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
        -DBUD_ENABLE_TESTS=ON \
        > /tmp/cov_cmake.txt 2>&1; then

        if ninja -j$(nproc) bud_tests > /tmp/cov_build.txt 2>&1; then
            ./bud_tests --gtest_brief=1 > /dev/null 2>&1 || true

            # Calculate coverage percentage (simplified)
            COVERAGE_INFO=$(gcov -n CMakeFiles/bud_flow_core.dir/src/**/*.cc.gcno 2>/dev/null | grep -oP "Lines executed:\K[0-9.]+" | head -1 || echo "N/A")
            check_info "Code coverage: ${COVERAGE_INFO}%"

            if [ "$COVERAGE_INFO" != "N/A" ]; then
                COV_NUM=$(echo "$COVERAGE_INFO" | cut -d'.' -f1)
                if [ "$COV_NUM" -ge 70 ]; then
                    check_pass "Code coverage meets threshold (>=70%)"
                else
                    check_warn "Code coverage below threshold (<70%)"
                fi
            fi
        else
            check_warn "Coverage build failed (non-critical)"
        fi
    else
        check_warn "Coverage CMake configuration failed (non-critical)"
    fi

    cd "$PROJECT_ROOT"
else
    check_warn "Code coverage skipped (quick mode or gcov not available)"
fi

# =============================================================================
# Stage 8: Documentation Check
# =============================================================================

print_section "Stage 8: Documentation Check"

# Check for required documentation files
[ -f "README.md" ] && check_pass "README.md exists" || check_warn "README.md missing"
[ -f "LICENSE" ] && check_pass "LICENSE exists" || check_warn "LICENSE missing"
[ -f "CHANGELOG.md" ] && check_pass "CHANGELOG.md exists" || check_warn "CHANGELOG.md missing"

# Check for TODO/FIXME/HACK comments (informational)
TODO_COUNT=$(grep -rE "TODO|FIXME|HACK|XXX" --include="*.cpp" --include="*.cc" --include="*.h" --include="*.hpp" "$PROJECT_ROOT/src" "$PROJECT_ROOT/include" 2>/dev/null | wc -l || echo "0")
if [ "$TODO_COUNT" -gt 0 ]; then
    check_warn "Found $TODO_COUNT TODO/FIXME/HACK comments"
else
    check_pass "No TODO/FIXME/HACK comments found"
fi

# =============================================================================
# Stage 9: Security Check
# =============================================================================

print_section "Stage 9: Security Check"

# Check for potential security issues
SECURITY_ISSUES=0

# Check for hardcoded secrets patterns
if grep -rE "(password|secret|api_key|private_key)\s*=\s*['\"][^'\"]+['\"]" --include="*.cpp" --include="*.cc" --include="*.h" "$PROJECT_ROOT/src" "$PROJECT_ROOT/include" 2>/dev/null; then
    check_fail "Potential hardcoded secrets found"
    ((SECURITY_ISSUES++))
else
    check_pass "No hardcoded secrets detected"
fi

# Check for unsafe functions
UNSAFE_FUNCS=$(grep -rE "\b(strcpy|strcat|sprintf|gets|scanf)\s*\(" --include="*.cpp" --include="*.cc" --include="*.h" "$PROJECT_ROOT/src" "$PROJECT_ROOT/include" 2>/dev/null | wc -l || echo "0")
if [ "$UNSAFE_FUNCS" -gt 0 ]; then
    check_warn "Found $UNSAFE_FUNCS uses of unsafe C functions"
else
    check_pass "No unsafe C functions found"
fi

# =============================================================================
# Final Summary
# =============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

print_header "QUALITY GATE SUMMARY"

echo ""
echo -e "  ${BOLD}Results:${NC}"
echo -e "    ${GREEN}Passed:${NC}   $PASSED_CHECKS"
echo -e "    ${RED}Failed:${NC}   $FAILED_CHECKS"
echo -e "    ${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "    ${BLUE}Total:${NC}    $TOTAL_CHECKS"
echo ""
echo -e "  ${BOLD}Duration:${NC} ${DURATION}s"
echo ""

if [ $FAILED_CHECKS -gt 0 ]; then
    echo -e "${RED}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  QUALITY GATE FAILED - $FAILED_CHECKS check(s) failed${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}Please fix the above issues before committing.${NC}"
    exit 1
else
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  QUALITY GATE PASSED - All critical checks passed${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"

    if [ $WARNINGS -gt 0 ]; then
        echo ""
        echo -e "${YELLOW}Note: $WARNINGS warning(s) found. Consider addressing them.${NC}"
    fi

    exit 0
fi
