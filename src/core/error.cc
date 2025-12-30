// =============================================================================
// Bud Flow Lang - Error Handling Implementation
// =============================================================================

#include "bud_flow_lang/error.h"

#include "bud_flow_lang/common.h"

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <cstdio>
#include <cstdlib>

namespace bud {

// =============================================================================
// Assert Failure
// =============================================================================

[[noreturn]] void assertFailed(const char* cond, const char* file, int line) {
    spdlog::critical("Assertion failed: {} at {}:{}", cond, file, line);
    std::fflush(stderr);
    std::abort();
}

// =============================================================================
// Bounds Check Failure (Always Active - Security Critical)
// =============================================================================

[[noreturn]] void boundsFailed(const char* cond, const char* file, int line) {
    spdlog::critical("SECURITY: Bounds check failed: {} at {}:{}", cond, file, line);
    std::fflush(stderr);
    std::abort();
}

// =============================================================================
// Error Code to String
// =============================================================================

std::string_view errorCodeToString(ErrorCode code) {
    switch (code) {
    case ErrorCode::kOk:
        return "OK";

    // Memory errors
    case ErrorCode::kOutOfMemory:
        return "OutOfMemory";
    case ErrorCode::kAllocationFailed:
        return "AllocationFailed";
    case ErrorCode::kAlignmentError:
        return "AlignmentError";
    case ErrorCode::kBufferTooSmall:
        return "BufferTooSmall";

    // Type errors
    case ErrorCode::kTypeMismatch:
        return "TypeMismatch";
    case ErrorCode::kInvalidType:
        return "InvalidType";
    case ErrorCode::kUnsupportedType:
        return "UnsupportedType";
    case ErrorCode::kShapeMismatch:
        return "ShapeMismatch";

    // IR errors
    case ErrorCode::kInvalidIR:
        return "InvalidIR";
    case ErrorCode::kMalformedIR:
        return "MalformedIR";
    case ErrorCode::kUnknownOp:
        return "UnknownOp";
    case ErrorCode::kInvalidOperand:
        return "InvalidOperand";
    case ErrorCode::kCyclicDependency:
        return "CyclicDependency";

    // JIT errors
    case ErrorCode::kCompilationFailed:
        return "CompilationFailed";
    case ErrorCode::kCodeGenFailed:
        return "CodeGenFailed";
    case ErrorCode::kStencilNotFound:
        return "StencilNotFound";
    case ErrorCode::kCacheError:
        return "CacheError";
    case ErrorCode::kOptimizationFailed:
        return "OptimizationFailed";

    // Runtime errors
    case ErrorCode::kExecutionFailed:
        return "ExecutionFailed";
    case ErrorCode::kDivisionByZero:
        return "DivisionByZero";
    case ErrorCode::kNumericalOverflow:
        return "NumericalOverflow";
    case ErrorCode::kInvalidInput:
        return "InvalidInput";
    case ErrorCode::kIndexOutOfBounds:
        return "IndexOutOfBounds";

    // Python binding errors
    case ErrorCode::kPythonError:
        return "PythonError";
    case ErrorCode::kAstParseFailed:
        return "AstParseFailed";
    case ErrorCode::kUnsupportedPythonFeature:
        return "UnsupportedPythonFeature";

    // Hardware errors
    case ErrorCode::kHardwareNotSupported:
        return "HardwareNotSupported";
    case ErrorCode::kIsaNotAvailable:
        return "IsaNotAvailable";

    // General errors
    case ErrorCode::kNotSupported:
        return "NotSupported";
    case ErrorCode::kRuntimeError:
        return "RuntimeError";

    // Internal errors
    case ErrorCode::kInternalError:
        return "InternalError";
    case ErrorCode::kNotImplemented:
        return "NotImplemented";
    case ErrorCode::kUnreachable:
        return "Unreachable";

    default:
        return "UnknownError";
    }
}

// =============================================================================
// Error::toString
// =============================================================================

std::string Error::toString() const {
    if (isOk()) {
        return "OK";
    }

    auto code_str = errorCodeToString(code_);
    if (message_.empty()) {
        return std::string(code_str);
    }

    return fmt::format("{}: {}", code_str, message_);
}

}  // namespace bud
