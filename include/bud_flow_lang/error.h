#pragma once

// =============================================================================
// Bud Flow Lang - Error Handling (Exception-Free)
// =============================================================================
//
// This module implements LLVM-style error handling using Result<T> types
// instead of exceptions. All functions that can fail return Result<T>.
//

#include "bud_flow_lang/common.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <variant>

namespace bud {

// =============================================================================
// Error Codes
// =============================================================================

enum class ErrorCode : uint32_t {
    // Success (not an error)
    kOk = 0,

    // Memory errors (1xx)
    kOutOfMemory = 100,
    kAllocationFailed = 101,
    kAlignmentError = 102,
    kBufferTooSmall = 103,

    // Type errors (2xx)
    kTypeMismatch = 200,
    kInvalidType = 201,
    kUnsupportedType = 202,
    kShapeMismatch = 203,

    // IR errors (3xx)
    kInvalidIR = 300,
    kMalformedIR = 301,
    kUnknownOp = 302,
    kInvalidOperand = 303,
    kCyclicDependency = 304,

    // JIT errors (4xx)
    kCompilationFailed = 400,
    kCodeGenFailed = 401,
    kStencilNotFound = 402,
    kCacheError = 403,
    kOptimizationFailed = 404,

    // Runtime errors (5xx)
    kExecutionFailed = 500,
    kDivisionByZero = 501,
    kNumericalOverflow = 502,
    kInvalidInput = 503,
    kIndexOutOfBounds = 504,

    // Python binding errors (6xx)
    kPythonError = 600,
    kAstParseFailed = 601,
    kUnsupportedPythonFeature = 602,

    // Hardware errors (7xx)
    kHardwareNotSupported = 700,
    kIsaNotAvailable = 701,

    // General errors (8xx)
    kNotSupported = 800,
    kRuntimeError = 801,

    // Internal errors (9xx)
    kInternalError = 900,
    kNotImplemented = 901,
    kUnreachable = 902,
};

// Convert error code to string
std::string_view errorCodeToString(ErrorCode code);

// =============================================================================
// Error Class
// =============================================================================

class Error {
  public:
    Error() : code_(ErrorCode::kOk) {}
    explicit Error(ErrorCode code) : code_(code) {}
    Error(ErrorCode code, std::string message) : code_(code), message_(std::move(message)) {}

    // Check if this is an error
    [[nodiscard]] bool isError() const { return code_ != ErrorCode::kOk; }
    [[nodiscard]] bool isOk() const { return code_ == ErrorCode::kOk; }
    explicit operator bool() const { return isError(); }

    // Accessors
    [[nodiscard]] ErrorCode code() const { return code_; }
    [[nodiscard]] std::string_view message() const { return message_; }

    // Get full error description
    [[nodiscard]] std::string toString() const;

    // Static factory methods
    static Error ok() { return Error(); }
    static Error make(ErrorCode code) { return Error(code); }
    static Error make(ErrorCode code, std::string message) {
        return Error(code, std::move(message));
    }

  private:
    ErrorCode code_;
    std::string message_;
};

// =============================================================================
// Result<T> - Error-or-Value Type
// =============================================================================

template <typename T>
class Result {
  public:
    // Construct from value (success)
    Result(T value) : data_(std::move(value)) {}  // NOLINT: intentional implicit

    // Construct from error
    Result(Error error) : data_(std::move(error)) {}  // NOLINT: intentional implicit
    Result(ErrorCode code) : data_(Error(code)) {}    // NOLINT: intentional implicit

    // Check state
    [[nodiscard]] bool hasValue() const { return std::holds_alternative<T>(data_); }
    [[nodiscard]] bool hasError() const { return std::holds_alternative<Error>(data_); }
    explicit operator bool() const { return hasValue(); }

    // Value access (undefined behavior if hasError())
    [[nodiscard]] T& value() & { return std::get<T>(data_); }
    [[nodiscard]] const T& value() const& { return std::get<T>(data_); }
    [[nodiscard]] T&& value() && { return std::get<T>(std::move(data_)); }

    // Error access (undefined behavior if hasValue())
    [[nodiscard]] const Error& error() const { return std::get<Error>(data_); }

    // Value access with default
    [[nodiscard]] T valueOr(T default_value) const& {
        return hasValue() ? value() : std::move(default_value);
    }

    // Pointer-like access (asserts hasValue() - do not use on error Results)
    T* operator->() {
        BUD_ASSERT(hasValue() && "Dereferencing error Result via operator->");
        return &std::get<T>(data_);
    }
    const T* operator->() const {
        BUD_ASSERT(hasValue() && "Dereferencing error Result via operator->");
        return &std::get<T>(data_);
    }
    T& operator*() & {
        BUD_ASSERT(hasValue() && "Dereferencing error Result via operator*");
        return std::get<T>(data_);
    }
    const T& operator*() const& {
        BUD_ASSERT(hasValue() && "Dereferencing error Result via operator*");
        return std::get<T>(data_);
    }
    T&& operator*() && {
        BUD_ASSERT(hasValue() && "Dereferencing error Result via operator*");
        return std::get<T>(std::move(data_));
    }

  private:
    std::variant<T, Error> data_;
};

// =============================================================================
// Result<void> Specialization
// =============================================================================

template <>
class Result<void> {
  public:
    Result() : error_(Error::ok()) {}
    Result(Error error) : error_(std::move(error)) {}  // NOLINT: intentional implicit
    Result(ErrorCode code) : error_(Error(code)) {}    // NOLINT: intentional implicit

    [[nodiscard]] bool hasValue() const { return error_.isOk(); }
    [[nodiscard]] bool hasError() const { return error_.isError(); }
    explicit operator bool() const { return hasValue(); }

    [[nodiscard]] const Error& error() const { return error_; }

  private:
    Error error_;
};

// =============================================================================
// Error Propagation Macros
// =============================================================================

// Return early if result is an error (like Rust's ? operator)
#define BUD_TRY(expr)               \
    do {                            \
        auto&& _result = (expr);    \
        if (!_result) {             \
            return _result.error(); \
        }                           \
    } while (0)

// Assign value or return error
#define BUD_ASSIGN_OR_RETURN(var, expr) \
    auto&& _result_##var = (expr);      \
    if (!_result_##var) {               \
        return _result_##var.error();   \
    }                                   \
    var = std::move(*_result_##var)

// Return error with message
#define BUD_RETURN_ERROR(code, msg) return ::bud::Error::make(code, msg)

// Return success
#define BUD_RETURN_OK() return ::bud::Error::ok()

}  // namespace bud
