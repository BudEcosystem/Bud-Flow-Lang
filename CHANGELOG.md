# Changelog

All notable changes to Bud Flow Lang will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and build system
- Core type system with Shape and TypeDesc
- Arena allocator for efficient IR node allocation
- IR builder with SSA form support
- Basic optimizer with constant folding and dead code elimination
- Copy-and-patch JIT compiler foundation
- Stencil registry for pre-compiled code templates
- Bunch abstraction for SIMD arrays
- Hardware detection for SIMD capabilities
- Memory pool for runtime allocations
- Comprehensive test suite (48 tests)
- Example programs demonstrating usage
- GitHub Actions CI/CD pipeline
- Pre-commit hooks for code quality enforcement
- Quality gate script for comprehensive checks

### Technical Details
- Highway SIMD library integration for portable vectorization
- Support for SSE4, AVX2, AVX-512, ARM NEON, SVE, RISC-V Vector
- Google Test for unit testing
- nanobind prepared for Python bindings
- spdlog for structured logging
- clang-format and clang-tidy integration

## [0.1.0] - 2024-XX-XX

Initial release (planned).

---

[Unreleased]: https://github.com/BudEcosystem/Bud-Flow-Lang/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/BudEcosystem/Bud-Flow-Lang/releases/tag/v0.1.0
