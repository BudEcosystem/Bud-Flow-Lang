#pragma once

// =============================================================================
// Bud Flow Lang - Arena Allocator
// =============================================================================
//
// High-performance arena allocator for SIMD-aligned allocations.
// Used for IR nodes and temporary compilation data.
//
// Features:
// - O(1) allocation (bump pointer)
// - SIMD-aligned allocations (128-byte alignment)
// - Bulk deallocation (reset entire arena)
// - Thread-local arenas for parallel compilation
//

#include "bud_flow_lang/common.h"
#include "bud_flow_lang/error.h"

#include <atomic>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

namespace bud {

// =============================================================================
// Arena Configuration
// =============================================================================

struct ArenaConfig {
    size_t initial_block_size = 64 * 1024;     // 64 KB default
    size_t max_block_size = 16 * 1024 * 1024;  // 16 MB max
    size_t alignment = kSimdAlignment;         // 128-byte alignment
    bool grow_exponentially = true;            // Double block size on grow
};

// =============================================================================
// Memory Block
// =============================================================================

class ArenaBlock : NonCopyable {
  public:
    explicit ArenaBlock(size_t size);
    ~ArenaBlock();

    ArenaBlock(ArenaBlock&& other) noexcept;
    ArenaBlock& operator=(ArenaBlock&& other) noexcept;

    [[nodiscard]] void* allocate(size_t size, size_t alignment);
    [[nodiscard]] size_t remaining() const { return end_ - current_; }
    [[nodiscard]] size_t used() const { return current_ - start_; }
    [[nodiscard]] size_t capacity() const { return end_ - start_; }

    void reset() { current_ = start_; }

  private:
    char* start_ = nullptr;
    char* current_ = nullptr;
    char* end_ = nullptr;
};

// =============================================================================
// Arena Allocator
// =============================================================================

class Arena : NonCopyable {
  public:
    explicit Arena(ArenaConfig config = {});
    ~Arena();

    Arena(Arena&& other) noexcept;
    Arena& operator=(Arena&& other) noexcept;

    // Allocate raw memory with SIMD alignment
    [[nodiscard]] void* allocate(size_t size);
    [[nodiscard]] void* allocate(size_t size, size_t alignment);

    // Allocate and construct object (with destructor tracking for non-trivial types)
    template <typename T, typename... Args>
    [[nodiscard]] T* create(Args&&... args) {
        static_assert(alignof(T) <= kSimdAlignment, "Type alignment exceeds SIMD alignment");
        void* ptr = allocate(sizeof(T), alignof(T));
        if (!ptr)
            return nullptr;
        T* obj = new (ptr) T(std::forward<Args>(args)...);
        // Register destructor if type has non-trivial destructor
        if constexpr (!std::is_trivially_destructible_v<T>) {
            registerDestructor(obj, [](void* p) { static_cast<T*>(p)->~T(); });
        }
        return obj;
    }

    // Allocate array with overflow protection
    template <typename T>
    [[nodiscard]] T* allocateArray(size_t count) {
        if (count == 0)
            return nullptr;
        // Integer overflow check: ensure sizeof(T) * count won't overflow
        if (count > SIZE_MAX / sizeof(T)) {
            return nullptr;  // Overflow would occur
        }
        void* ptr = allocate(sizeof(T) * count, alignof(T));
        if (!ptr)
            return nullptr;
        // Default construct all elements
        T* arr = static_cast<T*>(ptr);
        for (size_t i = 0; i < count; ++i) {
            new (&arr[i]) T();
        }
        // Register destructor for non-trivial types
        if constexpr (!std::is_trivially_destructible_v<T>) {
            registerArrayDestructor(arr, count, sizeof(T),
                                    [](void* p) { static_cast<T*>(p)->~T(); });
        }
        return arr;
    }

    // Reset arena (deallocate all at once, calls destructors)
    void reset();

    // Statistics
    [[nodiscard]] size_t totalAllocated() const;
    [[nodiscard]] size_t totalCapacity() const;
    [[nodiscard]] size_t blockCount() const { return blocks_.size(); }

  private:
    void addBlock(size_t min_size);
    void registerDestructor(void* ptr, void (*dtor)(void*));
    void registerArrayDestructor(void* ptr, size_t count, size_t element_size, void (*dtor)(void*));
    void callDestructors();

    ArenaConfig config_;
    std::vector<ArenaBlock> blocks_;
    size_t current_block_ = 0;
    size_t total_allocated_ = 0;

    // Destructor tracking for objects with non-trivial destructors
    struct DestructorEntry {
        void* ptr;
        size_t count;         // 1 for single objects, >1 for arrays
        size_t element_size;  // Size of each element (for arrays)
        void (*dtor)(void*);
    };
    std::vector<DestructorEntry> destructors_;
};

// =============================================================================
// Thread-Local Arena
// =============================================================================

// Get thread-local arena for temporary allocations during compilation
Arena& threadLocalArena();

// Scoped arena reset (RAII)
class ScopedArenaReset : NonCopyable {
  public:
    explicit ScopedArenaReset(Arena& arena);
    ~ScopedArenaReset();

  private:
    Arena& arena_;
    size_t saved_allocated_;
};

// =============================================================================
// Arena-Aware Smart Pointers
// =============================================================================

// Deleter that does nothing (arena handles deallocation)
struct ArenaDeleter {
    template <typename T>
    void operator()(T*) const noexcept {
        // Arena handles deallocation, so this is a no-op
        // Destructor is NOT called automatically
    }
};

template <typename T>
using ArenaPtr = std::unique_ptr<T, ArenaDeleter>;

template <typename T, typename... Args>
ArenaPtr<T> makeArenaPtr(Arena& arena, Args&&... args) {
    T* ptr = arena.create<T>(std::forward<Args>(args)...);
    return ArenaPtr<T>(ptr);
}

}  // namespace bud
