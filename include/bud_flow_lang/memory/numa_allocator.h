/**
 * @file numa_allocator.h
 * @brief NUMA-aware memory allocation
 *
 * Provides NUMA topology detection and memory allocation with
 * node affinity for optimal memory bandwidth on multi-socket systems.
 */

#pragma once

#include "bud_flow_lang/common.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace bud {
namespace memory {

/**
 * @brief NUMA topology information
 *
 * Describes the NUMA configuration of the system including
 * number of nodes and memory per node.
 */
class NumaTopology {
  public:
    /**
     * @brief Detect NUMA topology from the system
     * @return NumaTopology with detected values
     */
    [[nodiscard]] static NumaTopology detect();

    /**
     * @brief Get number of NUMA nodes
     */
    [[nodiscard]] size_t numNodes() const { return num_nodes_; }

    /**
     * @brief Get the NUMA node for the current thread
     */
    [[nodiscard]] size_t currentNode() const;

    /**
     * @brief Get total memory on a node
     * @param node Node index
     * @return Total memory in bytes
     */
    [[nodiscard]] size_t totalMemory(size_t node) const;

    /**
     * @brief Get free memory on a node
     * @param node Node index
     * @return Free memory in bytes
     */
    [[nodiscard]] size_t freeMemory(size_t node) const;

    /**
     * @brief Check if NUMA is available
     */
    [[nodiscard]] bool isNumaAvailable() const { return numa_available_; }

    /**
     * @brief Get CPU affinity mask for a node
     * @param node Node index
     * @return Bitmask of CPUs on this node
     */
    [[nodiscard]] uint64_t cpuMaskForNode(size_t node) const;

  private:
    friend class NumaAllocator;

    NumaTopology() = default;

    size_t num_nodes_ = 1;
    bool numa_available_ = false;
    std::vector<size_t> total_memory_;  // Per-node total memory
    std::vector<size_t> free_memory_;   // Per-node free memory (snapshot)
    std::vector<uint64_t> cpu_masks_;   // Per-node CPU affinity masks
};

/**
 * @brief NUMA-aware memory allocator
 *
 * Provides allocation methods with NUMA awareness:
 * - Local allocation (allocate on current thread's node)
 * - Node-specific allocation
 * - Interleaved allocation (round-robin across nodes)
 */
class NumaAllocator : NonCopyable {
  public:
    /**
     * @brief Construct allocator with auto-detected topology
     */
    NumaAllocator();

    /**
     * @brief Destructor
     */
    ~NumaAllocator();

    /**
     * @brief Get NUMA topology
     */
    [[nodiscard]] const NumaTopology& topology() const { return topology_; }

    /**
     * @brief Allocate memory (uses local policy by default)
     *
     * @param size Number of bytes to allocate
     * @return Pointer to allocated memory, or nullptr on failure
     */
    [[nodiscard]] void* allocate(size_t size);

    /**
     * @brief Allocate memory with SIMD alignment
     *
     * @param size Number of bytes to allocate
     * @param alignment Alignment requirement (must be power of 2)
     * @return Pointer to aligned memory, or nullptr on failure
     */
    [[nodiscard]] void* allocateAligned(size_t size, size_t alignment);

    /**
     * @brief Allocate memory on the current thread's NUMA node
     *
     * Memory is allocated local to the CPU where the calling thread
     * is currently running.
     *
     * @param size Number of bytes to allocate
     * @return Pointer to allocated memory, or nullptr on failure
     */
    [[nodiscard]] void* allocateLocal(size_t size);

    /**
     * @brief Allocate memory on a specific NUMA node
     *
     * @param size Number of bytes to allocate
     * @param node Target NUMA node
     * @return Pointer to allocated memory, or nullptr on failure
     */
    [[nodiscard]] void* allocateOnNode(size_t size, size_t node);

    /**
     * @brief Allocate memory interleaved across all NUMA nodes
     *
     * Memory pages are allocated in round-robin fashion across all
     * available NUMA nodes. Good for data accessed by multiple threads
     * on different nodes.
     *
     * @param size Number of bytes to allocate
     * @return Pointer to allocated memory, or nullptr on failure
     */
    [[nodiscard]] void* allocateInterleaved(size_t size);

    /**
     * @brief Deallocate memory
     *
     * @param ptr Pointer to memory to deallocate
     * @param size Size of allocation (required for NUMA deallocation)
     */
    void deallocate(void* ptr, size_t size);

    /**
     * @brief Get the NUMA node for a memory address
     *
     * @param addr Address to query
     * @return Node index, or -1 if cannot be determined
     */
    [[nodiscard]] int getNodeForAddress(const void* addr) const;

    /**
     * @brief Move memory to a specific NUMA node
     *
     * Migrates existing memory pages to a different NUMA node.
     *
     * @param ptr Pointer to memory
     * @param size Size of memory region
     * @param target_node Target NUMA node
     * @return true if successful
     */
    bool migrateToNode(void* ptr, size_t size, size_t target_node);

    /**
     * @brief Bind current thread to a NUMA node
     *
     * @param node Target NUMA node
     * @return true if successful
     */
    static bool bindThreadToNode(size_t node);

    /**
     * @brief Get global NUMA allocator singleton
     */
    [[nodiscard]] static NumaAllocator& global();

  private:
    NumaTopology topology_;

    // Internal allocation with specified policy
    void* allocateWithPolicy(size_t size, int policy, int node);
};

/**
 * @brief RAII wrapper for NUMA-allocated memory
 */
template <typename T>
class NumaPtr {
  public:
    NumaPtr() = default;

    NumaPtr(T* ptr, size_t count, NumaAllocator* allocator)
        : ptr_(ptr), count_(count), allocator_(allocator) {}

    ~NumaPtr() {
        if (ptr_ && allocator_) {
            allocator_->deallocate(ptr_, count_ * sizeof(T));
        }
    }

    // Move-only
    NumaPtr(NumaPtr&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_), allocator_(other.allocator_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    NumaPtr& operator=(NumaPtr&& other) noexcept {
        if (this != &other) {
            if (ptr_ && allocator_) {
                allocator_->deallocate(ptr_, count_ * sizeof(T));
            }
            ptr_ = other.ptr_;
            count_ = other.count_;
            allocator_ = other.allocator_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    NumaPtr(const NumaPtr&) = delete;
    NumaPtr& operator=(const NumaPtr&) = delete;

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }
    size_t size() const { return count_; }
    explicit operator bool() const { return ptr_ != nullptr; }

  private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
    NumaAllocator* allocator_ = nullptr;
};

/**
 * @brief Helper to create NUMA-allocated array
 */
template <typename T>
NumaPtr<T> makeNumaArray(size_t count, NumaAllocator& allocator) {
    void* ptr = allocator.allocateAligned(count * sizeof(T), kSimdAlignment);
    return NumaPtr<T>(static_cast<T*>(ptr), count, &allocator);
}

}  // namespace memory
}  // namespace bud
