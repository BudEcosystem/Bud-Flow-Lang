/**
 * @file numa_allocator.cc
 * @brief NUMA-aware memory allocator implementation
 *
 * Provides NUMA-aware allocation on Linux systems with libnuma,
 * and falls back to standard allocation on other platforms.
 */

#include "bud_flow_lang/memory/numa_allocator.h"

#include <cstdlib>
#include <cstring>
#include <new>

#if defined(__linux__)
    #include <fcntl.h>
    #include <sched.h>
    #include <sys/mman.h>
    #include <unistd.h>

    // Try to use libnuma if available, but provide fallbacks
    #if __has_include(<numa.h>) && __has_include(<numaif.h>)
        #define BUD_HAS_NUMA 1
        #include <numa.h>
        #include <numaif.h>
    #else
        #define BUD_HAS_NUMA 0
    #endif
#else
    #define BUD_HAS_NUMA 0
#endif

// For aligned_alloc
#if defined(_WIN32)
    #include <malloc.h>
    #define aligned_alloc(align, size) _aligned_malloc(size, align)
    #define aligned_free(ptr) _aligned_free(ptr)
#else
    #define aligned_free(ptr) free(ptr)
#endif

namespace bud {
namespace memory {

// =============================================================================
// NumaTopology Implementation
// =============================================================================

NumaTopology NumaTopology::detect() {
    NumaTopology topology;

#if BUD_HAS_NUMA
    if (numa_available() >= 0) {
        topology.numa_available_ = true;
        topology.num_nodes_ = static_cast<size_t>(numa_max_node() + 1);

        // Get memory info per node
        topology.total_memory_.resize(topology.num_nodes_);
        topology.free_memory_.resize(topology.num_nodes_);
        topology.cpu_masks_.resize(topology.num_nodes_);

        for (size_t node = 0; node < topology.num_nodes_; ++node) {
            long free_mem = 0;
            long total_mem = numa_node_size(static_cast<int>(node), &free_mem);
            if (total_mem > 0) {
                topology.total_memory_[node] = static_cast<size_t>(total_mem);
                topology.free_memory_[node] = static_cast<size_t>(free_mem);
            }

            // Get CPU mask for this node
            struct bitmask* cpus = numa_allocate_cpumask();
            if (numa_node_to_cpus(static_cast<int>(node), cpus) == 0) {
                uint64_t mask = 0;
                for (unsigned int cpu = 0; cpu < 64 && cpu < cpus->size * 8; ++cpu) {
                    if (numa_bitmask_isbitset(cpus, cpu)) {
                        mask |= (1ULL << cpu);
                    }
                }
                topology.cpu_masks_[node] = mask;
            }
            numa_free_cpumask(cpus);
        }
    } else {
        topology.numa_available_ = false;
        topology.num_nodes_ = 1;
    }
#else
    // No NUMA support - single node
    topology.numa_available_ = false;
    topology.num_nodes_ = 1;
    topology.total_memory_.resize(1);
    topology.free_memory_.resize(1);
    topology.cpu_masks_.resize(1);

    // Try to get system memory info
    #if defined(__linux__)
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        topology.total_memory_[0] = static_cast<size_t>(pages) * static_cast<size_t>(page_size);
        // Approximate free memory
        long avail_pages = sysconf(_SC_AVPHYS_PAGES);
        if (avail_pages > 0) {
            topology.free_memory_[0] =
                static_cast<size_t>(avail_pages) * static_cast<size_t>(page_size);
        }
    }
    #endif
    topology.cpu_masks_[0] = ~0ULL;  // All CPUs
#endif

    return topology;
}

size_t NumaTopology::currentNode() const {
#if BUD_HAS_NUMA
    if (numa_available_) {
        int cpu = sched_getcpu();
        if (cpu >= 0) {
            int node = numa_node_of_cpu(cpu);
            if (node >= 0 && static_cast<size_t>(node) < num_nodes_) {
                return static_cast<size_t>(node);
            }
        }
    }
#endif
    return 0;
}

size_t NumaTopology::totalMemory(size_t node) const {
    if (node < total_memory_.size()) {
        return total_memory_[node];
    }
    return 0;
}

size_t NumaTopology::freeMemory(size_t node) const {
    if (node < free_memory_.size()) {
        return free_memory_[node];
    }
    return 0;
}

uint64_t NumaTopology::cpuMaskForNode(size_t node) const {
    if (node < cpu_masks_.size()) {
        return cpu_masks_[node];
    }
    return 0;
}

// =============================================================================
// NumaAllocator Implementation
// =============================================================================

NumaAllocator::NumaAllocator() : topology_(NumaTopology::detect()) {}

NumaAllocator::~NumaAllocator() = default;

void* NumaAllocator::allocate(size_t size) {
    return allocateLocal(size);
}

void* NumaAllocator::allocateAligned(size_t size, size_t alignment) {
    // Ensure alignment is power of 2 and at least sizeof(void*)
    if (alignment < sizeof(void*)) {
        alignment = sizeof(void*);
    }
    if ((alignment & (alignment - 1)) != 0) {
        // Not power of 2, round up
        size_t p = 1;
        while (p < alignment)
            p <<= 1;
        alignment = p;
    }

    // Ensure size is multiple of alignment for aligned_alloc
    size = ((size + alignment - 1) / alignment) * alignment;

#if BUD_HAS_NUMA
    if (topology_.isNumaAvailable()) {
        // Use mmap for NUMA-aware aligned allocation
        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            return nullptr;
        }

        // Set local memory policy
        unsigned long nodemask = 1UL << topology_.currentNode();
        if (mbind(ptr, size, MPOL_PREFERRED, &nodemask, topology_.numNodes() + 1, MPOL_MF_MOVE) !=
            0) {
            // mbind failed, but memory is still usable
        }
        return ptr;
    }
#endif

    // Standard aligned allocation
    return aligned_alloc(alignment, size);
}

void* NumaAllocator::allocateLocal(size_t size) {
#if BUD_HAS_NUMA
    if (topology_.isNumaAvailable()) {
        void* ptr = numa_alloc_local(size);
        if (ptr)
            return ptr;
    }
#endif

    // Fallback to standard allocation
    return std::malloc(size);
}

void* NumaAllocator::allocateOnNode(size_t size, size_t node) {
#if BUD_HAS_NUMA
    if (topology_.isNumaAvailable() && node < topology_.numNodes()) {
        void* ptr = numa_alloc_onnode(size, static_cast<int>(node));
        if (ptr)
            return ptr;
    }
#endif

    // Fallback to standard allocation
    (void)node;
    return std::malloc(size);
}

void* NumaAllocator::allocateInterleaved(size_t size) {
#if BUD_HAS_NUMA
    if (topology_.isNumaAvailable() && topology_.numNodes() > 1) {
        void* ptr = numa_alloc_interleaved(size);
        if (ptr)
            return ptr;
    }
#endif

    // Fallback to standard allocation
    return std::malloc(size);
}

void NumaAllocator::deallocate(void* ptr, size_t size) {
    if (!ptr)
        return;

#if BUD_HAS_NUMA
    if (topology_.isNumaAvailable()) {
        numa_free(ptr, size);
        return;
    }
#endif

    // Check if this was an aligned allocation (mmap)
#if defined(__linux__)
    // Try munmap first for mmap allocations
    if (munmap(ptr, size) == 0) {
        return;
    }
#endif

    (void)size;
    std::free(ptr);
}

int NumaAllocator::getNodeForAddress(const void* addr) const {
#if BUD_HAS_NUMA
    if (topology_.isNumaAvailable()) {
        int node = -1;
        if (get_mempolicy(&node, nullptr, 0, const_cast<void*>(addr), MPOL_F_NODE | MPOL_F_ADDR) ==
            0) {
            return node;
        }
    }
#endif
    (void)addr;
    return 0;  // Default to node 0
}

bool NumaAllocator::migrateToNode(void* ptr, size_t size, size_t target_node) {
#if BUD_HAS_NUMA
    if (!topology_.isNumaAvailable() || target_node >= topology_.numNodes()) {
        return false;
    }

    unsigned long nodemask = 1UL << target_node;
    return mbind(ptr, size, MPOL_BIND, &nodemask, topology_.numNodes() + 1, MPOL_MF_MOVE) == 0;
#else
    (void)ptr;
    (void)size;
    (void)target_node;
    return false;
#endif
}

bool NumaAllocator::bindThreadToNode(size_t node) {
#if BUD_HAS_NUMA
    if (numa_available() >= 0 && node < static_cast<size_t>(numa_max_node() + 1)) {
        numa_run_on_node(static_cast<int>(node));
        numa_set_preferred(static_cast<int>(node));
        return true;
    }
#endif
    (void)node;
    return false;
}

void* NumaAllocator::allocateWithPolicy(size_t size, int policy, int node) {
#if BUD_HAS_NUMA
    if (topology_.isNumaAvailable()) {
        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            return nullptr;
        }

        unsigned long nodemask = 0;
        if (node >= 0) {
            nodemask = 1UL << node;
        }

        if (mbind(ptr, size, policy, node >= 0 ? &nodemask : nullptr, topology_.numNodes() + 1,
                  0) != 0) {
            // mbind failed, memory still usable with default policy
        }

        return ptr;
    }
#endif
    (void)policy;
    (void)node;
    return std::malloc(size);
}

NumaAllocator& NumaAllocator::global() {
    static NumaAllocator instance;
    return instance;
}

}  // namespace memory
}  // namespace bud
