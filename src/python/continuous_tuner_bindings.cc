// =============================================================================
// Bud Flow Lang - Continuous Auto-Tuner Python Bindings
// =============================================================================

#include "bud_flow_lang/bud_flow_lang.h"
#include "bud_flow_lang/jit/adaptive_executor.h"
#include "bud_flow_lang/jit/adaptive_kernel_cache.h"
#include "bud_flow_lang/scheduler/continuous_tuner.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

void bind_continuous_tuner(nb::module_& m) {
    using namespace bud::scheduler;

    // =========================================================================
    // TuningEventType enum
    // =========================================================================
    nb::enum_<TuningEventType>(m, "TuningEventType", "Type of tuning event")
        .value("VariantSelected", TuningEventType::kVariantSelected)
        .value("VariantRecorded", TuningEventType::kVariantRecorded)
        .value("BestChanged", TuningEventType::kBestChanged)
        .value("RetuneTriggered", TuningEventType::kRetuneTriggered)
        .value("VariantAdded", TuningEventType::kVariantAdded)
        .value("VariantEvicted", TuningEventType::kVariantEvicted);

    // =========================================================================
    // ContinuousTunerConfig struct
    // =========================================================================
    nb::class_<ContinuousTunerConfig>(m, "ContinuousTunerConfig",
                                      "Configuration for the continuous auto-tuner")
        .def(nb::init<>(), "Create default configuration")
        .def_rw("prior_alpha", &ContinuousTunerConfig::prior_alpha,
                "Beta distribution prior alpha (default: 1.0)")
        .def_rw("prior_beta", &ContinuousTunerConfig::prior_beta,
                "Beta distribution prior beta (default: 1.0)")
        .def_rw("exploration_bonus", &ContinuousTunerConfig::exploration_bonus,
                "Bonus for less-explored variants (default: 0.1)")
        .def_rw("warmup_executions", &ContinuousTunerConfig::warmup_executions,
                "Minimum executions before adaptive selection (default: 10)")
        .def_rw("min_samples_per_variant", &ContinuousTunerConfig::min_samples_per_variant,
                "Minimum samples before including variant in selection (default: 5)")
        .def_rw("improvement_threshold", &ContinuousTunerConfig::improvement_threshold,
                "Minimum improvement to switch variants (default: 0.05)")
        .def_rw("time_decay", &ContinuousTunerConfig::time_decay,
                "Weight for historical observations in EMA (default: 0.99)")
        .def_rw("retune_threshold", &ContinuousTunerConfig::retune_threshold,
                "Performance regression to trigger retune (default: 0.2)")
        .def_rw("retune_interval", &ContinuousTunerConfig::retune_interval,
                "Executions between retune checks (default: 1000)")
        .def_rw("max_variants_per_kernel", &ContinuousTunerConfig::max_variants_per_kernel,
                "Maximum variants to maintain per kernel (default: 8)")
        .def_rw("max_profiled_sizes", &ContinuousTunerConfig::max_profiled_sizes,
                "Maximum sizes to profile (default: 16)")
        .def_rw("enable_background_tuning", &ContinuousTunerConfig::enable_background_tuning,
                "Enable tuning new variants in background (default: true)")
        .def_rw("background_tune_batch", &ContinuousTunerConfig::background_tune_batch,
                "Variants to tune per batch (default: 4)");

    // =========================================================================
    // KernelVariant struct
    // =========================================================================
    nb::class_<KernelVariant>(m, "KernelVariant", "Represents a compiled kernel variant")
        .def(nb::init<>(), "Create empty variant")
        .def_rw("id", &KernelVariant::id, "Unique variant ID")
        .def_rw("name", &KernelVariant::name, "Human-readable name")
        .def_rw("is_specialized", &KernelVariant::is_specialized,
                "Whether this is size-specialized")
        .def_prop_ro(
            "total_executions", [](const KernelVariant& v) { return v.total_executions; },
            "Total number of executions")
        .def_prop_ro(
            "total_time_ns", [](const KernelVariant& v) { return v.total_time_ns; },
            "Total execution time in nanoseconds")
        .def_prop_ro(
            "avg_time_ns", [](const KernelVariant& v) { return v.avg_time_ns; },
            "Average execution time (EMA)")
        .def_prop_ro(
            "min_time_ns", [](const KernelVariant& v) { return v.min_time_ns; },
            "Minimum execution time")
        .def_prop_ro(
            "max_time_ns", [](const KernelVariant& v) { return v.max_time_ns; },
            "Maximum execution time");

    // =========================================================================
    // ThompsonState struct
    // =========================================================================
    nb::class_<ThompsonState>(m, "ThompsonState", "Thompson Sampling state for a variant")
        .def(nb::init<>())
        .def_ro("alpha", &ThompsonState::alpha, "Beta distribution alpha")
        .def_ro("beta", &ThompsonState::beta, "Beta distribution beta")
        .def_ro("num_samples", &ThompsonState::num_samples, "Number of samples")
        .def_ro("sum_rewards", &ThompsonState::sum_rewards, "Cumulative rewards")
        .def("mean", &ThompsonState::mean, "Get mean of posterior")
        .def("variance", &ThompsonState::variance, "Get variance of posterior");

    // =========================================================================
    // TuningEvent struct
    // =========================================================================
    nb::class_<TuningEvent>(m, "TuningEvent", "A tuning event for logging")
        .def_ro("type", &TuningEvent::type, "Event type")
        .def_ro("kernel_name", &TuningEvent::kernel_name, "Kernel name")
        .def_ro("variant_id", &TuningEvent::variant_id, "Variant ID")
        .def_ro("time_ns", &TuningEvent::time_ns, "Execution time")
        .def_ro("improvement", &TuningEvent::improvement, "Improvement ratio");

    // =========================================================================
    // ContinuousTunerStats struct
    // =========================================================================
    nb::class_<ContinuousTunerStats>(m, "ContinuousTunerStats",
                                     "Statistics from continuous auto-tuner")
        .def_ro("total_selections", &ContinuousTunerStats::total_selections,
                "Total variant selections")
        .def_ro("exploitation_count", &ContinuousTunerStats::exploitation_count,
                "Times known-best was selected")
        .def_ro("exploration_count", &ContinuousTunerStats::exploration_count,
                "Times new variant was explored")
        .def_ro("total_observations", &ContinuousTunerStats::total_observations,
                "Total execution observations")
        .def_ro("variants_created", &ContinuousTunerStats::variants_created,
                "Number of variants created")
        .def_ro("variants_evicted", &ContinuousTunerStats::variants_evicted,
                "Number of variants evicted")
        .def_ro("avg_selection_overhead_ns", &ContinuousTunerStats::avg_selection_overhead_ns,
                "Average selection overhead")
        .def_ro("cumulative_speedup", &ContinuousTunerStats::cumulative_speedup,
                "Cumulative speedup vs baseline")
        .def_ro("retune_triggers", &ContinuousTunerStats::retune_triggers,
                "Number of retune triggers")
        .def_ro("successful_retunes", &ContinuousTunerStats::successful_retunes,
                "Number of successful retunes");

    // =========================================================================
    // ContinuousAutoTuner class
    // =========================================================================
    nb::class_<ContinuousAutoTuner>(m, "ContinuousAutoTuner",
                                    "Runtime-adaptive kernel selector using Thompson Sampling.\n\n"
                                    "Continuously learns from execution times to select the best\n"
                                    "kernel variant for each workload.\n\n"
                                    "Example:\n"
                                    "  tuner = ContinuousAutoTuner()\n"
                                    "  tuner.register_kernel('my_kernel', variant)\n"
                                    "  idx = tuner.select_variant('my_kernel')\n"
                                    "  # Execute kernel variant idx\n"
                                    "  tuner.record_execution('my_kernel', idx, time_ns)\n")

        // Constructors
        .def(nb::init<>(), "Create with default configuration")
        .def(nb::init<const ContinuousTunerConfig&>(), nb::arg("config"),
             "Create with custom configuration")

        // Configuration
        .def_prop_ro("config", &ContinuousAutoTuner::config, "Get current configuration")
        .def("set_config", &ContinuousAutoTuner::setConfig, nb::arg("config"), "Set configuration")
        .def_prop_rw("enabled", &ContinuousAutoTuner::isEnabled, &ContinuousAutoTuner::setEnabled,
                     "Enable/disable continuous tuning")

        // Kernel registration
        .def("register_kernel",
             nb::overload_cast<const std::string&, KernelVariant>(
                 &ContinuousAutoTuner::registerKernel),
             nb::arg("name"), nb::arg("initial_variant"), "Register a kernel with initial variant")

        .def("register_kernel_variants",
             nb::overload_cast<const std::string&, std::vector<KernelVariant>>(
                 &ContinuousAutoTuner::registerKernel),
             nb::arg("name"), nb::arg("variants"), "Register a kernel with multiple variants")

        .def("add_variant", &ContinuousAutoTuner::addVariant, nb::arg("kernel_name"),
             nb::arg("variant"), "Add a new variant to an existing kernel")

        .def("has_kernel", &ContinuousAutoTuner::hasKernel, nb::arg("name"),
             "Check if kernel is registered")

        .def("num_variants", &ContinuousAutoTuner::numVariants, nb::arg("kernel_name"),
             "Get number of variants for a kernel")

        // Variant selection
        .def("select_variant",
             nb::overload_cast<const std::string&>(&ContinuousAutoTuner::selectVariant),
             nb::arg("kernel_name"), "Select best variant using Thompson Sampling")

        .def("select_variant_for_size",
             nb::overload_cast<const std::string&, size_t>(&ContinuousAutoTuner::selectVariant),
             nb::arg("kernel_name"), nb::arg("input_size"),
             "Select variant for specific input size")

        .def("get_variant", &ContinuousAutoTuner::getVariant, nb::arg("kernel_name"),
             nb::arg("variant_idx"), nb::rv_policy::reference, "Get variant by index")

        .def("get_best_variant", &ContinuousAutoTuner::getBestVariant, nb::arg("kernel_name"),
             nb::rv_policy::reference, "Get current best variant for a kernel")

        // Execution recording
        .def("record_execution",
             nb::overload_cast<const std::string&, size_t, float>(
                 &ContinuousAutoTuner::recordExecution),
             nb::arg("kernel_name"), nb::arg("variant_idx"), nb::arg("time_ns"),
             "Record execution time for a variant")

        .def("record_execution_with_size",
             nb::overload_cast<const std::string&, size_t, size_t, float>(
                 &ContinuousAutoTuner::recordExecution),
             nb::arg("kernel_name"), nb::arg("variant_idx"), nb::arg("input_size"),
             nb::arg("time_ns"), "Record execution with input size")

        .def("record_execution_batch", &ContinuousAutoTuner::recordExecutionBatch,
             nb::arg("kernel_name"), nb::arg("variant_indices"), nb::arg("times_ns"),
             "Batch record multiple executions")

        // Cost model integration
        .def("set_cost_model", &ContinuousAutoTuner::setCostModel, nb::arg("model"),
             "Set cost model for performance prediction")

        .def("sync_cost_model", &ContinuousAutoTuner::syncCostModel,
             "Update cost model with recorded observations")

        // Retuning
        .def("needs_retune", &ContinuousAutoTuner::needsRetune, nb::arg("kernel_name"),
             "Check if kernel needs retuning")

        .def("trigger_retune", &ContinuousAutoTuner::triggerRetune, nb::arg("kernel_name"),
             "Force retune for a kernel")

        // Statistics
        .def("statistics", &ContinuousAutoTuner::statistics, "Get overall statistics")

        .def("get_kernel_profile", &ContinuousAutoTuner::getKernelProfile, nb::arg("name"),
             "Get kernel-specific profile")

        .def("recent_events", &ContinuousAutoTuner::recentEvents, nb::arg("count") = 100,
             "Get recent tuning events")

        .def("reset_statistics", &ContinuousAutoTuner::resetStatistics, "Reset all statistics")

        .def("reset_kernel", &ContinuousAutoTuner::resetKernel, nb::arg("kernel_name"),
             "Reset specific kernel profile")

        // Persistence
        .def("serialize", &ContinuousAutoTuner::serialize, "Serialize tuner state to string")

        .def("deserialize", &ContinuousAutoTuner::deserialize, nb::arg("data"),
             "Deserialize tuner state from string")

        .def("save", &ContinuousAutoTuner::save, nb::arg("path"), "Save state to file")

        .def("load", &ContinuousAutoTuner::load, nb::arg("path"), "Load state from file")

        // Warm start
        .def("import_history", &ContinuousAutoTuner::importHistory, nb::arg("kernel_name"),
             nb::arg("variant_times"), "Import historical execution data for warm start")

        .def("export_history", &ContinuousAutoTuner::exportHistory, nb::arg("kernel_name"),
             "Export execution history");

    // =========================================================================
    // Module-level functions
    // =========================================================================
    m.def("get_global_tuner", &getGlobalTuner, nb::rv_policy::reference,
          "Get the global continuous tuner instance");

    m.def("configure_global_tuner", &configureGlobalTuner, nb::arg("config"),
          "Set the global continuous tuner config");

    // =========================================================================
    // Adaptive JIT Executor Bindings
    // =========================================================================
    using namespace bud::jit;

    // ExecutionTier enum
    nb::enum_<ExecutionTier>(m, "ExecutionTier", "Execution tier for adaptive JIT")
        .value("Interpreter", ExecutionTier::kInterpreter, "Tier 0: Direct Highway dispatch")
        .value("CopyPatch", ExecutionTier::kCopyPatch, "Tier 1: Copy-and-patch JIT")
        .value("FusedKernel", ExecutionTier::kFusedKernel, "Tier 2: Fused Highway kernels");

    // ExecutionResult struct
    nb::class_<ExecutionResult>(m, "ExecutionResult", "Result of an adaptive execution")
        .def(nb::init<>())
        .def_ro("success", &ExecutionResult::success, "Whether execution succeeded")
        .def_ro("execution_time_ns", &ExecutionResult::execution_time_ns,
                "Execution time in nanoseconds")
        .def_ro("tier_used", &ExecutionResult::tier_used, "Which tier was used for execution")
        .def_ro("version_idx", &ExecutionResult::version_idx, "Index of version used")
        .def_ro("was_promoted", &ExecutionResult::was_promoted, "Whether a tier promotion occurred")
        .def_ro("error_message", &ExecutionResult::error_message, "Error message if failed");

    // AdaptiveCacheConfig struct
    nb::class_<AdaptiveCacheConfig>(m, "AdaptiveCacheConfig",
                                    "Configuration for adaptive kernel cache")
        .def(nb::init<>())
        .def_rw("max_versions_per_context", &AdaptiveCacheConfig::max_versions_per_context,
                "Maximum versions per context (default: 8)")
        .def_rw("max_context_entries", &AdaptiveCacheConfig::max_context_entries,
                "Maximum context entries (default: 10000)")
        .def_rw("prior_alpha", &AdaptiveCacheConfig::prior_alpha,
                "Thompson prior alpha (default: 1.0)")
        .def_rw("prior_beta", &AdaptiveCacheConfig::prior_beta,
                "Thompson prior beta (default: 1.0)")
        .def_rw("exploration_bonus", &AdaptiveCacheConfig::exploration_bonus,
                "Exploration bonus (default: 0.1)")
        .def_rw("warmup_executions", &AdaptiveCacheConfig::warmup_executions,
                "Warmup executions (default: 10)")
        .def_rw("variance_threshold", &AdaptiveCacheConfig::variance_threshold,
                "Variance threshold for promotion (default: 0.1)")
        .def_rw("min_speedup_ratio", &AdaptiveCacheConfig::min_speedup_ratio,
                "Minimum speedup ratio for promotion (default: 1.2)")
        .def_rw("min_calls_tier1", &AdaptiveCacheConfig::min_calls_tier1,
                "Minimum calls for tier 0->1 (default: 10)")
        .def_rw("min_calls_tier2", &AdaptiveCacheConfig::min_calls_tier2,
                "Minimum calls for tier 1->2 (default: 100)");

    // AdaptiveExecutorConfig struct
    nb::class_<AdaptiveExecutorConfig>(m, "AdaptiveExecutorConfig",
                                       "Configuration for adaptive executor")
        .def(nb::init<>())
        .def_rw("cache_config", &AdaptiveExecutorConfig::cache_config, "Cache configuration")
        .def_rw("enable_timing", &AdaptiveExecutorConfig::enable_timing,
                "Record execution times (default: true)")
        .def_rw("enable_promotion", &AdaptiveExecutorConfig::enable_promotion,
                "Enable tier promotion (default: true)")
        .def_rw("enable_specialization", &AdaptiveExecutorConfig::enable_specialization,
                "Enable size specialization (default: true)")
        .def_rw("small_array_threshold", &AdaptiveExecutorConfig::small_array_threshold,
                "Arrays smaller than this bypass adaptive overhead (default: 10000)")
        .def_rw("enable_persistence", &AdaptiveExecutorConfig::enable_persistence,
                "Enable profile persistence (default: true)")
        .def_rw("profile_dir", &AdaptiveExecutorConfig::profile_dir,
                "Profile directory (default: ~/.bud_flow/profiles)")
        .def_rw("verbose", &AdaptiveExecutorConfig::verbose,
                "Enable verbose logging (default: false)");

    // AdaptiveKernelCache::CacheStats struct
    nb::class_<AdaptiveKernelCache::CacheStats>(m, "AdaptiveCacheStats", "Cache statistics")
        .def_ro("total_contexts", &AdaptiveKernelCache::CacheStats::total_contexts)
        .def_ro("total_versions", &AdaptiveKernelCache::CacheStats::total_versions)
        .def_ro("total_selections", &AdaptiveKernelCache::CacheStats::total_selections)
        .def_ro("total_executions", &AdaptiveKernelCache::CacheStats::total_executions)
        .def_ro("tier0_contexts", &AdaptiveKernelCache::CacheStats::tier0_contexts)
        .def_ro("tier1_contexts", &AdaptiveKernelCache::CacheStats::tier1_contexts)
        .def_ro("tier2_contexts", &AdaptiveKernelCache::CacheStats::tier2_contexts)
        .def_ro("specialized_versions", &AdaptiveKernelCache::CacheStats::specialized_versions);

    // AdaptiveExecutor::ExecutorStats struct
    nb::class_<AdaptiveExecutor::ExecutorStats>(m, "AdaptiveExecutorStats", "Executor statistics")
        .def_ro("total_executions", &AdaptiveExecutor::ExecutorStats::total_executions)
        .def_ro("tier0_executions", &AdaptiveExecutor::ExecutorStats::tier0_executions)
        .def_ro("tier1_executions", &AdaptiveExecutor::ExecutorStats::tier1_executions)
        .def_ro("tier2_executions", &AdaptiveExecutor::ExecutorStats::tier2_executions)
        .def_ro("fast_path_executions", &AdaptiveExecutor::ExecutorStats::fast_path_executions,
                "Executions that bypassed adaptive overhead (small arrays)")
        .def_ro("total_execution_time_ns",
                &AdaptiveExecutor::ExecutorStats::total_execution_time_ns)
        .def_ro("avg_execution_time_ns", &AdaptiveExecutor::ExecutorStats::avg_execution_time_ns)
        .def_ro("promotions_performed", &AdaptiveExecutor::ExecutorStats::promotions_performed)
        .def_ro("specializations_performed",
                &AdaptiveExecutor::ExecutorStats::specializations_performed)
        .def_ro("cache_stats", &AdaptiveExecutor::ExecutorStats::cache_stats);

    // AdaptiveExecutor class
    nb::class_<AdaptiveExecutor>(m, "AdaptiveExecutor",
                                 "Adaptive JIT executor with Thompson Sampling.\n\n"
                                 "Learns optimal kernel variants at runtime and\n"
                                 "persists optimizations across sessions.\n\n"
                                 "Example:\n"
                                 "  executor = get_adaptive_executor()\n"
                                 "  executor.enabled = True\n"
                                 "  stats = executor.statistics()\n"
                                 "  print(f'Promotions: {stats.promotions_performed}')\n")

        // Constructors
        .def(nb::init<>(), "Create with default configuration")
        .def(nb::init<const AdaptiveExecutorConfig&>(), nb::arg("config"),
             "Create with custom configuration")

        // Configuration
        .def_prop_ro("config", &AdaptiveExecutor::config, nb::rv_policy::reference,
                     "Get current configuration")
        .def("set_config", &AdaptiveExecutor::setConfig, nb::arg("config"), "Set configuration")
        .def_prop_rw("enabled", &AdaptiveExecutor::isEnabled, &AdaptiveExecutor::setEnabled,
                     "Enable/disable adaptive optimization")
        .def_prop_rw("small_array_threshold", &AdaptiveExecutor::smallArrayThreshold,
                     &AdaptiveExecutor::setSmallArrayThreshold,
                     "Arrays smaller than this bypass adaptive overhead (default: 10000)")

        // Statistics
        .def("statistics", &AdaptiveExecutor::statistics, "Get executor statistics")
        .def("reset", &AdaptiveExecutor::reset, "Reset all statistics and cache")

        // Profile persistence
        .def("save_profiles", &AdaptiveExecutor::saveProfiles, "Save profiles to disk")
        .def("load_profiles", &AdaptiveExecutor::loadProfiles, "Load profiles from disk")
        .def("set_profile_dir", &AdaptiveExecutor::setProfileDir, nb::arg("dir"),
             "Set profile directory");

    // Module-level functions for adaptive execution
    m.def("get_adaptive_executor", &getGlobalExecutor, nb::rv_policy::reference,
          "Get the global adaptive executor instance");

    m.def("configure_adaptive_executor", &configureGlobalExecutor, nb::arg("config"),
          "Configure the global adaptive executor");

    m.def("initialize_adaptive_execution", &initializeAdaptiveExecution,
          "Initialize adaptive execution (loads profiles)");

    m.def("shutdown_adaptive_execution", &shutdownAdaptiveExecution,
          "Shutdown adaptive execution (saves profiles)");

    // =========================================================================
    // TieredExecutor Adaptive Mode Integration
    // =========================================================================
    // These functions control adaptive mode for the global TieredExecutor,
    // enabling Thompson Sampling-based tier selection for all Bunch operations

    m.def("set_adaptive_execution_enabled", &bud::setAdaptiveExecutionEnabled, nb::arg("enabled"),
          "Enable/disable adaptive execution with Thompson Sampling.\n\n"
          "When enabled, all Bunch operations use Bayesian multi-armed bandit\n"
          "for tier selection instead of static call-count thresholds.\n\n"
          "Args:\n"
          "    enabled: True to enable, False to disable\n\n"
          "Example:\n"
          "    bud.set_adaptive_execution_enabled(True)\n"
          "    # Run operations - they will be profiled and optimized\n"
          "    for _ in range(100):\n"
          "        c = a + b\n"
          "    stats = bud.get_adaptive_executor_stats()\n"
          "    print(f'Tier 0: {stats.tier0_executions}')\n"
          "    print(f'Tier 1: {stats.tier1_executions}')\n");

    m.def("is_adaptive_execution_enabled", &bud::isAdaptiveExecutionEnabled,
          "Check if adaptive execution is enabled.\n\n"
          "Returns:\n"
          "    bool: True if adaptive execution is enabled");

    // AdaptiveExecutorStats binding (from bud_flow_lang.h)
    nb::class_<bud::AdaptiveExecutorStats>(
        m, "AdaptiveStats", "Statistics for adaptive execution (from TieredExecutor)")
        .def_ro("total_executions", &bud::AdaptiveExecutorStats::total_executions,
                "Total number of executions")
        .def_ro("tier0_executions", &bud::AdaptiveExecutorStats::tier0_executions,
                "Executions at tier 0 (Interpreter)")
        .def_ro("tier1_executions", &bud::AdaptiveExecutorStats::tier1_executions,
                "Executions at tier 1 (CopyPatch)")
        .def_ro("tier2_executions", &bud::AdaptiveExecutorStats::tier2_executions,
                "Executions at tier 2 (FusedKernel)")
        .def_ro("fast_path_executions", &bud::AdaptiveExecutorStats::fast_path_executions,
                "Executions bypassing adaptive overhead (small arrays)")
        .def_ro("promotions_performed", &bud::AdaptiveExecutorStats::promotions_performed,
                "Number of tier promotions")
        .def_ro("specializations_performed", &bud::AdaptiveExecutorStats::specializations_performed,
                "Number of size specializations")
        .def_ro("avg_execution_time_ns", &bud::AdaptiveExecutorStats::avg_execution_time_ns,
                "Average execution time in nanoseconds")
        .def_ro("total_contexts", &bud::AdaptiveExecutorStats::total_contexts,
                "Number of cached kernel contexts")
        .def_ro("total_versions", &bud::AdaptiveExecutorStats::total_versions,
                "Number of cached kernel versions");

    m.def("get_adaptive_executor_stats", &bud::getAdaptiveExecutorStats,
          "Get statistics for adaptive execution.\n\n"
          "Returns:\n"
          "    AdaptiveStats: Statistics including tier executions, promotions, etc.\n\n"
          "Example:\n"
          "    stats = bud.get_adaptive_executor_stats()\n"
          "    print(f'Total: {stats.total_executions}')\n"
          "    print(f'Promotions: {stats.promotions_performed}')\n");

    m.def("save_adaptive_profiles", &bud::saveAdaptiveProfiles,
          "Save adaptive execution profiles to disk.\n\n"
          "Profiles are saved to ~/.bud_flow/profiles/<hardware_id>.dat\n"
          "and include Thompson Sampling states for warm-start optimization.\n\n"
          "Returns:\n"
          "    bool: True if save succeeded");

    m.def("load_adaptive_profiles", &bud::loadAdaptiveProfiles,
          "Load adaptive execution profiles from disk.\n\n"
          "Loads previously saved profiles for warm-start optimization.\n\n"
          "Returns:\n"
          "    bool: True if load succeeded");

    m.def("set_small_array_threshold", &bud::setSmallArrayThreshold, nb::arg("threshold"),
          "Set the small array threshold for fast path optimization.\n\n"
          "Arrays smaller than this threshold bypass adaptive overhead\n"
          "(Thompson Sampling, timing, context lookup) and use direct\n"
          "Highway dispatch for minimal latency.\n\n"
          "Args:\n"
          "    threshold: Elements count. Set to 0 to disable fast path.\n"
          "               Default is 10000 elements.\n\n"
          "Example:\n"
          "    # Use fast path for arrays < 5000 elements\n"
          "    bud.set_small_array_threshold(5000)\n"
          "    \n"
          "    # Disable fast path (always use adaptive)\n"
          "    bud.set_small_array_threshold(0)\n");

    m.def("get_small_array_threshold", &bud::getSmallArrayThreshold,
          "Get the current small array threshold.\n\n"
          "Returns:\n"
          "    int: Current threshold in elements");
}
