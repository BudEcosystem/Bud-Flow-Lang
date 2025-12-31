#pragma once

// =============================================================================
// Bud Flow Lang - Cost Model
// =============================================================================
//
// Machine learning-based cost model for predicting schedule performance.
// Uses gradient boosted trees for fast inference and online learning.
//
// Features:
// - Feature extraction from schedules
// - Runtime prediction (orders of magnitude faster than measurement)
// - Online learning from measured results
// - Model persistence for reuse across sessions
//
// Inspired by TVM's Ansor cost model (OSDI 2020).
//

#include "bud_flow_lang/scheduler/compute_dag.h"
#include "bud_flow_lang/scheduler/schedule.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace bud {
namespace scheduler {

// =============================================================================
// Cost Model Configuration
// =============================================================================

/// Configuration for the cost model
struct CostModelConfig {
    // Gradient boosted tree parameters
    size_t num_trees = 64;
    size_t max_depth = 5;
    float learning_rate = 0.1f;
    float subsample = 0.8f;
    float colsample = 0.8f;

    // Regularization
    float reg_lambda = 1.0f;
    float reg_alpha = 0.0f;

    // Training
    size_t max_samples = 10000;  // Max samples to keep in memory
    bool incremental_update = true;
};

// =============================================================================
// Feature Configuration
// =============================================================================

/// Configuration for feature extraction
struct FeatureConfig {
    bool use_loop_features = true;
    bool use_memory_features = true;
    bool use_compute_features = true;
    bool use_reuse_features = true;

    // Feature normalization
    bool normalize = true;
    float clip_min = -10.0f;
    float clip_max = 10.0f;
};

// =============================================================================
// CostModel
// =============================================================================

/// Machine learning cost model for schedule evaluation
class CostModel {
  public:
    CostModel();
    explicit CostModel(const CostModelConfig& config);
    ~CostModel();

    // Move-only (no copying)
    CostModel(CostModel&& other) noexcept;
    CostModel& operator=(CostModel&& other) noexcept;
    CostModel(const CostModel&) = delete;
    CostModel& operator=(const CostModel&) = delete;

    // -------------------------------------------------------------------------
    // Validity
    // -------------------------------------------------------------------------

    /// Check if model is valid
    [[nodiscard]] bool isValid() const { return valid_; }

    /// Get config
    [[nodiscard]] const CostModelConfig& config() const { return config_; }

    // -------------------------------------------------------------------------
    // Feature Extraction
    // -------------------------------------------------------------------------

    /// Extract features from a schedule
    [[nodiscard]] std::vector<float> extractFeatures(const Schedule& schedule) const;

    /// Extract features from a schedule with DAG context
    [[nodiscard]] std::vector<float> extractFeatures(const Schedule& schedule,
                                                     const ComputeDAG& dag) const;

    /// Set feature configuration
    void setFeatureConfig(const FeatureConfig& config);

    /// Get feature configuration
    [[nodiscard]] const FeatureConfig& featureConfig() const { return feature_config_; }

    // -------------------------------------------------------------------------
    // Prediction
    // -------------------------------------------------------------------------

    /// Predict runtime for a schedule
    [[nodiscard]] float predict(const Schedule& schedule) const;

    /// Predict runtime for a schedule with DAG context
    [[nodiscard]] float predict(const Schedule& schedule, const ComputeDAG& dag) const;

    /// Predict runtime for multiple schedules (batch)
    [[nodiscard]] std::vector<float> predictBatch(const std::vector<Schedule>& schedules) const;

    /// Predict runtime for multiple schedules with DAG context
    [[nodiscard]] std::vector<float> predictBatch(const std::vector<Schedule>& schedules,
                                                  const ComputeDAG& dag) const;

    // -------------------------------------------------------------------------
    // Online Learning
    // -------------------------------------------------------------------------

    /// Update model with a single measurement
    void update(const Schedule& schedule, float measured_time);

    /// Update model with a single measurement (with DAG context)
    void update(const Schedule& schedule, const ComputeDAG& dag, float measured_time);

    /// Update model with batch of measurements
    void updateBatch(const std::vector<Schedule>& schedules,
                     const std::vector<float>& measured_times);

    /// Update model with batch of measurements (with DAG context)
    void updateBatch(const std::vector<Schedule>& schedules, const ComputeDAG& dag,
                     const std::vector<float>& measured_times);

    // -------------------------------------------------------------------------
    // Model Statistics
    // -------------------------------------------------------------------------

    /// Get number of training samples
    [[nodiscard]] size_t numSamples() const { return num_samples_; }

    /// Get training error (RMSE)
    [[nodiscard]] float trainingError() const;

    /// Clear all training data
    void clear();

    // -------------------------------------------------------------------------
    // Ranking
    // -------------------------------------------------------------------------

    /// Rank schedules by predicted cost (returns sorted pairs of cost, index)
    [[nodiscard]] std::vector<std::pair<float, size_t>>
    rankSchedules(const std::vector<Schedule>& schedules) const;

    /// Select top-K schedules by predicted cost
    [[nodiscard]] std::vector<Schedule> selectTopK(const std::vector<Schedule>& schedules,
                                                   size_t k) const;

    // -------------------------------------------------------------------------
    // Serialization
    // -------------------------------------------------------------------------

    /// Serialize to string
    [[nodiscard]] std::string serialize() const;

    /// Deserialize from string
    bool deserialize(const std::string& data);

    /// Save to file
    bool save(const std::string& path) const;

    /// Load from file
    bool load(const std::string& path);

  private:
    CostModelConfig config_;
    FeatureConfig feature_config_;
    bool valid_ = true;
    size_t num_samples_ = 0;

    // Training data storage
    std::vector<std::vector<float>> training_features_;
    std::vector<float> training_targets_;

    // Simple model: linear regression with feature weights
    // (In a full implementation, this would be XGBoost or similar)
    std::vector<float> weights_;
    float bias_ = 0.0f;

    // Feature extraction helpers
    [[nodiscard]] std::vector<float> extractLoopFeatures(const Schedule& schedule) const;
    [[nodiscard]] std::vector<float> extractMemoryFeatures(const Schedule& schedule) const;
    [[nodiscard]] std::vector<float> extractComputeFeatures(const Schedule& schedule) const;
    [[nodiscard]] std::vector<float> extractReuseFeatures(const Schedule& schedule) const;
    [[nodiscard]] std::vector<float> extractDAGFeatures(const ComputeDAG& dag) const;

    // Model training
    void trainModel();
    void incrementalUpdate(const std::vector<float>& features, float target);

    // Prediction helpers
    [[nodiscard]] float predictFromFeatures(const std::vector<float>& features) const;
};

}  // namespace scheduler
}  // namespace bud
