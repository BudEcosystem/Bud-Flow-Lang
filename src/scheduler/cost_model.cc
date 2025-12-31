// =============================================================================
// Bud Flow Lang - Cost Model Implementation
// =============================================================================

#include "bud_flow_lang/scheduler/cost_model.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

namespace bud {
namespace scheduler {

// =============================================================================
// CostModel Implementation
// =============================================================================

CostModel::CostModel() : config_{}, feature_config_{} {
    // Initialize with default weights (simple heuristic model)
    weights_.resize(32, 0.0f);

    // Set initial heuristic weights
    // These get refined through online learning
    weights_[0] = 1.0f;    // Base cost
    weights_[1] = -0.1f;   // Split factor (larger = faster, less overhead)
    weights_[2] = -0.2f;   // Vectorization (reduces element ops)
    weights_[3] = -0.15f;  // Parallelization
    weights_[4] = -0.05f;  // Tiling (cache locality)
    weights_[5] = 0.1f;    // Unroll factor (may increase I-cache pressure)

    bias_ = 1.0f;
}

CostModel::CostModel(const CostModelConfig& config) : config_(config), feature_config_{} {
    weights_.resize(32, 0.0f);
    weights_[0] = 1.0f;
    weights_[1] = -0.1f;
    weights_[2] = -0.2f;
    weights_[3] = -0.15f;
    weights_[4] = -0.05f;
    weights_[5] = 0.1f;
    bias_ = 1.0f;
}

CostModel::~CostModel() = default;

CostModel::CostModel(CostModel&& other) noexcept = default;
CostModel& CostModel::operator=(CostModel&& other) noexcept = default;

// =============================================================================
// Feature Extraction
// =============================================================================

std::vector<float> CostModel::extractFeatures(const Schedule& schedule) const {
    std::vector<float> features;

    if (feature_config_.use_loop_features) {
        auto loop_feats = extractLoopFeatures(schedule);
        features.insert(features.end(), loop_feats.begin(), loop_feats.end());
    }

    if (feature_config_.use_memory_features) {
        auto mem_feats = extractMemoryFeatures(schedule);
        features.insert(features.end(), mem_feats.begin(), mem_feats.end());
    }

    if (feature_config_.use_compute_features) {
        auto compute_feats = extractComputeFeatures(schedule);
        features.insert(features.end(), compute_feats.begin(), compute_feats.end());
    }

    if (feature_config_.use_reuse_features) {
        auto reuse_feats = extractReuseFeatures(schedule);
        features.insert(features.end(), reuse_feats.begin(), reuse_feats.end());
    }

    // Ensure minimum feature vector size
    while (features.size() < weights_.size()) {
        features.push_back(0.0f);
    }

    // Normalize features if configured
    if (feature_config_.normalize) {
        for (auto& f : features) {
            f = std::clamp(f, feature_config_.clip_min, feature_config_.clip_max);
        }
    }

    return features;
}

std::vector<float> CostModel::extractFeatures(const Schedule& schedule,
                                              const ComputeDAG& dag) const {

    auto features = extractFeatures(schedule);

    // Add DAG features
    auto dag_feats = extractDAGFeatures(dag);
    features.insert(features.end(), dag_feats.begin(), dag_feats.end());

    return features;
}

std::vector<float> CostModel::extractLoopFeatures(const Schedule& schedule) const {
    std::vector<float> features;

    const auto& transforms = schedule.transforms();

    // Count transform types
    int split_count = 0;
    int tile_count = 0;
    int fuse_count = 0;
    int reorder_count = 0;
    float max_split_factor = 0;
    float max_tile_size = 0;

    for (const auto& t : transforms) {
        switch (t.type) {
        case TransformType::kSplit:
            split_count++;
            max_split_factor = std::max(max_split_factor, static_cast<float>(t.factor));
            break;
        case TransformType::kTile:
            tile_count++;
            max_tile_size =
                std::max(max_tile_size, static_cast<float>(std::max(t.tile_x, t.tile_y)));
            break;
        case TransformType::kFuse:
            fuse_count++;
            break;
        case TransformType::kReorder:
            reorder_count++;
            break;
        default:
            break;
        }
    }

    features.push_back(static_cast<float>(split_count));
    features.push_back(static_cast<float>(tile_count));
    features.push_back(static_cast<float>(fuse_count));
    features.push_back(static_cast<float>(reorder_count));
    features.push_back(std::log2(max_split_factor + 1));  // Log scale
    features.push_back(std::log2(max_tile_size + 1));
    features.push_back(static_cast<float>(transforms.size()));

    return features;
}

std::vector<float> CostModel::extractMemoryFeatures(const Schedule& schedule) const {
    std::vector<float> features;

    const auto& transforms = schedule.transforms();

    // Memory-related features
    int cache_read_count = 0;
    int cache_write_count = 0;
    bool has_local_cache = false;
    bool has_shared_cache = false;

    for (const auto& t : transforms) {
        if (t.type == TransformType::kCacheRead) {
            cache_read_count++;
            if (t.cache_type == CacheType::kLocal)
                has_local_cache = true;
            if (t.cache_type == CacheType::kShared)
                has_shared_cache = true;
        }
        if (t.type == TransformType::kCacheWrite) {
            cache_write_count++;
        }
    }

    features.push_back(static_cast<float>(cache_read_count));
    features.push_back(static_cast<float>(cache_write_count));
    features.push_back(has_local_cache ? 1.0f : 0.0f);
    features.push_back(has_shared_cache ? 1.0f : 0.0f);

    return features;
}

std::vector<float> CostModel::extractComputeFeatures(const Schedule& schedule) const {
    std::vector<float> features;

    const auto& transforms = schedule.transforms();

    // Compute-related features
    int vectorize_count = 0;
    int parallel_count = 0;
    int unroll_count = 0;
    float max_vector_width = 0;
    float total_unroll_factor = 1.0f;

    for (const auto& t : transforms) {
        switch (t.type) {
        case TransformType::kVectorize:
            vectorize_count++;
            max_vector_width = std::max(max_vector_width, static_cast<float>(t.vector_width));
            break;
        case TransformType::kParallel:
            parallel_count++;
            break;
        case TransformType::kUnroll:
            unroll_count++;
            if (t.factor > 0) {
                total_unroll_factor *= static_cast<float>(t.factor);
            }
            break;
        default:
            break;
        }
    }

    features.push_back(static_cast<float>(vectorize_count));
    features.push_back(static_cast<float>(parallel_count));
    features.push_back(static_cast<float>(unroll_count));
    features.push_back(std::log2(max_vector_width + 1));
    features.push_back(std::log2(total_unroll_factor));

    return features;
}

std::vector<float> CostModel::extractReuseFeatures(const Schedule& schedule) const {
    std::vector<float> features;

    const auto& transforms = schedule.transforms();

    // Reuse-related features
    int compute_at_count = 0;
    int compute_inline_count = 0;

    for (const auto& t : transforms) {
        if (t.type == TransformType::kComputeAt) {
            compute_at_count++;
        }
        if (t.type == TransformType::kComputeInline) {
            compute_inline_count++;
        }
    }

    features.push_back(static_cast<float>(compute_at_count));
    features.push_back(static_cast<float>(compute_inline_count));

    // Estimate data reuse based on tiling
    float estimated_reuse = 0.0f;
    for (const auto& t : transforms) {
        if (t.type == TransformType::kTile) {
            estimated_reuse += std::log2(static_cast<float>(t.tile_x * t.tile_y));
        }
    }
    features.push_back(estimated_reuse);

    return features;
}

std::vector<float> CostModel::extractDAGFeatures(const ComputeDAG& dag) const {
    std::vector<float> features;

    auto analysis = dag.analyze();

    features.push_back(static_cast<float>(dag.numNodes()));
    features.push_back(static_cast<float>(dag.numEdges()));
    features.push_back(static_cast<float>(analysis.depth));
    features.push_back(static_cast<float>(analysis.width));
    features.push_back(std::log2(static_cast<float>(analysis.flops + 1)));
    features.push_back(analysis.has_fusion_opportunity ? 1.0f : 0.0f);
    features.push_back(analysis.has_parallel_opportunity ? 1.0f : 0.0f);
    features.push_back(analysis.has_vectorization_opportunity ? 1.0f : 0.0f);

    return features;
}

void CostModel::setFeatureConfig(const FeatureConfig& config) {
    feature_config_ = config;
}

// =============================================================================
// Prediction
// =============================================================================

float CostModel::predict(const Schedule& schedule) const {
    auto features = extractFeatures(schedule);
    return predictFromFeatures(features);
}

float CostModel::predict(const Schedule& schedule, const ComputeDAG& dag) const {
    auto features = extractFeatures(schedule, dag);
    return predictFromFeatures(features);
}

std::vector<float> CostModel::predictBatch(const std::vector<Schedule>& schedules) const {
    std::vector<float> predictions;
    predictions.reserve(schedules.size());
    for (const auto& schedule : schedules) {
        predictions.push_back(predict(schedule));
    }
    return predictions;
}

std::vector<float> CostModel::predictBatch(const std::vector<Schedule>& schedules,
                                           const ComputeDAG& dag) const {
    std::vector<float> predictions;
    predictions.reserve(schedules.size());
    for (const auto& schedule : schedules) {
        predictions.push_back(predict(schedule, dag));
    }
    return predictions;
}

float CostModel::predictFromFeatures(const std::vector<float>& features) const {
    // Simple linear model: y = sum(w_i * x_i) + b
    float prediction = bias_;
    size_t n = std::min(features.size(), weights_.size());
    for (size_t i = 0; i < n; ++i) {
        prediction += weights_[i] * features[i];
    }

    // Ensure non-negative prediction
    return std::max(0.0f, prediction);
}

// =============================================================================
// Online Learning
// =============================================================================

void CostModel::update(const Schedule& schedule, float measured_time) {
    auto features = extractFeatures(schedule);

    // Store training data
    training_features_.push_back(features);
    training_targets_.push_back(measured_time);
    num_samples_++;

    // Limit training data size
    if (training_features_.size() > config_.max_samples) {
        training_features_.erase(training_features_.begin());
        training_targets_.erase(training_targets_.begin());
    }

    // Incremental update
    if (config_.incremental_update) {
        incrementalUpdate(features, measured_time);
    } else {
        // Full retrain
        trainModel();
    }
}

void CostModel::update(const Schedule& schedule, const ComputeDAG& dag, float measured_time) {
    auto features = extractFeatures(schedule, dag);

    training_features_.push_back(features);
    training_targets_.push_back(measured_time);
    num_samples_++;

    if (training_features_.size() > config_.max_samples) {
        training_features_.erase(training_features_.begin());
        training_targets_.erase(training_targets_.begin());
    }

    if (config_.incremental_update) {
        incrementalUpdate(features, measured_time);
    } else {
        trainModel();
    }
}

void CostModel::updateBatch(const std::vector<Schedule>& schedules,
                            const std::vector<float>& measured_times) {
    for (size_t i = 0; i < schedules.size(); ++i) {
        auto features = extractFeatures(schedules[i]);
        training_features_.push_back(features);
        training_targets_.push_back(measured_times[i]);
        num_samples_++;
    }

    // Limit size
    while (training_features_.size() > config_.max_samples) {
        training_features_.erase(training_features_.begin());
        training_targets_.erase(training_targets_.begin());
    }

    // Full retrain for batch updates
    trainModel();
}

void CostModel::updateBatch(const std::vector<Schedule>& schedules, const ComputeDAG& dag,
                            const std::vector<float>& measured_times) {
    for (size_t i = 0; i < schedules.size(); ++i) {
        auto features = extractFeatures(schedules[i], dag);
        training_features_.push_back(features);
        training_targets_.push_back(measured_times[i]);
        num_samples_++;
    }

    while (training_features_.size() > config_.max_samples) {
        training_features_.erase(training_features_.begin());
        training_targets_.erase(training_targets_.begin());
    }

    trainModel();
}

void CostModel::trainModel() {
    if (training_features_.empty()) {
        return;
    }

    // Simple gradient descent for linear regression
    // (In production, use XGBoost or similar)
    size_t n = training_features_.size();
    size_t num_features = weights_.size();

    // Ensure all feature vectors have same size
    for (auto& f : training_features_) {
        while (f.size() < num_features) {
            f.push_back(0.0f);
        }
    }

    // Multiple epochs of gradient descent
    float lr = config_.learning_rate;
    for (size_t epoch = 0; epoch < 10; ++epoch) {
        std::vector<float> weight_gradients(num_features, 0.0f);
        float bias_gradient = 0.0f;

        // Compute gradients
        for (size_t i = 0; i < n; ++i) {
            float pred = predictFromFeatures(training_features_[i]);
            float error = pred - training_targets_[i];

            bias_gradient += error;
            for (size_t j = 0; j < num_features; ++j) {
                weight_gradients[j] += error * training_features_[i][j];
            }
        }

        // Update weights
        bias_ -= lr * (bias_gradient / static_cast<float>(n));
        for (size_t j = 0; j < num_features; ++j) {
            weights_[j] -= lr * (weight_gradients[j] / static_cast<float>(n) +
                                 config_.reg_lambda * weights_[j]);
        }
    }
}

void CostModel::incrementalUpdate(const std::vector<float>& features, float target) {
    // Online gradient descent update
    float pred = predictFromFeatures(features);
    float error = pred - target;
    float lr = config_.learning_rate;

    size_t n = std::min(features.size(), weights_.size());
    for (size_t i = 0; i < n; ++i) {
        weights_[i] -= lr * error * features[i];
    }
    bias_ -= lr * error;
}

// =============================================================================
// Model Statistics
// =============================================================================

float CostModel::trainingError() const {
    if (training_features_.empty()) {
        return 0.0f;
    }

    float mse = 0.0f;
    for (size_t i = 0; i < training_features_.size(); ++i) {
        float pred = predictFromFeatures(training_features_[i]);
        float error = pred - training_targets_[i];
        mse += error * error;
    }

    return std::sqrt(mse / static_cast<float>(training_features_.size()));
}

void CostModel::clear() {
    training_features_.clear();
    training_targets_.clear();
    num_samples_ = 0;

    // Reset to initial weights
    std::fill(weights_.begin(), weights_.end(), 0.0f);
    weights_[0] = 1.0f;
    weights_[1] = -0.1f;
    weights_[2] = -0.2f;
    weights_[3] = -0.15f;
    weights_[4] = -0.05f;
    weights_[5] = 0.1f;
    bias_ = 1.0f;
}

// =============================================================================
// Ranking
// =============================================================================

std::vector<std::pair<float, size_t>>
CostModel::rankSchedules(const std::vector<Schedule>& schedules) const {

    std::vector<std::pair<float, size_t>> ranked;
    ranked.reserve(schedules.size());

    for (size_t i = 0; i < schedules.size(); ++i) {
        float cost = predict(schedules[i]);
        ranked.emplace_back(cost, i);
    }

    std::sort(ranked.begin(), ranked.end());
    return ranked;
}

std::vector<Schedule> CostModel::selectTopK(const std::vector<Schedule>& schedules,
                                            size_t k) const {

    auto ranked = rankSchedules(schedules);

    std::vector<Schedule> top_k;
    k = std::min(k, ranked.size());
    top_k.reserve(k);

    for (size_t i = 0; i < k; ++i) {
        top_k.push_back(schedules[ranked[i].second]);
    }

    return top_k;
}

// =============================================================================
// Serialization
// =============================================================================

std::string CostModel::serialize() const {
    nlohmann::json j;

    // Config
    nlohmann::json cj;
    cj["num_trees"] = config_.num_trees;
    cj["max_depth"] = config_.max_depth;
    cj["learning_rate"] = config_.learning_rate;
    cj["subsample"] = config_.subsample;
    cj["colsample"] = config_.colsample;
    cj["reg_lambda"] = config_.reg_lambda;
    cj["reg_alpha"] = config_.reg_alpha;
    cj["max_samples"] = config_.max_samples;
    cj["incremental_update"] = config_.incremental_update;
    j["config"] = cj;

    // Feature config
    nlohmann::json fj;
    fj["use_loop_features"] = feature_config_.use_loop_features;
    fj["use_memory_features"] = feature_config_.use_memory_features;
    fj["use_compute_features"] = feature_config_.use_compute_features;
    fj["use_reuse_features"] = feature_config_.use_reuse_features;
    fj["normalize"] = feature_config_.normalize;
    fj["clip_min"] = feature_config_.clip_min;
    fj["clip_max"] = feature_config_.clip_max;
    j["feature_config"] = fj;

    // Model state
    j["weights"] = weights_;
    j["bias"] = bias_;
    j["num_samples"] = num_samples_;

    // Training data
    j["training_features"] = training_features_;
    j["training_targets"] = training_targets_;

    return j.dump();
}

bool CostModel::deserialize(const std::string& data) {
    try {
        auto j = nlohmann::json::parse(data);

        // Config
        auto cj = j["config"];
        config_.num_trees = cj["num_trees"].get<size_t>();
        config_.max_depth = cj["max_depth"].get<size_t>();
        config_.learning_rate = cj["learning_rate"].get<float>();
        config_.subsample = cj["subsample"].get<float>();
        config_.colsample = cj["colsample"].get<float>();
        config_.reg_lambda = cj["reg_lambda"].get<float>();
        config_.reg_alpha = cj["reg_alpha"].get<float>();
        config_.max_samples = cj["max_samples"].get<size_t>();
        config_.incremental_update = cj["incremental_update"].get<bool>();

        // Feature config
        auto fj = j["feature_config"];
        feature_config_.use_loop_features = fj["use_loop_features"].get<bool>();
        feature_config_.use_memory_features = fj["use_memory_features"].get<bool>();
        feature_config_.use_compute_features = fj["use_compute_features"].get<bool>();
        feature_config_.use_reuse_features = fj["use_reuse_features"].get<bool>();
        feature_config_.normalize = fj["normalize"].get<bool>();
        feature_config_.clip_min = fj["clip_min"].get<float>();
        feature_config_.clip_max = fj["clip_max"].get<float>();

        // Model state
        weights_ = j["weights"].get<std::vector<float>>();
        bias_ = j["bias"].get<float>();
        num_samples_ = j["num_samples"].get<size_t>();

        // Training data
        training_features_ = j["training_features"].get<std::vector<std::vector<float>>>();
        training_targets_ = j["training_targets"].get<std::vector<float>>();

        valid_ = true;
        return true;
    } catch (const std::exception& e) {
        spdlog::warn("Failed to deserialize CostModel: {}", e.what());
        return false;
    }
}

bool CostModel::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        spdlog::warn("Failed to open file for saving: {}", path);
        return false;
    }

    std::string data = serialize();
    file.write(data.data(), static_cast<std::streamsize>(data.size()));
    return file.good();
}

bool CostModel::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        spdlog::warn("Failed to open file for loading: {}", path);
        return false;
    }

    std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return deserialize(data);
}

}  // namespace scheduler
}  // namespace bud
