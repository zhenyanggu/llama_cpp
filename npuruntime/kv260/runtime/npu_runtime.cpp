#include "npu_runtime.h"
#include "../driver/npu_kv260_uapi.h"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <fcntl.h>
#include <cstring>
#include <chrono>
#include <cctype>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

// NOTE: Temporary compatibility behavior:
// - MVIN precision is forced to 1 regardless of API input.
// - MVOUT precision follows API input.
// API behavior is adjusted per current hardware contract.

// ==========================================
// Profiling
// ==========================================
#ifdef NPU_PROFILE
#define NPU_PROFILE_LOG(fmt, ...) \
    fprintf(stderr, "[NPU_PROFILE] " fmt "\n", ##__VA_ARGS__)

class ScopedLogTimer {
public:
    explicit ScopedLogTimer(const char* name)
        : name_(name), start_(std::chrono::steady_clock::now()) {}
    ~ScopedLogTimer() {
        auto end = std::chrono::steady_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
        NPU_PROFILE_LOG("%s: %lld ns", name_, (long long)ns);
    }

private:
    const char* name_;
    std::chrono::steady_clock::time_point start_;
};
#else
#define NPU_PROFILE_LOG(fmt, ...) do {} while(0)
#endif

#define NPU_TIMER_TOTAL(func_name) ((void)0)
#ifdef NPU_PROFILE
#define NPU_TIMER_SECTION_BEGIN(name) { ScopedLogTimer _sec_timer(name);
#define NPU_TIMER_SECTION_END() }
#else
#define NPU_TIMER_SECTION_BEGIN(name)
#define NPU_TIMER_SECTION_END()
#endif

namespace {

enum class ProfileStage {
    DmaIn,
    Compute,
    DmaOut,
    Layout,
    WaitIrq
};

enum class ProfileCallCounter {
    None,
    Mvin,
    Compute,
    Mvout,
    Layout
};

struct LayerProfileRecord {
    uint64_t invocations = 0;
    uint64_t totalNs = 0;
    uint64_t dmaInNs = 0;
    uint64_t computeNs = 0;
    uint64_t dmaOutNs = 0;
    uint64_t layoutNs = 0;
    uint64_t waitIrqNs = 0;
    uint64_t mvinCalls = 0;
    uint64_t computeCalls = 0;
    uint64_t mvoutCalls = 0;
    uint64_t layoutCalls = 0;
};

struct LayerManifestRecord {
    std::optional<int64_t> layerId;
    std::string layerName;
    std::string originOpType;
    std::string device;
    std::optional<std::string> fallbackReason;
    std::vector<std::string> fusedOps;
};

struct ProfilerState {
    bool initialized = false;
    bool registeredAtExit = false;
    std::string outputPath;
    std::map<int64_t, LayerProfileRecord> layers;
    std::mutex mutex;
};

struct ActiveLayerFrame {
    bool active = false;
    int64_t layerId = -1;
    std::chrono::steady_clock::time_point start;
};

thread_local ActiveLayerFrame g_activeLayer;

struct NpuLastOpContext {
    const char * op = "none";
    uint64_t reg0 = 0;
    uint64_t reg1 = 0;
    uint64_t cfg0 = 0;
    uint64_t cfg1 = 0;
    uint32_t start_bit = 0;
};

thread_local NpuLastOpContext g_last_op_ctx;

static bool abort_on_irq_timeout() {
    const char * env = std::getenv("NPU_ABORT_ON_IRQ_TIMEOUT");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    return std::strcmp(env, "0") != 0;
}

static std::string jsonEscape(const std::string& input) {
    std::ostringstream escaped;
    for (char c : input) {
        switch (c) {
        case '\\':
            escaped << "\\\\";
            break;
        case '"':
            escaped << "\\\"";
            break;
        case '\n':
            escaped << "\\n";
            break;
        case '\r':
            escaped << "\\r";
            break;
        case '\t':
            escaped << "\\t";
            break;
        default:
            escaped << c;
            break;
        }
    }
    return escaped.str();
}

static bool isProfilingDisabled() {
    const char* env = std::getenv("NPU_PROFILE_DISABLE");
    return env && std::strcmp(env, "1") == 0;
}

static std::string resolveProfileOutputPath(const char* overridePath) {
    if (overridePath && overridePath[0] != '\0') {
        return std::string(overridePath);
    }
    const char* envPath = std::getenv("NPU_PROFILE_OUT");
    if (envPath && envPath[0] != '\0') {
        return std::string(envPath);
    }
    return "profile_report.json";
}

static std::string trimCopy(const std::string& input) {
    size_t begin = 0;
    while (begin < input.size() &&
           std::isspace(static_cast<unsigned char>(input[begin]))) {
        ++begin;
    }

    size_t end = input.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(input[end - 1]))) {
        --end;
    }
    return input.substr(begin, end - begin);
}

static std::string stripTrailingComma(const std::string& input) {
    std::string trimmed = trimCopy(input);
    if (!trimmed.empty() && trimmed.back() == ',') {
        trimmed.pop_back();
    }
    return trimCopy(trimmed);
}

static bool hasPrefix(const std::string& input, const std::string& prefix) {
    return input.rfind(prefix, 0) == 0;
}

static bool hasSuffix(const std::string& input, const std::string& suffix) {
    return input.size() >= suffix.size() &&
           input.compare(input.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::string unescapeJsonString(const std::string& input) {
    std::ostringstream decoded;
    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i] != '\\' || i + 1 >= input.size()) {
            decoded << input[i];
            continue;
        }

        char escaped = input[++i];
        switch (escaped) {
        case '\\':
            decoded << '\\';
            break;
        case '"':
            decoded << '"';
            break;
        case 'n':
            decoded << '\n';
            break;
        case 'r':
            decoded << '\r';
            break;
        case 't':
            decoded << '\t';
            break;
        default:
            decoded << escaped;
            break;
        }
    }
    return decoded.str();
}

static std::optional<std::string> parseJsonStringValue(const std::string& rawValue) {
    const std::string value = stripTrailingComma(rawValue);
    if (value == "null") {
        return std::nullopt;
    }
    if (value.size() < 2 || value.front() != '"' || value.back() != '"') {
        return std::nullopt;
    }
    return unescapeJsonString(value.substr(1, value.size() - 2));
}

static std::optional<std::string> parseManifestStringField(
    const std::string& trimmedLine, const char* key) {
    const std::string prefix = std::string("\"") + key + "\":";
    if (!hasPrefix(trimmedLine, prefix)) {
        return std::nullopt;
    }
    return parseJsonStringValue(trimmedLine.substr(prefix.size()));
}

static std::optional<int64_t> parseManifestIntField(
    const std::string& trimmedLine, const char* key) {
    const std::string prefix = std::string("\"") + key + "\":";
    if (!hasPrefix(trimmedLine, prefix)) {
        return std::nullopt;
    }

    const std::string value = stripTrailingComma(trimmedLine.substr(prefix.size()));
    if (value.empty()) {
        return std::nullopt;
    }

    try {
        return std::stoll(value);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

static bool fileExists(const std::string& path) {
    return !path.empty() && access(path.c_str(), F_OK) == 0;
}

static std::string parentDirectory(const std::string& path) {
    const size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return ".";
    }
    if (pos == 0) {
        return "/";
    }
    return path.substr(0, pos);
}

static std::string joinPath(const std::string& dir, const std::string& fileName) {
    if (dir.empty() || dir == ".") {
        return fileName;
    }
    if (dir.back() == '/') {
        return dir + fileName;
    }
    return dir + "/" + fileName;
}

static std::optional<std::string> replaceSuffix(
    const std::string& path, const std::string& suffix,
    const std::string& replacement) {
    if (!hasSuffix(path, suffix)) {
        return std::nullopt;
    }
    return path.substr(0, path.size() - suffix.size()) + replacement;
}

static std::string resolveProfileManifestPath(const std::string& outputPath) {
    const char* envPath = std::getenv("NPU_PROFILE_MANIFEST");
    if (envPath && envPath[0] != '\0') {
        return std::string(envPath);
    }

    const std::string outputDir = parentDirectory(outputPath);
    std::vector<std::string> candidates;

    if (auto replaced = replaceSuffix(
            outputPath, "_profile_report.json", "_profile_manifest.json")) {
        candidates.push_back(*replaced);
    }
    if (auto replaced =
            replaceSuffix(outputPath, "profile_report.json", "profile_manifest.json")) {
        candidates.push_back(*replaced);
    }

    candidates.push_back(joinPath(outputDir, "profile_manifest.json"));
    candidates.push_back(joinPath(outputDir, "NpuPartition/profile_manifest.json"));

    for (const std::string& candidate : candidates) {
        if (fileExists(candidate)) {
            return candidate;
        }
    }
    return "";
}

static std::map<int64_t, LayerManifestRecord> loadManifestMetadata(
    const std::string& manifestPath) {
    std::map<int64_t, LayerManifestRecord> manifestLayers;
    if (manifestPath.empty() || !fileExists(manifestPath)) {
        return manifestLayers;
    }

    std::ifstream in(manifestPath);
    if (!in.is_open()) {
        return manifestLayers;
    }

    bool inLayersArray = false;
    bool inLayerObject = false;
    bool inFusedOps = false;
    LayerManifestRecord currentLayer;
    std::string line;

    while (std::getline(in, line)) {
        const std::string trimmed = trimCopy(line);
        if (trimmed.empty()) {
            continue;
        }

        if (!inLayersArray) {
            if (trimmed == "\"layers\": [") {
                inLayersArray = true;
            }
            continue;
        }

        if (!inLayerObject) {
            if (trimmed == "{") {
                currentLayer = LayerManifestRecord{};
                inLayerObject = true;
            } else if (trimmed == "]" || trimmed == "],") {
                break;
            }
            continue;
        }

        if (inFusedOps) {
            if (trimmed == "]" || trimmed == "],") {
                inFusedOps = false;
                continue;
            }
            if (auto fusedOp = parseJsonStringValue(trimmed)) {
                currentLayer.fusedOps.push_back(*fusedOp);
            }
            continue;
        }

        if (trimmed == "}" || trimmed == "},") {
            if (currentLayer.layerId.has_value()) {
                manifestLayers[*currentLayer.layerId] = currentLayer;
            }
            inLayerObject = false;
            continue;
        }

        if (trimmed == "\"fused_ops\": [") {
            inFusedOps = true;
            continue;
        }

        if (auto layerId = parseManifestIntField(trimmed, "layer_id")) {
            currentLayer.layerId = *layerId;
            continue;
        }
        if (auto layerName = parseManifestStringField(trimmed, "onnx_node_name")) {
            currentLayer.layerName = *layerName;
            continue;
        }
        if (auto originOpType =
                parseManifestStringField(trimmed, "origin_op_type")) {
            currentLayer.originOpType = *originOpType;
            continue;
        }
        if (auto device = parseManifestStringField(trimmed, "device")) {
            currentLayer.device = *device;
            continue;
        }
        if (hasPrefix(trimmed, "\"fallback_reason\":")) {
            currentLayer.fallbackReason =
                parseJsonStringValue(trimmed.substr(std::strlen("\"fallback_reason\":")));
            continue;
        }
    }

    return manifestLayers;
}

static ProfilerState& getProfilerState() {
    static ProfilerState state;
    return state;
}

static void dumpProfilerReport(const char* pathOverride);

static void ensureProfilerInitialized() {
    if (isProfilingDisabled()) {
        return;
    }

    ProfilerState& state = getProfilerState();
    if (state.initialized) {
        return;
    }

    state.initialized = true;
    state.outputPath = resolveProfileOutputPath(nullptr);
    if (!state.registeredAtExit) {
        std::atexit([]() { dumpProfilerReport(nullptr); });
        state.registeredAtExit = true;
    }
}

static void finishActiveLayer(int64_t expectedLayerId) {
    if (isProfilingDisabled() || !g_activeLayer.active) {
        return;
    }

    const int64_t activeLayerId = g_activeLayer.layerId;
    if (expectedLayerId >= 0 && activeLayerId != expectedLayerId) {
        NPU_PROFILE_LOG("layer id mismatch: active=%lld requested=%lld",
            static_cast<long long>(activeLayerId),
            static_cast<long long>(expectedLayerId));
    }

    auto end = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  end - g_activeLayer.start)
                  .count();

    ProfilerState& state = getProfilerState();
    std::lock_guard<std::mutex> lock(state.mutex);
    LayerProfileRecord& record = state.layers[activeLayerId];
    record.totalNs += static_cast<uint64_t>(ns);
    record.invocations += 1;

    g_activeLayer = ActiveLayerFrame{};
}

static void beginLayerProfile(int64_t layerId) {
    if (isProfilingDisabled()) {
        return;
    }

    ensureProfilerInitialized();
    if (g_activeLayer.active) {
        finishActiveLayer(g_activeLayer.layerId);
    }

    g_activeLayer.active = true;
    g_activeLayer.layerId = layerId;
    g_activeLayer.start = std::chrono::steady_clock::now();
}

static void endLayerProfile(int64_t layerId) {
    if (isProfilingDisabled() || !g_activeLayer.active) {
        return;
    }
    finishActiveLayer(layerId);
}

static void addStageSample(
    ProfileStage stage, uint64_t ns, ProfileCallCounter counter) {
    if (isProfilingDisabled() || !g_activeLayer.active || g_activeLayer.layerId < 0) {
        return;
    }

    ensureProfilerInitialized();
    ProfilerState& state = getProfilerState();
    std::lock_guard<std::mutex> lock(state.mutex);
    LayerProfileRecord& record = state.layers[g_activeLayer.layerId];

    switch (stage) {
    case ProfileStage::DmaIn:
        record.dmaInNs += ns;
        break;
    case ProfileStage::Compute:
        record.computeNs += ns;
        break;
    case ProfileStage::DmaOut:
        record.dmaOutNs += ns;
        break;
    case ProfileStage::Layout:
        record.layoutNs += ns;
        break;
    case ProfileStage::WaitIrq:
        record.waitIrqNs += ns;
        break;
    }

    switch (counter) {
    case ProfileCallCounter::Mvin:
        record.mvinCalls += 1;
        break;
    case ProfileCallCounter::Compute:
        record.computeCalls += 1;
        break;
    case ProfileCallCounter::Mvout:
        record.mvoutCalls += 1;
        break;
    case ProfileCallCounter::Layout:
        record.layoutCalls += 1;
        break;
    case ProfileCallCounter::None:
        break;
    }
}

class ScopedStageTimer {
public:
    ScopedStageTimer(ProfileStage stage, ProfileCallCounter counter = ProfileCallCounter::None)
        : stage_(stage), counter_(counter), enabled_(!isProfilingDisabled() && g_activeLayer.active),
          start_(enabled_ ? std::chrono::steady_clock::now()
                          : std::chrono::steady_clock::time_point()) {}

    ~ScopedStageTimer() {
        if (!enabled_) {
            return;
        }
        auto end = std::chrono::steady_clock::now();
        auto ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
        addStageSample(stage_, static_cast<uint64_t>(ns), counter_);
    }

private:
    ProfileStage stage_;
    ProfileCallCounter counter_;
    bool enabled_;
    std::chrono::steady_clock::time_point start_;
};

static bool layerHasNpuWork(const LayerProfileRecord& record) {
    return record.dmaInNs > 0 || record.computeNs > 0 || record.dmaOutNs > 0 ||
           record.layoutNs > 0;
}

static void writeJsonString(std::ostream& os, const std::string& value) {
    os << '"' << jsonEscape(value) << '"';
}

static void writeJsonOptionalString(
    std::ostream& os, const std::optional<std::string>& value) {
    if (value.has_value()) {
        writeJsonString(os, *value);
        return;
    }
    os << "null";
}

static void writeJsonStringArray(
    std::ostream& os, const std::vector<std::string>& values) {
    os << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        writeJsonString(os, values[i]);
    }
    os << "]";
}

static void dumpProfilerReport(const char* pathOverride) {
    if (isProfilingDisabled()) {
        return;
    }

    ProfilerState& state = getProfilerState();
    if (!state.initialized) {
        return;
    }
    if (g_activeLayer.active) {
        finishActiveLayer(g_activeLayer.layerId);
    }

    std::map<int64_t, LayerProfileRecord> snapshot;
    std::string outputPath;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        snapshot = state.layers;
        outputPath = resolveProfileOutputPath(pathOverride ? pathOverride : state.outputPath.c_str());
        state.outputPath = outputPath;
    }
    const std::string manifestPath = resolveProfileManifestPath(outputPath);
    const std::map<int64_t, LayerManifestRecord> manifestLayers =
        loadManifestMetadata(manifestPath);

    uint64_t totalNs = 0;
    uint64_t cpuTotalNs = 0;
    uint64_t npuTotalNs = 0;
    uint64_t npuLayerCount = 0;
    std::vector<std::pair<int64_t, uint64_t>> hotLayers;
    hotLayers.reserve(snapshot.size());

    for (const auto& entry : snapshot) {
        const LayerProfileRecord& record = entry.second;
        totalNs += record.totalNs;
        if (layerHasNpuWork(record)) {
            npuTotalNs += record.totalNs;
            npuLayerCount += 1;
        } else {
            cpuTotalNs += record.totalNs;
        }
        hotLayers.emplace_back(entry.first, record.totalNs);
    }

    std::sort(hotLayers.begin(), hotLayers.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

    std::ofstream out(outputPath);
    if (!out.is_open()) {
        fprintf(
            stderr, "[NPU_ERROR] Failed to open profiling report output: %s\n",
            outputPath.c_str());
        return;
    }

    out << "{\n";
    out << "  \"summary\": {\n";
    out << "    \"total_ns\": " << totalNs << ",\n";
    out << "    \"cpu_total_ns\": " << cpuTotalNs << ",\n";
    out << "    \"npu_total_ns\": " << npuTotalNs << ",\n";
    out << "    \"npu_latency_share\": "
        << (totalNs == 0 ? 0.0 : static_cast<double>(npuTotalNs) / static_cast<double>(totalNs))
        << ",\n";
    out << "    \"npu_layer_share\": "
        << (snapshot.empty() ? 0.0
                             : static_cast<double>(npuLayerCount) /
                                   static_cast<double>(snapshot.size()))
        << ",\n";
    out << "    \"npu_macs_share\": null,\n";
    out << "    \"top_hot_layers\": [";
    for (size_t i = 0; i < hotLayers.size() && i < 5; ++i) {
        if (i != 0) {
            out << ", ";
        }
        const auto metadataIt = manifestLayers.find(hotLayers[i].first);
        out << "{";
        out << "\"layer_id\": " << hotLayers[i].first << ", ";
        if (metadataIt != manifestLayers.end()) {
            out << "\"layer_name\": ";
            writeJsonString(out, metadataIt->second.layerName);
            out << ", ";
            out << "\"origin_op_type\": ";
            writeJsonString(out, metadataIt->second.originOpType);
            out << ", ";
        }
        out << "\"total_ns\": " << hotLayers[i].second;
        out << "}";
    }
    out << "]\n";
    out << "  },\n";
    out << "  \"layers\": [\n";

    bool firstLayer = true;
    for (const auto& entry : snapshot) {
        const int64_t layerId = entry.first;
        const LayerProfileRecord& record = entry.second;
        const bool isNpuLayer = layerHasNpuWork(record);
        const auto metadataIt = manifestLayers.find(layerId);
        const LayerManifestRecord* metadata =
            metadataIt == manifestLayers.end() ? nullptr : &metadataIt->second;
        const std::string device =
            metadata && !metadata->device.empty() ? metadata->device
                                                 : (isNpuLayer ? "npu" : "cpu");
        const uint64_t stagedNs =
            record.dmaInNs + record.computeNs + record.dmaOutNs + record.layoutNs;
        const uint64_t otherNs =
            record.totalNs > stagedNs ? record.totalNs - stagedNs
                                      : (isNpuLayer ? 0 : record.totalNs);

        if (!firstLayer) {
            out << ",\n";
        }
        firstLayer = false;

        out << "    {\n";
        out << "      \"layer_id\": " << layerId << ",\n";
        out << "      \"layer_name\": ";
        writeJsonOptionalString(
            out, metadata ? std::optional<std::string>(metadata->layerName)
                          : std::nullopt);
        out << ",\n";
        out << "      \"onnx_node_name\": ";
        writeJsonOptionalString(
            out, metadata ? std::optional<std::string>(metadata->layerName)
                          : std::nullopt);
        out << ",\n";
        out << "      \"origin_op_type\": ";
        writeJsonOptionalString(
            out, metadata ? std::optional<std::string>(metadata->originOpType)
                          : std::nullopt);
        out << ",\n";
        out << "      \"device\": ";
        writeJsonString(out, device);
        out << ",\n";
        out << "      \"fused_ops\": ";
        if (metadata) {
            writeJsonStringArray(out, metadata->fusedOps);
        } else {
            out << "[]";
        }
        out << ",\n";
        out << "      \"fallback_reason\": ";
        writeJsonOptionalString(
            out, metadata ? metadata->fallbackReason : std::nullopt);
        out << ",\n";
        out << "      \"invocations\": " << record.invocations << ",\n";
        out << "      \"total_ns\": " << record.totalNs << ",\n";
        out << "      \"avg_ns\": "
            << (record.invocations == 0
                    ? 0.0
                    : static_cast<double>(record.totalNs) /
                          static_cast<double>(record.invocations))
            << ",\n";
        out << "      \"pct_total\": "
            << (totalNs == 0
                    ? 0.0
                    : static_cast<double>(record.totalNs) /
                          static_cast<double>(totalNs) * 100.0)
            << ",\n";
        out << "      \"dma_in_ns\": " << record.dmaInNs << ",\n";
        out << "      \"compute_ns\": " << record.computeNs << ",\n";
        out << "      \"dma_out_ns\": " << record.dmaOutNs << ",\n";
        out << "      \"layout_ns\": " << record.layoutNs << ",\n";
        out << "      \"wait_irq_ns\": " << record.waitIrqNs << ",\n";
        out << "      \"other_ns\": " << otherNs << ",\n";
        out << "      \"mvin_calls\": " << record.mvinCalls << ",\n";
        out << "      \"compute_calls\": " << record.computeCalls << ",\n";
        out << "      \"mvout_calls\": " << record.mvoutCalls << ",\n";
        out << "      \"layout_calls\": " << record.layoutCalls << "\n";
        out << "    }";
    }
    out << "\n  ]\n";
    out << "}\n";
    out.close();

    NPU_PROFILE_LOG("report written to %s", outputPath.c_str());
}

} // namespace

// ==========================================
// Debug/Release Mode Configuration
// ==========================================

// #define NPU_DEBUG 

#ifndef NPU_CPU_WIDTH
#define NPU_CPU_WIDTH 64
#endif

#ifdef NPU_DEBUG
    #define NPU_LOG(fmt, ...) \
        fprintf(stdout, "[NPU_DEBUG] " fmt "\n", ##__VA_ARGS__)
    
    #define NPU_REG_LOG(offset, val, width) \
        fprintf(stdout, "[NPU_REG] WR%d Offset:0x%04X Val:0x%0llX\n", width, offset, (unsigned long long)val)
#else
    #define NPU_LOG(fmt, ...) do {} while(0)
    #define NPU_REG_LOG(offset, val, width) do {} while(0)
#endif

#define NPU_ERR(fmt, ...) fprintf(stderr, "[NPU_ERROR] " fmt "\n", ##__VA_ARGS__)

#ifndef NPU_CAPI_TRACE
#define NPU_CAPI_TRACE 0
#endif

#if NPU_CAPI_TRACE
    #define NPU_CAPI_LOG(fmt, ...) \
        fprintf(stdout, "[NPU_CAPI] " fmt "\n", ##__VA_ARGS__)
#else
    #define NPU_CAPI_LOG(fmt, ...) do {} while(0)
#endif

// ==========================================
// Constants & Definitions
// ==========================================

// Shadow register cache enable (1=on, 0=off)
#ifndef NPU_SHADOW_REGS
#define NPU_SHADOW_REGS 1
#endif


#define REG_MAP_OFFSET      NPU_KV260_MMAP_REGS_OFFSET
#define REG_MAP_SIZE        NPU_KV260_REG_MMAP_SIZE
#define DEFAULT_CMA_MAP_SIZE (256ULL * 1024ULL * 1024ULL)

// IOCTL Commands (Must match driver)
#define IOCTL_WAIT_IRQ      NPU_KV260_IOC_WAIT_IRQ
#define IOCTL_RESET_DEV     NPU_KV260_IOC_RESET_DEV
#define IOCTL_SET_IRQ_MODE  NPU_KV260_IOC_SET_IRQ_MODE

// IRQ Mode Constants
#define IRQ_MODE_KERNEL     NPU_KV260_IRQ_MODE_KERNEL
#define IRQ_MODE_USERSPACE  NPU_KV260_IRQ_MODE_USERSPACE

// ==========================================
// Hybrid Polling Configuration
// ==========================================
// 先轮询后中断策略：先快速轮询一定次数，若未完成则切换到中断等待
// 可根据实际硬件延迟调整以下参数

#ifndef NPU_POLL_SPIN_COUNT
#define NPU_POLL_SPIN_COUNT     1000    // 首轮自旋轮询次数（无延迟）
#endif

#ifndef NPU_POLL_YIELD_COUNT  
#define NPU_POLL_YIELD_COUNT    100     // 让出CPU的轮询次数
#endif

#ifndef NPU_POLL_YIELD_US
#define NPU_POLL_YIELD_US       1       // 每次让出的微秒数
#endif

static NpuRuntime* g_npu_runtime = nullptr;

static uint64_t parse_size_with_suffix(const char* value, uint64_t fallback) {
    if (!value || value[0] == '\0') return fallback;

    errno = 0;
    char* end = nullptr;
    unsigned long long parsed = strtoull(value, &end, 0);
    if (errno || end == value) return fallback;

    while (*end && std::isspace(static_cast<unsigned char>(*end))) ++end;

    uint64_t multiplier = 1;
    if (*end) {
        char suffix = static_cast<char>(std::tolower(static_cast<unsigned char>(*end)));
        if (suffix == 'k') {
            multiplier = 1024ULL;
            ++end;
        } else if (suffix == 'm') {
            multiplier = 1024ULL * 1024ULL;
            ++end;
        } else if (suffix == 'g') {
            multiplier = 1024ULL * 1024ULL * 1024ULL;
            ++end;
        } else {
            return fallback;
        }

        if (*end == 'i' || *end == 'I') ++end;
        if (*end == 'b' || *end == 'B') ++end;
        while (*end && std::isspace(static_cast<unsigned char>(*end))) ++end;
        if (*end) return fallback;
    }

    if (parsed > std::numeric_limits<uint32_t>::max() / multiplier) {
        return fallback;
    }
    return static_cast<uint64_t>(parsed) * multiplier;
}

static uint32_t resolve_cma_request_size() {
    uint64_t size = parse_size_with_suffix(std::getenv("NPU_CMA_SIZE"), DEFAULT_CMA_MAP_SIZE);
    if (size == 0 || size > std::numeric_limits<uint32_t>::max()) {
        size = DEFAULT_CMA_MAP_SIZE;
    }
    return static_cast<uint32_t>(size);
}

// ==========================================
// Lifecycle
// ==========================================

NpuRuntime::NpuRuntime()
    : fd(-1),
      regs_virt_base(nullptr),
      data_virt_base(nullptr),
      data_phy_base(0),
      data_map_size(0),
      free_list_head(nullptr) {
    reset_shadows();
    std::memset(pending_mvin_staging, 0, sizeof(pending_mvin_staging));
}

NpuRuntime::~NpuRuntime() {
    release_mvin_staging((1u << DMA_CHANNEL_COUNT) - 1u);
    if (regs_virt_base) munmap(regs_virt_base, REG_MAP_SIZE);
    if (data_virt_base) munmap(data_virt_base, data_map_size);
    if (fd >= 0) ioctl(fd, NPU_KV260_IOC_FREE_BUFFER);
    if (fd >= 0) close(fd);
    NPU_LOG("NpuRuntime Destroyed.");
}

bool NpuRuntime::init() {
    fd = open(NPU_KV260_DEV_PATH, O_RDWR);
    if (fd < 0) {
        perror("Failed to open " NPU_KV260_DEV_PATH);
        return false;
    }

    // 1. 从 driver/CMA 申请 NPU 可访问的连续内存。
    struct npu_kv260_buffer_request req = {};
    req.size = resolve_cma_request_size();
    if (ioctl(fd, NPU_KV260_IOC_ALLOC_BUFFER, &req) < 0) {
        perror("IOCTL_ALLOC_BUFFER failed");
        close(fd); fd = -1;
        return false;
    }
    if (req.dma_addr > std::numeric_limits<uint32_t>::max() ||
        req.size > std::numeric_limits<uint32_t>::max() ||
        req.dma_addr == 0 || req.size == 0) {
        NPU_ERR("Invalid CMA buffer from driver: dma=0x%llx size=0x%llx",
                static_cast<unsigned long long>(req.dma_addr),
                static_cast<unsigned long long>(req.size));
        ioctl(fd, NPU_KV260_IOC_FREE_BUFFER);
        close(fd); fd = -1;
        return false;
    }
    data_phy_base = static_cast<uint32_t>(req.dma_addr);
    data_map_size = static_cast<uint32_t>(req.size);

    // 2. 映射寄存器
    regs_virt_base = mmap(NULL, REG_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)REG_MAP_OFFSET);
    if (regs_virt_base == MAP_FAILED) {
        perror("MMAP Registers failed");
        ioctl(fd, NPU_KV260_IOC_FREE_BUFFER);
        close(fd); fd = -1;
        return false;
    }

    // 3. 映射 CMA 数据区
    data_virt_base = mmap(NULL, data_map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data_virt_base == MAP_FAILED) {
        perror("MMAP CMA buffer failed");
        munmap(regs_virt_base, REG_MAP_SIZE); regs_virt_base = nullptr;
        ioctl(fd, NPU_KV260_IOC_FREE_BUFFER);
        close(fd); fd = -1;
        return false;
    }

    NPU_LOG("NPU Runtime Initialized (Shadow Regs: Enabled).");
    init_allocator();
    
    // 4. [新增] 启动时执行一次复位，确保硬件和软件状态同步
    // 注意：必须先设置IRQ模式，因为reset()内部会根据当前模式配置IER
    ioctl(fd, IOCTL_SET_IRQ_MODE, IRQ_MODE_USERSPACE);
    
    // 5. 执行复位，驱动会根据当前IRQ模式正确配置IER
    reset();
    
    // 6. 再次确认IRQ模式（防止reset后状态不一致）
    ioctl(fd, IOCTL_SET_IRQ_MODE, IRQ_MODE_USERSPACE); 
    
    return true;
}

// ----------------------------------------------------
// [新增] 核心复位实现
// ----------------------------------------------------
void NpuRuntime::reset() {
    NPU_LOG("Requesting NPU Hardware Reset...");
    release_mvin_staging((1u << DMA_CHANNEL_COUNT) - 1u);
    
    // 1. 调用驱动接口执行硬件脉冲复位
    if (ioctl(fd, IOCTL_RESET_DEV) < 0) {
        // 在 Release 模式下可能需要根据 errno 判断是否要报警
        // 比如如果驱动太旧不支持该 IOCTL，返回 ENOTTY
        NPU_ERR("Failed to reset NPU hardware via IOCTL (errno=%d)", errno);
    } else {
        NPU_LOG("NPU Hardware Reset Pulse OK.");
    }

    // 2. [关键] 重置影子寄存器
    // 硬件复位后，内部寄存器归零，必须让 Runtime 知道这一点
    // 否则 Runtime 以为缓存值仍有效，导致配置不生效
    reset_shadows();
}

void* NpuRuntime::get_memory_base() {
    return data_virt_base;
}

uint32_t NpuRuntime::get_memory_size() {
    return data_map_size;
}

// ==========================================
// Register Operations & Shadow Logic
// ==========================================

void NpuRuntime::reg_write(uint32_t offset, uint32_t val) {
    NPU_REG_LOG(offset, val, 32);
    volatile uint32_t* reg_ptr = (volatile uint32_t*)((char*)regs_virt_base + offset);
    *reg_ptr = val;
}

void NpuRuntime::reg_write64(uint32_t offset, uint64_t val) {
    NPU_REG_LOG(offset, val, 64);

#if NPU_CPU_WIDTH == 64
    volatile uint64_t* reg_ptr = (volatile uint64_t*)((char*)regs_virt_base + offset);
    *reg_ptr = val;
#elif NPU_CPU_WIDTH == 32
    reg_write(offset, (uint32_t)(val & 0xFFFFFFFF));
    reg_write(offset + 4, (uint32_t)((val >> 32) & 0xFFFFFFFF));
#else
#error "NPU_CPU_WIDTH must be 32 or 64"
#endif
}

uint32_t NpuRuntime::reg_read(uint32_t offset) {
    volatile uint32_t* reg_ptr = (volatile uint32_t*)((char*)regs_virt_base + offset);
    uint32_t val = *reg_ptr;
    #ifdef NPU_DEBUG
    fprintf(stdout, "[NPU_REG] READ Offset:0x%04X Val:0x%08X\n", offset, val);
    #endif
    return val;
}

uint32_t NpuRuntime::read_dma_busy_mask(bool is_mvin) {
    const uint32_t raw_status = reg_read(RegOffset::DMA_STATUS);
    return is_mvin
        ? REG_GET_FIELD(DMA_STATUS, MVIN_BUSY, raw_status)
        : REG_GET_FIELD(DMA_STATUS, MVOUT_BUSY, raw_status);
}

void NpuRuntime::validate_dma_id(uint32_t dma_id) const {
    if (dma_id >= DMA_CHANNEL_COUNT) {
        throw std::runtime_error(std::string("Invalid DMA id ") + std::to_string(dma_id));
    }
}

void NpuRuntime::validate_dma_mask(uint32_t dma_mask) const {
    const uint32_t valid_mask = (1u << DMA_CHANNEL_COUNT) - 1u;
    if ((dma_mask & ~valid_mask) != 0) {
        throw std::runtime_error(std::string("Invalid DMA mask 0x") + std::to_string(dma_mask));
    }
}

void NpuRuntime::validate_mvin_dma_cfg(uint32_t dma_id, const MvinConfig& cfg) const {
    validate_dma_id(dma_id);
    if (dma_id == 0) {
        return;
    }
    if (cfg.dest || cfg.is_bias || cfg.is_quant || cfg.input_type == 2) {
        throw std::runtime_error("DMA1 only supports direct DRAM->SPM MVIN");
    }
}

void NpuRuntime::validate_mvout_dma_cfg(uint32_t dma_id, const MvoutConfig& cfg) const {
    validate_dma_id(dma_id);
    if (dma_id == 0) {
        return;
    }
    if (cfg.source || cfg.output_type != 0) {
        throw std::runtime_error("DMA1 only supports direct SPM->DRAM MVOUT");
    }
}

void NpuRuntime::release_mvin_staging(uint32_t dma_mask) {
    if (dma_mask == 0) {
        return;
    }
    validate_dma_mask(dma_mask);
    for (uint32_t dma_id = 0; dma_id < DMA_CHANNEL_COUNT; ++dma_id) {
        if ((dma_mask & (1u << dma_id)) == 0) {
            continue;
        }
        if (pending_mvin_staging[dma_id]) {
            free(pending_mvin_staging[dma_id]);
            pending_mvin_staging[dma_id] = nullptr;
        }
    }
}

void NpuRuntime::wait_dma_idle(bool is_mvin, uint32_t dma_mask) {
    if (dma_mask == 0) {
        return;
    }
    validate_dma_mask(dma_mask);

    const char* op_name = is_mvin ? "MVIN" : "MVOUT";
    NPU_LOG("Waiting for %s DMA mask=0x%X...", op_name, dma_mask);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    int iter = 0;
    uint32_t busy_mask = 0;

    while (true) {
        busy_mask = read_dma_busy_mask(is_mvin) & dma_mask;
        if (busy_mask == 0) {
            return;
        }

        ++iter;
        if (iter <= NPU_POLL_SPIN_COUNT) {
            continue;
        }

        if (std::chrono::steady_clock::now() >= deadline) {
            const uint32_t raw_status = reg_read(RegOffset::DMA_STATUS);
            dump_irq_regs();
            throw std::runtime_error(
                std::string(op_name) + " wait timeout, dma_mask=0x" +
                std::to_string(dma_mask) + " busy_mask=0x" + std::to_string(busy_mask) +
                " dma_status=0x" + std::to_string(raw_status));
        }

        usleep(NPU_POLL_YIELD_US);
    }
}

void NpuRuntime::reset_shadows() {
    // 0xFF 意味着所有缓存失效（假设没有合法配置全是 0xFF）
    // 或者，因为硬件复位后通常是0，我们也可以设为 0xFFFFFFFF 来强制第一次写入
#if NPU_SHADOW_REGS
    memset(&shadow, 0xFF, sizeof(ShadowRegs));
    NPU_LOG("Shadow registers cache cleared.");
#endif
}

void NpuRuntime::reg_write64_cached(uint32_t offset, uint64_t val, uint64_t* cache_ptr) {
#if NPU_SHADOW_REGS
    if (*cache_ptr != val) {
        #ifdef NPU_DEBUG
        fprintf(stdout, "[NPU_REG] CACHE_MISS Write Offset:0x%04X Val:0x%016llX\n", offset, (unsigned long long)val);
        #endif
        
        reg_write64(offset, val);
        *cache_ptr = val;
    } else {
        // NPU_LOG("Cache Hit Offset:0x%04X", offset);
    }
#else
    (void)cache_ptr;
    reg_write64(offset, val);
#endif
}

// 检查中断是否挂起（读取 ISR 寄存器）- 强制内联
// 由于IER被置为0，不能通过IRQ或者IPR来获得中断，应该时ISR
inline __attribute__((always_inline)) bool NpuRuntime::check_irq_pending() {
    volatile uint32_t* reg_ptr = (volatile uint32_t*)((char*)regs_virt_base + RegOffset::ISR);
    return (*reg_ptr != 0);
}

// 清除中断（写 IAR 寄存器）- 强制内联
inline __attribute__((always_inline)) void NpuRuntime::ack_irq(uint32_t mask) {
    volatile uint32_t* reg_ptr = (volatile uint32_t*)((char*)regs_virt_base + RegOffset::IAR);
    *reg_ptr = mask;
}

void NpuRuntime::dump_irq_regs() {
    
    // uint32_t iar = reg_read(RegOffset::IAR);
    uint32_t mer = reg_read(RegOffset::MER);
    uint32_t ier = reg_read(RegOffset::IER);
    uint32_t isr = reg_read(RegOffset::ISR);
    uint32_t ipr = reg_read(RegOffset::IPR);
    NPU_ERR("IRQ timeout:  MER=0x%08X IER=0x%08X ISR=0x%08X IPR=0x%08X", mer, ier, isr, ipr);
    // NPU_ERR("IRQ timeout: IAR=0x%08X MER=0x%08X IER=0x%08X ISR=0x%08X IPR=0x%08X", iar, mer, ier, isr, ipr);
}

void NpuRuntime::wait_irq() {
    ScopedStageTimer waitTimer(ProfileStage::WaitIrq);
    NPU_TIMER_SECTION_BEGIN("wait_irq")
    NPU_LOG("Waiting for IRQ (hybrid polling)...");
    
    // ========== Phase 1: 自旋轮询（无延迟，最低延迟路径）==========
    // 在用户态轮询模式下，IER=0，因此 IPR = ISR & IER = 0（永远为0）
    // 所以必须直接检查 ISR 寄存器，而不是 IPR
    // ISR 是原始中断状态，不受 IER 影响
    volatile uint32_t* isr_ptr = (volatile uint32_t*)((char*)regs_virt_base + RegOffset::ISR);
    
    // 循环展开：每次迭代检测4次，减少循环开销
    int i = 0;
    for (; i < NPU_POLL_SPIN_COUNT - 3; i += 4) {
        if (*isr_ptr) { ack_irq(); NPU_LOG("IRQ received via spin polling (iter=%d)", i); NPU_TIMER_SECTION_END() return; }
        if (*isr_ptr) { ack_irq(); NPU_LOG("IRQ received via spin polling (iter=%d)", i+1); NPU_TIMER_SECTION_END() return; }
        if (*isr_ptr) { ack_irq(); NPU_LOG("IRQ received via spin polling (iter=%d)", i+2); NPU_TIMER_SECTION_END() return; }
        if (*isr_ptr) { ack_irq(); NPU_LOG("IRQ received via spin polling (iter=%d)", i+3); NPU_TIMER_SECTION_END() return; }
    }
    // 处理剩余迭代
    for (; i < NPU_POLL_SPIN_COUNT; ++i) {
        if (*isr_ptr) { ack_irq(); NPU_LOG("IRQ received via spin polling (iter=%d)", i); NPU_TIMER_SECTION_END() return; }
    }
    
    // ========== Phase 2: 让出式轮询（短暂sleep，减少CPU占用）==========
    for (int i = 0; i < NPU_POLL_YIELD_COUNT; ++i) {
        if (check_irq_pending()) {
            ack_irq();
            NPU_LOG("IRQ received via yield polling (iter=%d)", i);
            NPU_TIMER_SECTION_END()
            return;
        }
        usleep(NPU_POLL_YIELD_US);
    }
    
    // ========== Phase 3: 中断等待（回退到阻塞模式）==========
    // 轮询超时，先打印中断相关寄存器用于诊断
    dump_irq_regs();
    // 【关键修复】在进入内核等待前，先清除可能已经挂起的中断，
    // 然后切换到内核模式，让内核ISR负责ACK后续中断
    ack_irq();
    
    // 切换到内核中断模式：从此刻起，内核ISR会ACK中断
    ioctl(fd, IOCTL_SET_IRQ_MODE, IRQ_MODE_KERNEL);
    
    NPU_LOG("Polling timeout, falling back to kernel IRQ wait...");
    uint32_t status = 0;
    int ret = ioctl(fd, IOCTL_WAIT_IRQ, &status);
    
    // 返回用户态轮询模式：为下一次 wait_irq 做准备
    ioctl(fd, IOCTL_SET_IRQ_MODE, IRQ_MODE_USERSPACE);
    
    if (ret < 0) {
        perror("Wait IRQ failed");
        NPU_ERR(
            "wait_irq failed after op=%s start_bit=0x%08X reg0=0x%016llX reg1=0x%016llX cfg0=0x%016llX cfg1=0x%016llX",
            g_last_op_ctx.op,
            g_last_op_ctx.start_bit,
            (unsigned long long) g_last_op_ctx.reg0,
            (unsigned long long) g_last_op_ctx.reg1,
            (unsigned long long) g_last_op_ctx.cfg0,
            (unsigned long long) g_last_op_ctx.cfg1);
        if (abort_on_irq_timeout()) {
            NPU_ERR("abort due to IRQ timeout (set NPU_ABORT_ON_IRQ_TIMEOUT=0 to disable abort)");
            std::abort();
        }
        NPU_TIMER_SECTION_END()
        return;
    }
    NPU_LOG("IRQ received via interrupt (status=0x%X).", status);
    NPU_TIMER_SECTION_END()
}

uint32_t NpuRuntime::virt_to_phys(void* ptr) {
    uintptr_t virt_addr = (uintptr_t)ptr;
    uintptr_t base_virt = (uintptr_t)data_virt_base;
    if (virt_addr < base_virt || virt_addr >= base_virt + data_map_size) {
        throw std::runtime_error("Pointer out of NPU CMA range, virt_addr=0x" + std::to_string(virt_addr) + " base_virt=0x" + std::to_string(base_virt) + " data_map_size=0x" + std::to_string(data_map_size));
    }
    return data_phy_base + (uint32_t)(virt_addr - base_virt);
}

// ==========================================
// Instruction Implementation
// ==========================================

void NpuRuntime::run_mvin(const MvinConfig& cfg) {
    NPU_TIMER_TOTAL("run_mvin");
    ScopedStageTimer profileTimer(ProfileStage::DmaIn, ProfileCallCounter::Mvin);
    validate_mvin_dma_cfg(0, cfg);

    const uint32_t busy_mask = read_dma_busy_mask(true);
    if (busy_mask & (1u << 0)) {
        NPU_LOG("DMA0 MVIN busy before sync launch, waiting for idle (busy_mask=0x%X)", busy_mask);
        wait_mvin(1u << 0);
    }
    if (pending_mvin_staging[0]) {
        throw std::runtime_error("DMA0 has unreleased MVIN staging buffer");
    }

    const uint8_t precision = 1; // force precision regardless of API input

    auto calc_transfer_bytes = [&](const MvinConfig& c) -> size_t {
        const uint64_t cols = static_cast<uint64_t>(c.col_num) + 1ULL;
        const uint64_t rows = static_cast<uint64_t>(c.row_num) + 1ULL;
        const uint64_t elem_bytes = 1ULL; // precision is forced to int8 path.
        const uint64_t stride_bytes =
            static_cast<uint64_t>(std::max<uint32_t>(c.dram_stride, c.col_num + 1U)) * elem_bytes;
        if (rows <= 1ULL) return static_cast<size_t>(cols * elem_bytes);
        const uint64_t total =
            (rows - 1ULL) * stride_bytes + cols * elem_bytes;
        return static_cast<size_t>(total);
    };

    void *dma_src_ptr = cfg.host_ptr;
    uint32_t phys_dram = 0;
    try {
        phys_dram = virt_to_phys(dma_src_ptr);
    } catch (const std::runtime_error &) {
        size_t transfer_bytes = calc_transfer_bytes(cfg);
        void * staging_ptr = alloc(transfer_bytes);
        if (!staging_ptr) {
            throw std::runtime_error(
                "run_mvin staging alloc failed for non-NPU host pointer");
        }
        std::memcpy(staging_ptr, cfg.host_ptr, transfer_bytes);
        pending_mvin_staging[0] = staging_ptr;
        dma_src_ptr = staging_ptr;
        phys_dram = virt_to_phys(dma_src_ptr);
        NPU_LOG(
            "MVIN sync DMA0 staged host ptr %p -> NPU ptr %p (%zu bytes)",
            cfg.host_ptr, staging_ptr, transfer_bytes);
    }

    NPU_TIMER_SECTION_BEGIN("run_mvin(pre_reg)")
    NPU_LOG("Running MVIN sync DMA0 (HostPtr=%p, phy_dram=0x%x, SRAM=0x%x)",
            dma_src_ptr, phys_dram, cfg.sram_addr);
    NPU_TIMER_SECTION_END()

    NPU_TIMER_SECTION_BEGIN("run_mvin(reg_write)")
    uint64_t val_dram = REG_FIELD(MVIN_CTRL0, DRAM_ADDR, phys_dram) |
                        REG_FIELD(MVIN_CTRL0, ROW_NUM, cfg.row_num);
    reg_write64(RegOffset::MVIN_DRAM_ADDR, val_dram);

    uint64_t val_sram = REG_FIELD(MVIN_CTRL1, SRAM_ADDR, cfg.sram_addr) |
                        REG_FIELD(MVIN_CTRL1, COL_NUM, cfg.col_num);
    reg_write64(RegOffset::MVIN_SRAM_ADDR, val_sram);

    uint64_t val_cfg = REG_FIELD(CFG_MVIN0, INPUT_TYPE, cfg.input_type) |
                       REG_FIELD(CFG_MVIN0, INPUT_PRECISION, precision) |
                       REG_FIELD(CFG_MVIN0, IS_QUANT, cfg.is_quant) |
                       REG_FIELD(CFG_MVIN0, DEST, cfg.dest) |
                       REG_FIELD(CFG_MVIN0, IS_BIAS, cfg.is_bias) |
                       REG_FIELD(CFG_MVIN0, SRAM_STRIDE, cfg.sram_stride) |
                       REG_FIELD(CFG_MVIN0, DRAM_STRIDE, cfg.dram_stride);
    reg_write64_cached(RegOffset::MVIN_CFG, val_cfg, &shadow.mvin_cfg);
    g_last_op_ctx = {
        "MVIN",
        val_dram,
        val_sram,
        val_cfg,
        0,
        BIT_START_DMA_MVIN | static_cast<uint32_t>(REG_FIELD(START_REG, MVIN_DMA_SEL, 0))
    };

    if (cfg.is_quant) {
        uint64_t val_quant = REG_FIELD(CFG_MVIN1, ZEROPOINT, cfg.quant_zero) |
                             REG_FIELD(CFG_MVIN1, SCALE, cfg.quant_scale) |
                             REG_FIELD(CFG_MVIN1, SCALE_SHIFT, cfg.quant_shift);
        reg_write64_cached(RegOffset::MVIN_QUANT, val_quant, &shadow.mvin_quant);
        g_last_op_ctx.cfg1 = val_quant;
    }
    const uint32_t start_word =
        BIT_START_DMA_MVIN | static_cast<uint32_t>(REG_FIELD(START_REG, MVIN_DMA_SEL, 0));
    reg_write(RegOffset::START, start_word);
    NPU_TIMER_SECTION_END()

    NPU_TIMER_SECTION_BEGIN("run_mvin(wait_dma)")
    wait_mvin(1u << 0);
    NPU_TIMER_SECTION_END()
}

void NpuRuntime::run_mvout(const MvoutConfig& cfg) {
    NPU_TIMER_TOTAL("run_mvout");
    ScopedStageTimer profileTimer(ProfileStage::DmaOut, ProfileCallCounter::Mvout);
    run_mvout_async(0, cfg);
    NPU_TIMER_SECTION_BEGIN("run_mvout(wait_dma)")
    wait_mvout(1u << 0);
    NPU_TIMER_SECTION_END()
}

void NpuRuntime::run_mvin_async(uint32_t dma_id, const MvinConfig& cfg) {
    validate_mvin_dma_cfg(dma_id, cfg);

    const uint32_t busy_mask = read_dma_busy_mask(true);
    if (busy_mask & (1u << dma_id)) {
        NPU_LOG(
            "DMA%u MVIN busy before async launch, waiting for idle (busy_mask=0x%X)",
            dma_id,
            busy_mask);
        wait_mvin(1u << dma_id);
    }
    if (pending_mvin_staging[dma_id]) {
        throw std::runtime_error(
            std::string("DMA") + std::to_string(dma_id) + " has unreleased MVIN staging buffer");
    }

    const uint8_t precision = 1; // force precision regardless of API input

    // Some compiler paths may pass constant pointers that are not inside NPU
    // DDR mmap range. In that case, stage data into NPU DDR first.
    auto calc_transfer_bytes = [&](const MvinConfig& c) -> size_t {
        const uint64_t cols = static_cast<uint64_t>(c.col_num) + 1ULL;
        const uint64_t rows = static_cast<uint64_t>(c.row_num) + 1ULL;
        const uint64_t elem_bytes = 1ULL; // precision is forced to int8 path.
        const uint64_t stride_bytes =
            static_cast<uint64_t>(std::max<uint32_t>(c.dram_stride, c.col_num + 1U)) * elem_bytes;
        if (rows <= 1ULL) return static_cast<size_t>(cols * elem_bytes);
        const uint64_t total =
            (rows - 1ULL) * stride_bytes + cols * elem_bytes;
        return static_cast<size_t>(total);
    };

    void *dma_src_ptr = cfg.host_ptr;
    uint32_t phys_dram = 0;
    try {
        phys_dram = virt_to_phys(dma_src_ptr);
    } catch (const std::runtime_error &) {
        size_t transfer_bytes = calc_transfer_bytes(cfg);
        void * staging_ptr = alloc(transfer_bytes);
        if (!staging_ptr) {
            throw std::runtime_error(
                "run_mvin_async staging alloc failed for non-NPU host pointer");
        }
        std::memcpy(staging_ptr, cfg.host_ptr, transfer_bytes);
        pending_mvin_staging[dma_id] = staging_ptr;
        dma_src_ptr = staging_ptr;
        phys_dram = virt_to_phys(dma_src_ptr);
        NPU_LOG(
            "MVIN async DMA%u staged host ptr %p -> NPU ptr %p (%zu bytes)",
            dma_id, cfg.host_ptr, staging_ptr, transfer_bytes);
    }

    NPU_TIMER_SECTION_BEGIN("run_mvin(pre_reg)")
    NPU_LOG("Running MVIN async DMA%u (HostPtr=%p, phy_dram=0x%x, SRAM=0x%x)",
            dma_id, dma_src_ptr, phys_dram, cfg.sram_addr);
    NPU_TIMER_SECTION_END()

    NPU_TIMER_SECTION_BEGIN("run_mvin(reg_write)")
    uint64_t val_dram = REG_FIELD(MVIN_CTRL0, DRAM_ADDR, phys_dram) |
                        REG_FIELD(MVIN_CTRL0, ROW_NUM, cfg.row_num);
    reg_write64(RegOffset::MVIN_DRAM_ADDR, val_dram);
    
    uint64_t val_sram = REG_FIELD(MVIN_CTRL1, SRAM_ADDR, cfg.sram_addr) |
                        REG_FIELD(MVIN_CTRL1, COL_NUM, cfg.col_num);
    reg_write64(RegOffset::MVIN_SRAM_ADDR, val_sram);
    
    uint64_t val_cfg = REG_FIELD(CFG_MVIN0, INPUT_TYPE, cfg.input_type) |
                       REG_FIELD(CFG_MVIN0, INPUT_PRECISION, precision) |
                       REG_FIELD(CFG_MVIN0, IS_QUANT, cfg.is_quant) |
                       REG_FIELD(CFG_MVIN0, DEST, cfg.dest) |
                       REG_FIELD(CFG_MVIN0, IS_BIAS, cfg.is_bias) |
                       REG_FIELD(CFG_MVIN0, SRAM_STRIDE, cfg.sram_stride) |
                       REG_FIELD(CFG_MVIN0, DRAM_STRIDE, cfg.dram_stride);
    reg_write64_cached(RegOffset::MVIN_CFG, val_cfg, &shadow.mvin_cfg);
    g_last_op_ctx = {
        "MVIN",
        val_dram,
        val_sram,
        val_cfg,
        0,
        BIT_START_DMA_MVIN | static_cast<uint32_t>(REG_FIELD(START_REG, MVIN_DMA_SEL, dma_id))
    };
    
    if (cfg.is_quant) {
        uint64_t val_quant = REG_FIELD(CFG_MVIN1, ZEROPOINT, cfg.quant_zero) |
                             REG_FIELD(CFG_MVIN1, SCALE, cfg.quant_scale) |
                             REG_FIELD(CFG_MVIN1, SCALE_SHIFT, cfg.quant_shift);
        reg_write64_cached(RegOffset::MVIN_QUANT, val_quant, &shadow.mvin_quant);
        g_last_op_ctx.cfg1 = val_quant;
    }
    const uint32_t start_word =
        BIT_START_DMA_MVIN | static_cast<uint32_t>(REG_FIELD(START_REG, MVIN_DMA_SEL, dma_id));
    reg_write(RegOffset::START, start_word);
    NPU_TIMER_SECTION_END()
}

void NpuRuntime::run_mvout_async(uint32_t dma_id, const MvoutConfig& cfg) {
    validate_mvout_dma_cfg(dma_id, cfg);

    const uint32_t busy_mask = read_dma_busy_mask(false);
    if (busy_mask & (1u << dma_id)) {
        NPU_LOG(
            "DMA%u MVOUT busy before async launch, waiting for idle (busy_mask=0x%X)",
            dma_id,
            busy_mask);
        wait_mvout(1u << dma_id);
    }

    const uint8_t precision = static_cast<uint8_t>(cfg.precision & 0x3);
    
    uint32_t phys_dram = virt_to_phys(cfg.host_ptr);
    NPU_TIMER_SECTION_BEGIN("run_mvout(pre_reg)")
    NPU_LOG("Running MVOUT async DMA%u (HostPtr=%p, phy_dram=0x%x, SRAM=0x%x)",
            dma_id, cfg.host_ptr, phys_dram, cfg.sram_addr);
    NPU_TIMER_SECTION_END()
    
    NPU_TIMER_SECTION_BEGIN("run_mvout(reg_write)")
    uint64_t val_dram = REG_FIELD(MVOUT_CTRL0, DRAM_ADDR, phys_dram) |
                        REG_FIELD(MVOUT_CTRL0, ROW_NUM, cfg.row_num);
    reg_write64(RegOffset::MVOUT_DRAM_ADDR, val_dram);
    
    uint64_t val_sram = REG_FIELD(MVOUT_CTRL1, SRAM_ADDR, cfg.sram_addr) |
                        REG_FIELD(MVOUT_CTRL1, COL_NUM, cfg.col_num);
    reg_write64(RegOffset::MVOUT_SRAM_ADDR, val_sram);
    
    uint64_t val_cfg = REG_FIELD(CFG_MVOUT0, OUTPUT_TYPE, cfg.output_type) |
                       REG_FIELD(CFG_MVOUT0, OUTPUT_PRECISION, precision) |
                       REG_FIELD(CFG_MVOUT0, IS_QUANT, cfg.is_quant) |
                       REG_FIELD(CFG_MVOUT0, SOURCE, cfg.source) |
                       REG_FIELD(CFG_MVOUT0, SRAM_STRIDE, cfg.sram_stride) |
                       REG_FIELD(CFG_MVOUT0, DRAM_STRIDE, cfg.dram_stride);
    reg_write64_cached(RegOffset::MVOUT_CFG, val_cfg, &shadow.mvout_cfg);
    g_last_op_ctx = {
        "MVOUT",
        val_dram,
        val_sram,
        val_cfg,
        0,
        BIT_START_DMA_MVOUT | static_cast<uint32_t>(REG_FIELD(START_REG, MVOUT_DMA_SEL, dma_id))
    };
    
    if (cfg.is_quant && precision == 3) {
        uint16_t quant_scale = static_cast<uint16_t>(cfg.f32_scale & 0xFFFF);//低位
        uint16_t quant_scaleshift = static_cast<uint16_t>((cfg.f32_scale >> 16) & 0xFFFF);//高位
        uint64_t val_quant = REG_FIELD(CFG_MVOUT1, ZEROPOINT, cfg.quant_zero) |
                             REG_FIELD(CFG_MVOUT1, SCALE, quant_scale) |
                             REG_FIELD(CFG_MVOUT1, SCALE_SHIFT, quant_scaleshift);
        reg_write64_cached(RegOffset::MVOUT_QUANT, val_quant, &shadow.mvout_quant);
        g_last_op_ctx.cfg1 = val_quant;
    }
    const uint32_t start_word =
        BIT_START_DMA_MVOUT | static_cast<uint32_t>(REG_FIELD(START_REG, MVOUT_DMA_SEL, dma_id));
    reg_write(RegOffset::START, start_word);
    NPU_TIMER_SECTION_END()
}

void NpuRuntime::wait_mvin(uint32_t dma_mask) {
    wait_dma_idle(true, dma_mask);
    release_mvin_staging(dma_mask);
    ack_irq(BIT_START_DMA_MVIN);
}

void NpuRuntime::wait_mvout(uint32_t dma_mask) {
    wait_dma_idle(false, dma_mask);
    ack_irq(BIT_START_DMA_MVOUT);
}

void NpuRuntime::run_double_mvin(const MvinConfig& dma0_cfg, const MvinConfig& dma1_cfg) {
    if (dma0_cfg.dest || dma0_cfg.is_bias || dma0_cfg.is_quant || dma0_cfg.input_type == 2 ||
        dma1_cfg.dest || dma1_cfg.is_bias || dma1_cfg.is_quant || dma1_cfg.input_type == 2) {
        throw std::runtime_error("npu_dma_double_mvin only supports direct DRAM->SPM MVIN");
    }

    bool dma0_started = false;
    bool dma1_started = false;
    try {
        run_mvin_async(0, dma0_cfg);
        dma0_started = true;
        run_mvin_async(1, dma1_cfg);
        dma1_started = true;
    } catch (...) {
        if (dma1_started) {
            try { wait_mvin(1u << 1); } catch (...) {}
        }
        if (dma0_started) {
            try { wait_mvin(1u << 0); } catch (...) {}
        }
        throw;
    }

    wait_mvin((1u << 0) | (1u << 1));
}

void NpuRuntime::run_sfu(const SfuConfig& cfg) {
    NPU_TIMER_TOTAL("run_sfu");
    ScopedStageTimer profileTimer(ProfileStage::Compute, ProfileCallCounter::Compute);
    
    NPU_TIMER_SECTION_BEGIN("run_sfu(pre_reg)")
    NPU_LOG("Running SFU (Op=%d)", cfg.op_type);
    NPU_TIMER_SECTION_END()
    
    NPU_TIMER_SECTION_BEGIN("run_sfu(reg_write)")
    uint64_t val_cfg1 = REG_FIELD(CFG_SFU0, OP, cfg.op_type) |
                        REG_FIELD(CFG_SFU0, INT_TYPE, cfg.int_type) |
                        REG_FIELD(CFG_SFU0, IS_QUANT, cfg.is_quant) |
                        REG_FIELD(CFG_SFU0, OUT_ZP, cfg.output_zeropoint) |
                        REG_FIELD(CFG_SFU0, IN_ZP, cfg.input_zeropoint);
    reg_write64_cached(RegOffset::CFG_SFU_1, val_cfg1, &shadow.sfu_cfg1);

    uint64_t val_cfg2 = REG_FIELD(CFG_SFU1, IN_SCALE, cfg.input_scale) |
                        REG_FIELD(CFG_SFU1, IN_SHIFT, cfg.input_scale_shift) |
                        REG_FIELD(CFG_SFU1, OUT_SCALE, cfg.output_scale) |
                        REG_FIELD(CFG_SFU1, OUT_SHIFT, cfg.output_scale_shift);
    reg_write64_cached(RegOffset::CFG_SFU_2, val_cfg2, &shadow.sfu_cfg2);

    uint64_t val_input = REG_FIELD(SFU_EXE0, IN_ADDR, cfg.input_sram_addr) |
                         REG_FIELD(SFU_EXE0, COL, cfg.input_col_num) |
                         REG_FIELD(SFU_EXE0, ROW, cfg.input_row_num);
    reg_write64(RegOffset::SFU_INPUT, val_input);

    uint64_t val_output = REG_FIELD(SFU_EXE1, OUT_ADDR, cfg.output_sram_addr);
    reg_write64(RegOffset::SFU_OUTPUT, val_output);

    reg_write(RegOffset::START, BIT_START_SFU);
    NPU_TIMER_SECTION_END()

    NPU_TIMER_SECTION_BEGIN("run_sfu(wait_irq)")
    wait_irq();
    NPU_TIMER_SECTION_END()
}

void NpuRuntime::run_conv(const ConvConfig& cfg) {
    NPU_TIMER_TOTAL("run_conv");
    ScopedStageTimer profileTimer(ProfileStage::Compute, ProfileCallCounter::Compute);
    
    NPU_TIMER_SECTION_BEGIN("run_conv(pre_reg)")
    NPU_LOG("Running CONV (DataFlow=%d)", cfg.dataflow_mode);
    NPU_TIMER_SECTION_END()
    
    NPU_TIMER_SECTION_BEGIN("run_conv(reg_write)")
    // 1. Compute Config 1
    uint64_t val_cfg1 = REG_FIELD(CFG_COMPUTE0, DATAFLOW, cfg.dataflow_mode) |
                        REG_FIELD(CFG_COMPUTE0, PAD_L, cfg.pad_left) |
                        REG_FIELD(CFG_COMPUTE0, PAD_R, cfg.pad_right) |
                        REG_FIELD(CFG_COMPUTE0, PAD_T, cfg.pad_top) |
                        REG_FIELD(CFG_COMPUTE0, PAD_B, cfg.pad_bottom) |
                        REG_FIELD(CFG_COMPUTE0, PAD_MODE, cfg.pad_mode) |
                        REG_FIELD(CFG_COMPUTE0, WEIGHT_SHAPE, cfg.weight_shape_m1) |
                        REG_FIELD(CFG_COMPUTE0, WEIGHT_STRIDE, cfg.weight_stride_m1) |
                        REG_FIELD(CFG_COMPUTE0, WEIGHT_DILATION, cfg.weight_dilation_m1) |
                        REG_FIELD(CFG_COMPUTE0, IS_GROUP, cfg.is_group_conv) |
                        REG_FIELD(CFG_COMPUTE0, INT_TYPE, cfg.int_type) |
                        REG_FIELD(CFG_COMPUTE0, OPTYPE, cfg.op_type) |
                        REG_FIELD(CFG_COMPUTE0, ACCOUT_DEST, cfg.accout_dest) |
                        REG_FIELD(CFG_COMPUTE0, ASYMMETRIC_ACTIVATIONS, 0) |
                        REG_FIELD(CFG_COMPUTE0, INPUTA_ZP, cfg.input_a_zeropoint) |
                        REG_FIELD(CFG_COMPUTE0, INPUTB_ZP, cfg.input_b_zeropoint);
    reg_write64_cached(RegOffset::CFG_COMPUTE_1, val_cfg1, &shadow.compute_cfg1);

    // 2. Compute Config 2
    uint64_t val_cfg2 = REG_FIELD(CFG_COMPUTE1, OUT_ZP, cfg.output_zeropoint) |
                        REG_FIELD(CFG_COMPUTE1, OUT_SCALE, cfg.quant_scale) |
                        REG_FIELD(CFG_COMPUTE1, OUT_SHIFT, cfg.quant_scaleshift);
    reg_write64_cached(RegOffset::CFG_COMPUTE_2, val_cfg2, &shadow.compute_cfg2);

    // 3. SA Input A
    uint64_t val_input_a = REG_FIELD(SA_IN_A, ADDR, cfg.input_a_addr) |
                           REG_FIELD(SA_IN_A, COL, cfg.input_a_col_num_m1) |
                           REG_FIELD(SA_IN_A, ROW, cfg.input_a_row_num_m1) |
                           REG_FIELD(SA_IN_A, STRIDE, cfg.input_a_stride);
    reg_write64(RegOffset::SA_INPUT_A, val_input_a);

    // 4. SA Input B
    uint64_t val_input_b = REG_FIELD(SA_IN_B, ADDR, cfg.input_b_addr) |
                           REG_FIELD(SA_IN_B, COL, cfg.input_b_col_num_m1) |
                           REG_FIELD(SA_IN_B, ROW, cfg.input_b_row_num_m1) |
                           REG_FIELD(SA_IN_B, STRIDE, cfg.input_b_stride);
    reg_write64(RegOffset::SA_INPUT_B, val_input_b);

    // 5. Accumulator 1
    uint64_t val_acc1 = REG_FIELD(CFG_ACCU0, BIASPSUM_ADDR, cfg.biaspsum_addr) |
                        REG_FIELD(CFG_ACCU0, STRIDE, cfg.biaspsum_stride) |
                        REG_FIELD(CFG_ACCU0, WIDTH, cfg.biaspsum_width) |
                        REG_FIELD(CFG_ACCU0, HEIGHT, cfg.biaspsum_height);
    reg_write64_cached(RegOffset::CFG_ACCU_1, val_acc1, &shadow.accu_cfg1);

    // 6. Accumulator 2
    uint64_t val_acc2 = REG_FIELD(CFG_ACCU1, OUT_ADDR, cfg.output_addr) |
                        REG_FIELD(CFG_ACCU1, OUT_STRIDE, cfg.output_stride) |
                        REG_FIELD(CFG_ACCU1, IS_ACCU, cfg.is_accumulate) |
                        REG_FIELD(CFG_ACCU1, RELU, cfg.relu_enable) |
                        REG_FIELD(CFG_ACCU1, RELU_TYPE, cfg.relu_type) |
                        REG_FIELD(CFG_ACCU1, IS_BIAS, cfg.is_bias);
    reg_write64_cached(RegOffset::CFG_ACCU_2, val_acc2, &shadow.accu_cfg2);

    // 7. Start SA
    reg_write(RegOffset::START, BIT_START_SA);
    NPU_TIMER_SECTION_END()

    NPU_TIMER_SECTION_BEGIN("run_conv(wait_irq)")
    wait_irq();
    NPU_TIMER_SECTION_END()
}

int NpuRuntime::run_conv_tile(const NpuConvTileConfig& cfg) {
    if (!g_npu_runtime) return -1;

    constexpr int SA_SIZE = 32;
    constexpr int ACC_ELEM_SIZE = 4;

    auto is_mult32 = [](int32_t v) { return (v % SA_SIZE) == 0; };
    auto fits_u16 = [](int32_t v) { return (v >= 0) && (v <= 0xFFFF); };

    // --- 参数校验 ---
    if (cfg.c_in <= 0) return -1;
    if (cfg.k_h <= 0 || cfg.k_w <= 0 || cfg.stride <= 0 || cfg.dilation <= 0) return -1;
    if (cfg.t_h_out <= 0 || cfg.t_w_out <= 0 || cfg.t_cin <= 0 || cfg.t_cout <= 0) return -1;
    if (cfg.t_cout > SA_SIZE) return -1;
    if (cfg.i_cin < 0) return -1;
    if (cfg.i_cin + cfg.t_cin > cfg.c_in) return -1;
    if (!fits_u16(cfg.t_w_out)) return -1;

    if (cfg.c_in < SA_SIZE) {
        // 允许 cin < 32：仅支持单个 cin block
        if (cfg.i_cin != 0) return -1;
        if (cfg.t_cin != cfg.c_in) return -1;
    } else {
        if (!is_mult32(cfg.c_in) || !is_mult32(cfg.t_cin) || !is_mult32(cfg.t_cout)) return -1;
        if ((cfg.i_cin % SA_SIZE) != 0) return -1;
    }

    // --- IFM 感受野计算（无 padding，编译器已处理好） ---
    int32_t t_h_in = (cfg.t_h_out - 1) * cfg.stride +
                     (cfg.k_h - 1) * cfg.dilation + 1;
    int32_t t_w_in = (cfg.t_w_out - 1) * cfg.stride +
                     (cfg.k_w - 1) * cfg.dilation + 1;
    if (t_h_in <= 0 || t_w_in <= 0) return -1;

    const int32_t cin_block_size = (cfg.c_in < SA_SIZE) ? cfg.c_in : SA_SIZE;
    const int32_t weight_block_stride = cfg.k_h * cfg.k_w * cfg.t_cout * cin_block_size;
    if (weight_block_stride <= 0) return -1;

    // ---------------------------------------------------
    // Micro-Tile Compute
    // ---------------------------------------------------
    // IFM / Weight / Bias 均由调用者预先 MVIN 到 SPM/ACC。
    // 本函数处理单个 cout block 的 j_h × j_w × j_cin 三层循环。
    // Padding 已由编译器在 IFM 数据中完成，此处全部为 0。
    int32_t t_hm_out = (cfg.t_h_out <= SA_SIZE) ? cfg.t_h_out : SA_SIZE;
    int32_t t_wm_out = (cfg.t_w_out * t_hm_out <= SA_SIZE) ?
                       cfg.t_w_out : (SA_SIZE / t_hm_out);
    if (t_hm_out <= 0 || t_wm_out <= 0) return -1;

    for (int32_t j_h = 0; j_h < cfg.t_h_out; j_h += t_hm_out) {
        int32_t h_this_block = ((j_h + t_hm_out) <= cfg.t_h_out) ?
                               t_hm_out : (cfg.t_h_out - j_h);

        for (int32_t j_w = 0; j_w < cfg.t_w_out; j_w += t_wm_out) {
            int32_t w_this_block = ((j_w + t_wm_out) <= cfg.t_w_out) ?
                                   t_wm_out : (cfg.t_w_out - j_w);

            for (int32_t j_cin = 0; j_cin < cfg.t_cin; j_cin += SA_SIZE) {
                bool is_first_cin_global = (cfg.i_cin == 0) && (j_cin == 0);
                bool is_last_cin_global = (cfg.i_cin + cfg.t_cin >= cfg.c_in) &&
                                          (j_cin + SA_SIZE >= cfg.t_cin);
                bool is_only_cin_global = is_first_cin_global && is_last_cin_global;
                
                ConvConfig conv_cfg = {};

                // Padding 由编译器完成，硬件侧全部置 0
                conv_cfg.pad_top = 0;
                conv_cfg.pad_bottom = 0;
                conv_cfg.pad_left = 0;
                conv_cfg.pad_right = 0;
                conv_cfg.pad_mode = 0;

                conv_cfg.weight_shape_m1 = static_cast<uint8_t>(cfg.k_h - 1);
                conv_cfg.weight_stride_m1 = static_cast<uint8_t>(cfg.stride - 1);
                conv_cfg.weight_dilation_m1 = static_cast<uint8_t>(cfg.dilation - 1);
                conv_cfg.is_group_conv = cfg.is_group_conv;

                conv_cfg.int_type = 0;
                conv_cfg.op_type = 1;
                conv_cfg.dataflow_mode = 0;

                // 根据 cin 在全局维度的位置决定累加/输出策略
                if (is_only_cin_global) {
                    // 只有一轮 cin：按需加 bias → 量化输出到 SPM
                    conv_cfg.is_accumulate =  (cfg.bias_enable == false) ? 0 : 1;//第一轮cin如果不加bias就取消累加
                    conv_cfg.accout_dest = 0;
                    conv_cfg.is_bias = cfg.bias_enable ? 1 : 0;
                } else if (is_first_cin_global) {
                    // 第一轮 cin：按需加 bias → 暂存到 ACC
                    conv_cfg.is_accumulate = 1;
                    conv_cfg.accout_dest = 1;
                    conv_cfg.is_bias = cfg.bias_enable ? 1 : 0;
                } else if (is_last_cin_global) {
                    // 最后一轮 cin：累加 psum → 量化输出到 SPM
                    conv_cfg.is_accumulate = 1;
                    conv_cfg.accout_dest = 0;
                    conv_cfg.is_bias = 0;
                } else {
                    // 中间轮 cin：累加 psum → 暂存到 ACC
                    conv_cfg.is_accumulate = 1;
                    conv_cfg.accout_dest = 1;
                    conv_cfg.is_bias = 0;
                }

                // 对称量化，zeropoint 全 0
                conv_cfg.input_a_zeropoint = 0;
                conv_cfg.input_b_zeropoint = 0;

                int32_t cin_blk_idx = j_cin / SA_SIZE;
                int32_t cin_this_block = cfg.t_cin - j_cin;
                if (cin_this_block > SA_SIZE) cin_this_block = SA_SIZE;

                // 无 padding，micro-tile 的 IFM 感受野直接由 h/w_this_block 推导
                int32_t in_h_block = (h_this_block - 1) * cfg.stride +
                                     (cfg.k_h - 1) * cfg.dilation + 1;
                int32_t in_w_block = (w_this_block - 1) * cfg.stride +
                                     (cfg.k_w - 1) * cfg.dilation + 1;
                if (in_h_block <= 0 || in_w_block <= 0) return -1;
                if (!fits_u16(in_w_block - 1) || (in_h_block - 1) > 0xFF) return -1;
                if (!fits_u16(t_w_in)) return -1;

                // 无 padding，IFM 起始位置直接由 j_h/j_w 和 stride 确定
                int32_t in_h_start = j_h * cfg.stride;
                int32_t in_w_start = j_w * cfg.stride;

                int32_t ifm_spm_offset = (cin_blk_idx * t_h_in * t_w_in +
                                          in_h_start * t_w_in +
                                          in_w_start) * cin_block_size;
                conv_cfg.input_a_addr = cfg.sram_addr_ifm + static_cast<uint32_t>(ifm_spm_offset);
                conv_cfg.input_a_col_num_m1 = static_cast<uint16_t>(in_w_block - 1);
                conv_cfg.input_a_row_num_m1 = static_cast<uint8_t>(in_h_block - 1);//这个位宽是5bit，所以input的行数不能大于32,也就是oh不能太大
                conv_cfg.input_a_stride = static_cast<uint16_t>(t_w_in);

                int32_t wgt_spm_offset = cin_blk_idx * weight_block_stride;
                conv_cfg.input_b_addr = cfg.sram_addr_weight + static_cast<uint32_t>(wgt_spm_offset);
                conv_cfg.input_b_col_num_m1 = static_cast<uint8_t>(cfg.t_cout - 1);
                conv_cfg.input_b_row_num_m1 = static_cast<uint16_t>(cin_this_block - 1);
                conv_cfg.input_b_stride = static_cast<uint16_t>(cfg.t_cout);

                // psum / bias 地址：bias 由调用者 MVIN 到 ACC bias 寄存器，
                // is_bias=1 时硬件自动读 bias 寄存器，biaspsum_addr 不影响；
                // is_bias=0 时从 acc_addr_psum 读取已有部分和。
                int32_t psum_offset = (j_h * cfg.t_w_out + j_w) * SA_SIZE;
                conv_cfg.biaspsum_addr = conv_cfg.is_bias ?
                    0 :
                    (cfg.acc_addr_psum + psum_offset * ACC_ELEM_SIZE);
                conv_cfg.biaspsum_stride = static_cast<uint16_t>(cfg.t_w_out);
                conv_cfg.biaspsum_width = static_cast<uint8_t>(w_this_block);
                conv_cfg.biaspsum_height = static_cast<uint8_t>(h_this_block);

                int32_t ofm_spm_offset = (j_h * cfg.t_w_out + j_w) * SA_SIZE;
                if (is_last_cin_global) {
                    conv_cfg.output_addr = cfg.sram_addr_ofm + static_cast<uint32_t>(ofm_spm_offset);
                } else {
                    conv_cfg.output_addr = cfg.acc_addr_psum + psum_offset * ACC_ELEM_SIZE;
                }
                conv_cfg.output_stride = static_cast<uint16_t>(cfg.t_w_out);

                conv_cfg.relu_enable = is_last_cin_global ? cfg.relu_enable : 0;
                conv_cfg.relu_type = cfg.relu_type;

                // 对称量化，output_zeropoint = 0
                conv_cfg.output_zeropoint = 0;
                conv_cfg.quant_scale = cfg.quant_scale;
                conv_cfg.quant_scaleshift = cfg.quant_scaleshift;

                g_npu_runtime->run_conv(conv_cfg);
            }
        }
    }

    return 0;
}

void NpuRuntime::run_gemm(const GemmConfig& cfg) {
    NPU_TIMER_TOTAL("run_gemm");
    ScopedStageTimer profileTimer(ProfileStage::Compute, ProfileCallCounter::Compute);
    
    NPU_TIMER_SECTION_BEGIN("run_gemm(pre_reg)")
    NPU_LOG("Running GEMM (DataFlow=%d)", cfg.dataflow);
    NPU_TIMER_SECTION_END()

    NPU_TIMER_SECTION_BEGIN("run_gemm(reg_write)")
    // 1. Compute Config 1
    uint64_t val_cfg1 = REG_FIELD(CFG_COMPUTE0, DATAFLOW, cfg.dataflow) |
                        REG_FIELD(CFG_COMPUTE0, INT_TYPE, cfg.int_type) |
                        REG_FIELD(CFG_COMPUTE0, OPTYPE, cfg.optype) |
                        REG_FIELD(CFG_COMPUTE0, ACCOUT_DEST, cfg.accout_dest) |
                        REG_FIELD(CFG_COMPUTE0, ASYMMETRIC_ACTIVATIONS, cfg.asymmetric_activations) |
                        REG_FIELD(CFG_COMPUTE0, INPUTA_ZP, cfg.input_a_zeropoint) |
                        REG_FIELD(CFG_COMPUTE0, INPUTB_ZP, cfg.input_b_zeropoint);
    reg_write64_cached(RegOffset::CFG_COMPUTE_1, val_cfg1, &shadow.compute_cfg1);

    // 2. Compute Config 2
    uint64_t val_cfg2 = REG_FIELD(CFG_COMPUTE1, OUT_ZP, cfg.output_zeropoint) |
                        REG_FIELD(CFG_COMPUTE1, OUT_SCALE, cfg.output_scale) |
                        REG_FIELD(CFG_COMPUTE1, OUT_SHIFT, cfg.output_scaleshift);
    reg_write64_cached(RegOffset::CFG_COMPUTE_2, val_cfg2, &shadow.compute_cfg2);

    // 3. Accumulator 1
    uint64_t val_acc1 = REG_FIELD(CFG_ACCU0, BIASPSUM_ADDR, cfg.biaspsum_addr) |
                        REG_FIELD(CFG_ACCU0, STRIDE, cfg.biaspsum_stride) |
                        REG_FIELD(CFG_ACCU0, WIDTH, cfg.biaspsum_width) |
                        REG_FIELD(CFG_ACCU0, HEIGHT, cfg.biaspsum_height);
    reg_write64_cached(RegOffset::CFG_ACCU_1, val_acc1, &shadow.accu_cfg1);

    // 4. Accumulator 2
    uint64_t val_acc2 = REG_FIELD(CFG_ACCU1, OUT_ADDR, cfg.output_addr) |
                        REG_FIELD(CFG_ACCU1, OUT_STRIDE, cfg.output_stride) |
                        REG_FIELD(CFG_ACCU1, IS_ACCU, cfg.isaccu) |
                        REG_FIELD(CFG_ACCU1, RELU, cfg.relu) |
                        REG_FIELD(CFG_ACCU1, RELU_TYPE, cfg.relu_type) |
                        REG_FIELD(CFG_ACCU1, IS_BIAS, cfg.is_bias);
    reg_write64_cached(RegOffset::CFG_ACCU_2, val_acc2, &shadow.accu_cfg2);

    // 5. SA Input A
    uint64_t val_input_a = REG_FIELD(SA_IN_A, ADDR, cfg.input_a_addr) |
                           REG_FIELD(SA_IN_A, COL, cfg.input_a_col_num) |
                           REG_FIELD(SA_IN_A, ROW, cfg.input_a_row_num) |
                           REG_FIELD(SA_IN_A, STRIDE, cfg.input_a_stride);
    reg_write64(RegOffset::SA_INPUT_A, val_input_a);

    // 6. SA Input B
    uint64_t val_input_b = REG_FIELD(SA_IN_B, ADDR, cfg.input_b_addr) |
                           REG_FIELD(SA_IN_B, COL, cfg.input_b_col_num) |
                           REG_FIELD(SA_IN_B, ROW, cfg.input_b_row_num) |
                           REG_FIELD(SA_IN_B, STRIDE, cfg.input_b_stride);
    reg_write64(RegOffset::SA_INPUT_B, val_input_b);
    g_last_op_ctx = {
        "GEMM",
        val_input_a,
        val_input_b,
        val_cfg1,
        val_cfg2,
        BIT_START_SA
    };

    // 7. Start SA
    reg_write(RegOffset::START, BIT_START_SA);
    NPU_TIMER_SECTION_END()

    NPU_TIMER_SECTION_BEGIN("run_gemm(wait_irq)")
    wait_irq();
    NPU_TIMER_SECTION_END()
}

void NpuRuntime::run_matadd(const MataddConfig& cfg) {
    NPU_TIMER_TOTAL("run_matadd");
    ScopedStageTimer profileTimer(ProfileStage::Compute, ProfileCallCounter::Compute);
    
    NPU_TIMER_SECTION_BEGIN("run_matadd(pre_reg)")
    NPU_LOG("Running MATADD (A=0x%X, B=0x%X, Out=0x%X, ColM1=%d, RowM1=%d)", 
            cfg.input_a_addr, cfg.input_b_addr, cfg.output_addr,
            cfg.col_num_m1, cfg.row_num_m1);
    NPU_TIMER_SECTION_END()
    
    NPU_TIMER_SECTION_BEGIN("run_matadd(reg_write)")
    // 1. Compute Config 2 - Quantization output parameters
    uint64_t val_cfg2 = REG_FIELD(CFG_COMPUTE1, OUT_ZP, cfg.output_zeropoint) |
                        REG_FIELD(CFG_COMPUTE1, OUT_SCALE, cfg.output_scale) |
                        REG_FIELD(CFG_COMPUTE1, OUT_SHIFT, cfg.output_scaleshift);
    reg_write64_cached(RegOffset::CFG_COMPUTE_2, val_cfg2, &shadow.compute_cfg2);
    
    // 2. MATADD CTRL0 - Input A and B addresses
    uint64_t val_ctrl0 = REG_FIELD(MATADD_CTRL0, A_ADDR, cfg.input_a_addr) |
                         REG_FIELD(MATADD_CTRL0, B_ADDR, cfg.input_b_addr);
    reg_write64(RegOffset::MATADD_CTRL_0, val_ctrl0);
    
    // 3. MATADD CTRL1 - Output address and dimensions
    uint64_t val_ctrl1 = REG_FIELD(MATADD_CTRL1, OUT_ADDR, cfg.output_addr) |
                         REG_FIELD(MATADD_CTRL1, COL, cfg.col_num_m1) |
                         REG_FIELD(MATADD_CTRL1, ROW, cfg.row_num_m1);
    reg_write64(RegOffset::MATADD_CTRL_1, val_ctrl1);
    
    // 4. Start MATADD
    reg_write(RegOffset::START, BIT_START_MATADD);
    NPU_TIMER_SECTION_END()
    
    NPU_TIMER_SECTION_BEGIN("run_matadd(wait_irq)")
    wait_irq();
    NPU_TIMER_SECTION_END()
}

void NpuRuntime::run_transpose(const TransposeConfig& cfg) {
    NPU_TIMER_TOTAL("run_transpose");
    ScopedStageTimer profileTimer(ProfileStage::Layout, ProfileCallCounter::Layout);
    
    NPU_TIMER_SECTION_BEGIN("run_transpose(pre_reg)")
    NPU_LOG("Running TRANSPOSE (In=0x%X, Out=0x%X, Col=%d, Row=%d)", 
            cfg.input_sram_addr, cfg.output_sram_addr,
            cfg.col_num, cfg.row_num);
    NPU_TIMER_SECTION_END()
    
    NPU_TIMER_SECTION_BEGIN("run_transpose(reg_write)")
    // Transpose 使用 SFU 模块实现，操作码为 SFU_OP_TRANSPOSE (8)
    // 不需要量化参数，直接配置输入输出地址和尺寸
    
    // 1. SFU Config 0 - 设置操作码为 TRANSPOSE
    uint64_t val_cfg1 = REG_FIELD(CFG_SFU0, OP, SFU_OP_TRANSPOSE) |
                        REG_FIELD(CFG_SFU0, INT_TYPE, 0) |  // int8
                        REG_FIELD(CFG_SFU0, IS_QUANT, 0) |  // 不量化
                        REG_FIELD(CFG_SFU0, TRANSPOSE_OUT_IS_PADDING_ROW, cfg.out_padding_row) |
                        REG_FIELD(CFG_SFU0, TRANSPOSE_OUT_IS_PADDING_COL, cfg.out_padding_col) |
                        REG_FIELD(CFG_SFU0, OUT_ZP, 0) |
                        REG_FIELD(CFG_SFU0, IN_ZP, 0);
    reg_write64_cached(RegOffset::CFG_SFU_1, val_cfg1, &shadow.sfu_cfg1);
    
    // 2. SFU Config 1 - Scale 参数置 0
    // 同样不用配置量化参数
    // uint64_t val_cfg2 = REG_FIELD(CFG_SFU1, IN_SCALE, 0) |
    //                     REG_FIELD(CFG_SFU1, IN_SHIFT, 0) |
    //                     REG_FIELD(CFG_SFU1, OUT_SCALE, 0) |
    //                     REG_FIELD(CFG_SFU1, OUT_SHIFT, 0);
    // reg_write64_cached(RegOffset::CFG_SFU_2, val_cfg2, &shadow.sfu_cfg2);
    
    // 3. SFU Input - 输入地址和尺寸
    uint64_t val_input = REG_FIELD(SFU_EXE0, IN_ADDR, cfg.input_sram_addr) |
                         REG_FIELD(SFU_EXE0, COL, cfg.col_num) |
                         REG_FIELD(SFU_EXE0, ROW, cfg.row_num);
    reg_write64(RegOffset::SFU_INPUT, val_input);
    
    // 4. SFU Output - 输出地址
    uint64_t val_output = REG_FIELD(SFU_EXE1, OUT_ADDR, cfg.output_sram_addr);
    reg_write64(RegOffset::SFU_OUTPUT, val_output);
    
    // 5. Start SFU
    reg_write(RegOffset::START, BIT_START_SFU);
    NPU_TIMER_SECTION_END()
    
    NPU_TIMER_SECTION_BEGIN("run_transpose(wait_irq)")
    wait_irq();
    NPU_TIMER_SECTION_END()
}

void NpuRuntime::run_resample(const ResampleConfig& cfg) {
    NPU_TIMER_TOTAL("run_resample");
    ScopedStageTimer profileTimer(ProfileStage::Compute, ProfileCallCounter::Compute);
    
    NPU_TIMER_SECTION_BEGIN("run_resample(pre_reg)")
    NPU_LOG("Running RESAMPLE (Type=%d, Op=%d, In=0x%X, Out=0x%X, Col=%d, Row=%d)", 
            cfg.resample_type, cfg.resample_op,
            cfg.input_sram_addr, cfg.output_sram_addr,
            cfg.input_col_num, cfg.input_row_num);
    NPU_TIMER_SECTION_END()
    
    NPU_TIMER_SECTION_BEGIN("run_resample(reg_write)")
    // Resample 使用 SFU 模块实现
    // 根据 resample_type 和 resample_op 构造 SFU 操作码
    // 硬件使用 cfg_sfu_op 的低 2 位作为 resample_type，bit[2] 作为 resample_op
    // SFU 操作码:
    //   3 = DOWNSAMPLE_MAX (type=0, op=0 -> 最大值下采样)
    //   4 = DOWNSAMPLE_AVG (type=0, op=1 -> 平均值下采样)
    //   5 = UPSAMPLE_NEAREST (type=1, op=0 -> 最近邻上采样)

    uint8_t sfu_op;
    if (cfg.resample_type == RESAMPLE_TYPE_DOWNSAMPLE ||
        cfg.resample_type == RESAMPLE_TYPE_POOLING) {
        // 下采样或池化
        sfu_op = (cfg.resample_op == RESAMPLE_OP_MAX) ?
                 SFU_OP_DOWNSAMPLE_MAX : SFU_OP_DOWNSAMPLE_AVG;
    } else {
        // 上采样 (当前仅支持最近邻)
        sfu_op = SFU_OP_UPSAMPLE_NEAREST;
    }

    // 1. SFU Config 0 - 设置操作码
    uint64_t val_cfg1 = REG_FIELD(CFG_SFU0, OP, sfu_op) |
                        REG_FIELD(CFG_SFU0, INT_TYPE, 0) |  // int8
                        REG_FIELD(CFG_SFU0, IS_QUANT, 0) |  // 不量化
                        REG_FIELD(CFG_SFU0, OUT_ZP, 0) |
                        REG_FIELD(CFG_SFU0, IN_ZP, 0);
    reg_write64_cached(RegOffset::CFG_SFU_1, val_cfg1, &shadow.sfu_cfg1);

    // 2. SFU Config 1 - Scale 参数置 0
    // 由于不涉及量化计算，量化参数随便是什么值，这里直接不写寄存器，减少寄存器访问
    // uint64_t val_cfg2 = REG_FIELD(CFG_SFU1, IN_SCALE, 0) |
    //                     REG_FIELD(CFG_SFU1, IN_SHIFT, 0) |
    //                     REG_FIELD(CFG_SFU1, OUT_SCALE, 0) |
    //                     REG_FIELD(CFG_SFU1, OUT_SHIFT, 0);
    // reg_write64_cached(RegOffset::CFG_SFU_2, val_cfg2, &shadow.sfu_cfg2);

    constexpr uint32_t HW_MAX_RESAMPLE_ROWS = 2048; // 实测硬件稳定上限（real rows）
    const uint32_t in_cols = static_cast<uint32_t>(cfg.input_col_num) + 1;
    const uint32_t total_rows = static_cast<uint32_t>(cfg.input_row_num) + 1;

    auto output_cols_from_input_cols = [&](uint32_t cols) -> uint32_t {
        if (cfg.resample_type == RESAMPLE_TYPE_UPSAMPLE) {
            return cols << 1;
        }
        // downsample / pooling
        return (cols + 1) >> 1;
    };

    auto output_rows_from_input_rows = [&](uint32_t rows) -> uint32_t {
        if (cfg.resample_type == RESAMPLE_TYPE_UPSAMPLE) {
            return rows << 1;
        }
        // downsample / pooling
        return (rows + 1) >> 1;
    };

    uint32_t row_base = 0;      // 输入已处理 real-row 数
    uint32_t out_row_base = 0;  // 输出已生成 real-row 数

    while (row_base < total_rows) {
        uint32_t remain = total_rows - row_base;
        uint32_t chunk_rows = (remain > HW_MAX_RESAMPLE_ROWS) ? HW_MAX_RESAMPLE_ROWS : remain;

        // 对于 downsample/pooling，非最后一块需要偶数行，避免 2x2 跨块配对问题
        if ((cfg.resample_type == RESAMPLE_TYPE_DOWNSAMPLE || cfg.resample_type == RESAMPLE_TYPE_POOLING) &&
            (row_base + chunk_rows < total_rows) &&
            (chunk_rows & 1U)) {
            chunk_rows -= 1U;
        }

        if (chunk_rows == 0) {
            NPU_ERR("run_resample chunking failed: zero chunk rows (total_rows=%u, row_base=%u)",
                    total_rows, row_base);
            break;
        }

        uint32_t in_addr_chunk = cfg.input_sram_addr + row_base * in_cols;
        uint32_t out_cols = output_cols_from_input_cols(in_cols);
        uint32_t out_addr_chunk = cfg.output_sram_addr + out_row_base * out_cols;

        // 3. SFU Input - 输入地址和尺寸（按块）
        uint64_t val_input = REG_FIELD(SFU_EXE0, IN_ADDR, in_addr_chunk) |
                             REG_FIELD(SFU_EXE0, COL, cfg.input_col_num) |
                             REG_FIELD(SFU_EXE0, ROW, static_cast<uint16_t>(chunk_rows - 1));
        reg_write64(RegOffset::SFU_INPUT, val_input);

        // 4. SFU Output - 输出地址（按块）
        uint64_t val_output = REG_FIELD(SFU_EXE1, OUT_ADDR, out_addr_chunk);
        reg_write64(RegOffset::SFU_OUTPUT, val_output);

        // 5. Start SFU
        reg_write(RegOffset::START, BIT_START_SFU);

        NPU_TIMER_SECTION_END()

        NPU_TIMER_SECTION_BEGIN("run_resample(wait_irq)")
        wait_irq();
        NPU_TIMER_SECTION_END()

        NPU_TIMER_SECTION_BEGIN("run_resample(reg_write)")
        row_base += chunk_rows;
        out_row_base += output_rows_from_input_rows(chunk_rows);
    }

    NPU_TIMER_SECTION_END()
}

// ==========================================
// Layout Convert Implementation
// ==========================================

void NpuRuntime::run_nchw_to_nchwc32(const LayoutConvertConfig& cfg) {
    NPU_TIMER_TOTAL("run_nchw_to_nchwc32");

    if (cfg.sram_addr == cfg.output_addr) {
        NPU_ERR("Layout convert does not support in-place operation.");
        return;
    }
    if (cfg.n == 0 || cfg.c == 0 || cfg.h == 0 || cfg.w == 0) {
        NPU_ERR("Layout convert: invalid dimensions.");
        return;
    }

    uint32_t hw = static_cast<uint32_t>(cfg.h) * static_cast<uint32_t>(cfg.w);
    if (hw == 0 || hw > 65536) {
        NPU_ERR("Layout convert: H*W=%u exceeds max 65536.", hw);
        return;
    }

    if (cfg.c < 32) {
        // NCHW -> NHWC by transpose (C, HW) -> (HW, C)
        for (uint16_t n = 0; n < cfg.n; ++n) {
            uint32_t batch_offset = n * cfg.c * hw;
            TransposeConfig tcfg = {
                cfg.sram_addr + batch_offset,
                cfg.output_addr + batch_offset,
                static_cast<uint16_t>(hw - 1),
                static_cast<uint16_t>(cfg.c - 1),
                false,
                false
            };
            run_transpose(tcfg);
        }
        return;
    }

    uint16_t group_count = static_cast<uint16_t>((cfg.c + 31) / 32);
    for (uint16_t n = 0; n < cfg.n; ++n) {
        uint32_t batch_offset = n * cfg.c * hw;
        for (uint16_t g = 0; g < group_count; ++g) {
            uint32_t src_offset = batch_offset + g * 32u * hw;
            uint32_t dst_offset = batch_offset + g * hw * 32u;
            uint16_t rem = (cfg.c > g * 32u) ? static_cast<uint16_t>(cfg.c - g * 32u) : 0;
            if (rem == 0) {
                continue;
            }
            bool pad_col = rem < 32;
            uint16_t row_num = pad_col ? static_cast<uint16_t>(rem - 1) : static_cast<uint16_t>(32 - 1);
            TransposeConfig tcfg = {
                cfg.sram_addr + src_offset,
                cfg.output_addr + dst_offset,
                static_cast<uint16_t>(hw - 1),
                row_num,
                false,
                pad_col
            };
            run_transpose(tcfg);
        }
    }
}

void NpuRuntime::run_nchwc32_to_nchw(const LayoutConvertConfig& cfg) {
    NPU_TIMER_TOTAL("run_nchwc32_to_nchw");

    if (cfg.sram_addr == cfg.output_addr) {
        NPU_ERR("Layout convert does not support in-place operation.");
        return;
    }
    if (cfg.n == 0 || cfg.c == 0 || cfg.h == 0 || cfg.w == 0) {
        NPU_ERR("Layout convert: invalid dimensions.");
        return;
    }

    uint32_t hw = static_cast<uint32_t>(cfg.h) * static_cast<uint32_t>(cfg.w);
    if (hw == 0 || hw > 65536) {
        NPU_ERR("Layout convert: H*W=%u exceeds max 65536.", hw);
        return;
    }

    if (cfg.c < 32) {
        // NHWC -> NCHW by transpose (HW, C) -> (C, HW)
        for (uint16_t n = 0; n < cfg.n; ++n) {
            uint32_t batch_offset = n * cfg.c * hw;
            TransposeConfig tcfg = {
                cfg.sram_addr + batch_offset,
                cfg.output_addr + batch_offset,
                static_cast<uint16_t>(cfg.c - 1),
                static_cast<uint16_t>(hw - 1),
                false,
                false
            };
            run_transpose(tcfg);
        }
        return;
    }

    if (cfg.c % 32 != 0) {
        NPU_ERR("Layout convert: C=%u must be a multiple of 32 when C>=32.", cfg.c);
        return;
    }

    uint16_t group_count = cfg.c / 32;
    for (uint16_t n = 0; n < cfg.n; ++n) {
        uint32_t batch_offset = n * cfg.c * hw;
        for (uint16_t g = 0; g < group_count; ++g) {
            uint32_t src_offset = batch_offset + g * hw * 32u;
            uint32_t dst_offset = batch_offset + g * 32u * hw;
            TransposeConfig tcfg = {
                cfg.sram_addr + src_offset,
                cfg.output_addr + dst_offset,
                static_cast<uint16_t>(32 - 1),
                static_cast<uint16_t>(hw - 1),
                false,
                false
            };
            run_transpose(tcfg);
        }
    }
}

// ==========================================
// Allocator Implementation
// ==========================================

void NpuRuntime::init_allocator() {
    if (!data_virt_base) return;
    free_list_head = (BlockHeader*)data_virt_base;
    if (data_map_size <= sizeof(BlockHeader)) {
        free_list_head = nullptr;
        return;
    }
    free_list_head->size = data_map_size - sizeof(BlockHeader);
    free_list_head->is_free = true;
    free_list_head->next = nullptr;
    free_list_head->prev = nullptr;
    NPU_LOG("Allocator initialized. Total size: 0x%X", data_map_size);
}

void* NpuRuntime::alloc(size_t size) {
    size_t aligned_size = (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    BlockHeader* curr = free_list_head;
    
    while (curr) {
        if (curr->is_free && curr->size >= aligned_size) {
            if (curr->size >= aligned_size + sizeof(BlockHeader) + ALIGNMENT) {
                // Split
                BlockHeader* new_block = (BlockHeader*)((uint8_t*)curr + sizeof(BlockHeader) + aligned_size);
                new_block->size = curr->size - aligned_size - sizeof(BlockHeader);
                new_block->is_free = true;
                new_block->next = curr->next;
                new_block->prev = curr;
                if (curr->next) curr->next->prev = new_block;
                
                curr->next = new_block;
                curr->size = aligned_size;
            }
            curr->is_free = false;
            NPU_LOG("Allocated %zu bytes at offset 0x%lX", size, (uint8_t*)curr - (uint8_t*)data_virt_base);
            return (void*)((uint8_t*)curr + sizeof(BlockHeader));
        }
        curr = curr->next;
    }
    NPU_ERR("NPU OOM: Requested %zu", size);
    return nullptr;
}

void NpuRuntime::free(void* ptr) {
    if (!ptr) return;
    BlockHeader* block = (BlockHeader*)((uint8_t*)ptr - sizeof(BlockHeader));
    if (block->is_free) return;
    
    NPU_LOG("Freeing block at offset 0x%lX", (uint8_t*)ptr - (uint8_t*)data_virt_base);
    block->is_free = true;
    coalesce(block);
}

void NpuRuntime::coalesce(BlockHeader* block) {
    if (block->next && block->next->is_free) {
        block->size += sizeof(BlockHeader) + block->next->size;
        block->next = block->next->next;
        if (block->next) block->next->prev = block;
    }
    if (block->prev && block->prev->is_free) {
        block->prev->size += sizeof(BlockHeader) + block->size;
        block->prev->next = block->next;
        if (block->next) block->next->prev = block->prev;
    }
}

// ==========================================
// C Interface Implementation
// ==========================================

extern "C" {

int npu_init() {
    NPU_CAPI_LOG("npu_init()");
    if (g_npu_runtime) return 0;
    g_npu_runtime = new NpuRuntime();
    if (!g_npu_runtime->init()) {
        delete g_npu_runtime; g_npu_runtime = nullptr;
        return -1;
    }
    return 0;
}

void npu_destroy() {
    NPU_CAPI_LOG("npu_destroy()");
    dumpProfilerReport(nullptr);
    if (g_npu_runtime) { delete g_npu_runtime; g_npu_runtime = nullptr; }
}

void npu_reset() { // <--- [新增]
    NPU_CAPI_LOG("npu_reset()");
    if (g_npu_runtime) {
        g_npu_runtime->reset();
    }
}

void npu_profile_begin(int64_t layer_id) {
    NPU_CAPI_LOG("npu_profile_begin(layer_id=%lld)", (long long)layer_id);
    beginLayerProfile(layer_id);
}

void _mlir_ciface_npu_profile_begin(int64_t layer_id) {
    npu_profile_begin(layer_id);
}

void npu_profile_end(int64_t layer_id) {
    NPU_CAPI_LOG("npu_profile_end(layer_id=%lld)", (long long)layer_id);
    endLayerProfile(layer_id);
}

void _mlir_ciface_npu_profile_end(int64_t layer_id) {
    npu_profile_end(layer_id);
}

void npu_profile_dump(const char* path) {
    NPU_CAPI_LOG("npu_profile_dump(path=%s)", path ? path : "(null)");
    dumpProfilerReport(path);
}

void npu_profile_reset_summary() {
    NPU_CAPI_LOG("npu_profile_reset_summary()");
    if (isProfilingDisabled()) {
        return;
    }

    ProfilerState & state = getProfilerState();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.layers.clear();
    g_activeLayer = ActiveLayerFrame{};
}

void npu_profile_get_summary(struct npu_profile_runtime_summary * out) {
    if (out == nullptr) {
        return;
    }

    *out = {};
    if (isProfilingDisabled()) {
        return;
    }

    ProfilerState & state = getProfilerState();
    std::lock_guard<std::mutex> lock(state.mutex);
    out->layer_count = state.layers.size();

    for (const auto & entry : state.layers) {
        const LayerProfileRecord & record = entry.second;
        out->layer_invocations += record.invocations;
        out->total_ns += record.totalNs;
        out->dma_in_ns += record.dmaInNs;
        out->compute_ns += record.computeNs;
        out->dma_out_ns += record.dmaOutNs;
        out->layout_ns += record.layoutNs;
        out->wait_irq_ns += record.waitIrqNs;
        out->mvin_calls += record.mvinCalls;
        out->compute_calls += record.computeCalls;
        out->mvout_calls += record.mvoutCalls;
        out->layout_calls += record.layoutCalls;
    }
}

void* npu_mem_alloc(size_t size) {
    NPU_CAPI_LOG("npu_mem_alloc(size=%zu)", size);
    if (!g_npu_runtime && npu_init() < 0) return nullptr;
    return g_npu_runtime->alloc(size);
}

void npu_mem_free(void* ptr) {
    NPU_CAPI_LOG("npu_mem_free(ptr=%p)", ptr);
    if (g_npu_runtime) g_npu_runtime->free(ptr);
}

void npu_dma_mvin(
    void* host_ptr, uint32_t sram_addr, uint32_t col_num, uint32_t row_num,
    uint16_t sram_stride, uint32_t dram_stride, uint8_t precision, uint8_t input_type,
    bool dest, bool is_bias, bool is_quant, uint32_t quant_zero, uint16_t quant_scale, uint16_t quant_shift
) {
    NPU_CAPI_LOG(
        "npu_dma_mvin(host_ptr=%p, sram_addr=0x%08X, col_num=%u, row_num=%u, sram_stride=%u, dram_stride=%u, precision=%u, input_type=%u, dest=%d, is_bias=%d, is_quant=%d, quant_zero=%u, quant_scale=%u, quant_shift=%d)",
        host_ptr,
        sram_addr,
        (unsigned)col_num,
        (unsigned)row_num,
        (unsigned)sram_stride,
        dram_stride,
        (unsigned)precision,
        (unsigned)input_type,
        (int)dest,
        (int)is_bias,
        (int)is_quant,
        quant_zero,
        (unsigned)quant_scale,
        (int)static_cast<int16_t>(quant_shift));
    if (g_npu_runtime) {
        MvinConfig cfg = {host_ptr, sram_addr, col_num, row_num, sram_stride, dram_stride,
                          precision, input_type, dest, is_bias, is_quant, quant_zero, quant_scale, quant_shift};
        g_npu_runtime->run_mvin(cfg);
    }
}

void npu_dma_mvout(
    void* host_ptr, uint32_t sram_addr, uint32_t col_num, uint32_t row_num,
    uint16_t sram_stride, uint32_t dram_stride, uint8_t precision, uint8_t output_type,
    bool source, bool is_quant, uint32_t quant_zero, uint32_t f32_scale
) {
    NPU_CAPI_LOG(
        "npu_dma_mvout(host_ptr=%p, sram_addr=0x%08X, col_num=%u, row_num=%u, sram_stride=%u, dram_stride=%u, precision=%u, output_type=%u, source=%d, is_quant=%d, quant_zero=%u, f32_scale=0x%08X)",
        host_ptr,
        sram_addr,
        (unsigned)col_num,
        (unsigned)row_num,
        (unsigned)sram_stride,
        dram_stride,
        (unsigned)precision,
        (unsigned)output_type,
        (int)source,
        (int)is_quant,
        quant_zero,
        (unsigned)f32_scale);
    if (g_npu_runtime) {
        MvoutConfig cfg = {host_ptr, sram_addr, col_num, row_num, sram_stride, dram_stride,
                           precision, output_type, source, is_quant, quant_zero, f32_scale};
        g_npu_runtime->run_mvout(cfg);
    }
}

void npu_dma_mvin_async(uint32_t dma_id, const MvinConfig* cfg) {
    NPU_CAPI_LOG("npu_dma_mvin_async(dma_id=%u, cfg=%p)", dma_id, cfg);
    if (!g_npu_runtime || !cfg) {
        return;
    }
    g_npu_runtime->run_mvin_async(dma_id, *cfg);
}

void npu_dma_mvout_async(uint32_t dma_id, const MvoutConfig* cfg) {
    NPU_CAPI_LOG("npu_dma_mvout_async(dma_id=%u, cfg=%p)", dma_id, cfg);
    if (!g_npu_runtime || !cfg) {
        return;
    }
    g_npu_runtime->run_mvout_async(dma_id, *cfg);
}

void npu_dma_wait_mvin(uint32_t dma_mask) {
    NPU_CAPI_LOG("npu_dma_wait_mvin(dma_mask=0x%X)", dma_mask);
    if (!g_npu_runtime) {
        return;
    }
    g_npu_runtime->wait_mvin(dma_mask);
}

void npu_dma_wait_mvout(uint32_t dma_mask) {
    NPU_CAPI_LOG("npu_dma_wait_mvout(dma_mask=0x%X)", dma_mask);
    if (!g_npu_runtime) {
        return;
    }
    g_npu_runtime->wait_mvout(dma_mask);
}

void npu_dma_double_mvin(const MvinConfig* dma0_cfg, const MvinConfig* dma1_cfg) {
    NPU_CAPI_LOG("npu_dma_double_mvin(dma0_cfg=%p, dma1_cfg=%p)", dma0_cfg, dma1_cfg);
    if (!g_npu_runtime || !dma0_cfg || !dma1_cfg) {
        return;
    }
    g_npu_runtime->run_double_mvin(*dma0_cfg, *dma1_cfg);
}

void npu_sfu_run(
    uint8_t op_type, uint8_t int_type, bool is_quant, uint32_t input_sram_addr,
    uint16_t input_col_num, uint16_t input_row_num, uint32_t output_sram_addr,
    uint32_t input_zeropoint, uint16_t output_zeropoint, uint16_t input_scale,
    uint16_t input_scale_shift, uint16_t output_scale, uint16_t output_scale_shift
) {
    NPU_CAPI_LOG(
        "npu_sfu_run(op_type=%u, int_type=%u, is_quant=%d, input_sram_addr=0x%08X, input_col_num=%u, input_row_num=%u, output_sram_addr=0x%08X, input_zeropoint=%u, output_zeropoint=%u, input_scale=%u, input_scale_shift=%d, output_scale=%u, output_scale_shift=%d)",
        (unsigned)op_type,
        (unsigned)int_type,
        (int)is_quant,
        input_sram_addr,
        (unsigned)input_col_num,
        (unsigned)input_row_num,
        output_sram_addr,
        input_zeropoint,
        (unsigned)output_zeropoint,
        (unsigned)input_scale,
        (int)static_cast<int16_t>(input_scale_shift),
        (unsigned)output_scale,
        (int)static_cast<int16_t>(output_scale_shift));
    if (g_npu_runtime) {
        SfuConfig cfg = {op_type, int_type, is_quant, input_sram_addr, input_col_num, input_row_num,
                         output_sram_addr, input_zeropoint, output_zeropoint, input_scale,
                         input_scale_shift, output_scale, output_scale_shift};
        g_npu_runtime->run_sfu(cfg);
    }
}

void npu_conv_run(
    uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left, uint8_t pad_right, uint8_t pad_mode,
    uint8_t weight_shape_m1, uint8_t weight_stride_m1, uint8_t weight_dilation_m1, bool is_group_conv,
    uint8_t int_type, uint8_t op_type, bool dataflow_mode, bool accout_dest,
    uint16_t input_a_zeropoint, uint16_t input_b_zeropoint,
    uint32_t input_a_addr, uint16_t input_a_col_num_m1, uint8_t input_a_row_num_m1, uint16_t input_a_stride,
    uint32_t input_b_addr, uint8_t input_b_col_num_m1, uint16_t input_b_row_num_m1, uint16_t input_b_stride,
    uint8_t biaspsum_width, uint8_t biaspsum_height, uint32_t biaspsum_addr, uint16_t biaspsum_stride,
    uint32_t output_addr, uint16_t output_stride, bool is_accumulate, bool relu_enable, uint8_t relu_type,
    bool is_bias, uint32_t output_zeropoint, uint16_t quant_scale, uint16_t quant_scaleshift
) {
    NPU_CAPI_LOG(
        "npu_conv_run(pad=[%u,%u,%u,%u], pad_mode=%u, weight_shape_m1=%u, weight_stride_m1=%u, weight_dilation_m1=%u, is_group_conv=%d, int_type=%u, op_type=%u, dataflow_mode=%d, accout_dest=%d, input_a_zeropoint=%u, input_b_zeropoint=%u, input_a_addr=0x%08X, input_a_col_num_m1=%u, input_a_row_num_m1=%u, input_a_stride=%u, input_b_addr=0x%08X, input_b_col_num_m1=%u, input_b_row_num_m1=%u, input_b_stride=%u, biaspsum_width=%u, biaspsum_height=%u, biaspsum_addr=0x%08X, biaspsum_stride=%u, output_addr=0x%08X, output_stride=%u, is_accumulate=%d, relu_enable=%d, relu_type=%u, is_bias=%d, output_zeropoint=%u, quant_scale=%u, quant_scaleshift=%d)",
        (unsigned)pad_top,
        (unsigned)pad_bottom,
        (unsigned)pad_left,
        (unsigned)pad_right,
        (unsigned)pad_mode,
        (unsigned)weight_shape_m1,
        (unsigned)weight_stride_m1,
        (unsigned)weight_dilation_m1,
        (int)is_group_conv,
        (unsigned)int_type,
        (unsigned)op_type,
        (int)dataflow_mode,
        (int)accout_dest,
        (unsigned)input_a_zeropoint,
        (unsigned)input_b_zeropoint,
        input_a_addr,
        (unsigned)input_a_col_num_m1,
        (unsigned)input_a_row_num_m1,
        (unsigned)input_a_stride,
        input_b_addr,
        (unsigned)input_b_col_num_m1,
        (unsigned)input_b_row_num_m1,
        (unsigned)input_b_stride,
        (unsigned)biaspsum_width,
        (unsigned)biaspsum_height,
        biaspsum_addr,
        (unsigned)biaspsum_stride,
        output_addr,
        (unsigned)output_stride,
        (int)is_accumulate,
        (int)relu_enable,
        (unsigned)relu_type,
        (int)is_bias,
        output_zeropoint,
        (unsigned)quant_scale,
        (int)static_cast<int16_t>(quant_scaleshift));
    if (g_npu_runtime) {
        ConvConfig cfg = {
            pad_top, pad_bottom, pad_left, pad_right, pad_mode,
            weight_shape_m1, weight_stride_m1, weight_dilation_m1, is_group_conv,
            int_type, op_type, dataflow_mode, accout_dest,
            input_a_zeropoint, input_b_zeropoint,
            input_a_addr, input_a_col_num_m1, input_a_row_num_m1, input_a_stride,
            input_b_addr, input_b_col_num_m1, input_b_row_num_m1, input_b_stride,
            biaspsum_width, biaspsum_height, biaspsum_addr, biaspsum_stride,
            output_addr, output_stride, is_accumulate, relu_enable, relu_type, is_bias,
            output_zeropoint, quant_scale, quant_scaleshift
        };
        g_npu_runtime->run_conv(cfg);
    }
}

void npu_conv_tile_run(
    uint32_t sram_addr_ifm,
    uint32_t sram_addr_weight,
    uint32_t sram_addr_ofm,
    uint32_t acc_addr_psum,
    int32_t  c_in,
    int32_t  k_h,
    int32_t  k_w,
    int32_t  stride,
    int32_t  dilation,
    int32_t  i_cin,
    int32_t  t_cout,
    int32_t  t_h_out,
    int32_t  t_w_out,
    int32_t  t_cin,
    uint16_t quant_scale,
    uint16_t quant_scaleshift,
    bool     relu_enable,
    uint8_t  relu_type,
    bool     bias_enable,
    bool     is_group_conv

) {
    NPU_CAPI_LOG(
        "npu_conv_tile_run(sram_addr_ifm=0x%08X, sram_addr_weight=0x%08X, sram_addr_ofm=0x%08X, acc_addr_psum=0x%08X, c_in=%d, k_h=%d, k_w=%d, stride=%d, dilation=%d, i_cin=%d, t_cout=%d, t_h_out=%d, t_w_out=%d, t_cin=%d, quant_scale=%u, quant_scaleshift=%d, relu_enable=%d, relu_type=%u, bias_enable=%d, is_group_conv=%d)",
        sram_addr_ifm,
        sram_addr_weight,
        sram_addr_ofm,
        acc_addr_psum,
        c_in,
        k_h,
        k_w,
        stride,
        dilation,
        i_cin,
        t_cout,
        t_h_out,
        t_w_out,
        t_cin,
        (unsigned)quant_scale,
        (int)static_cast<int16_t>(quant_scaleshift),
        (int)relu_enable,
        (unsigned)relu_type,
        (int)bias_enable,
        (int)is_group_conv);
    if (g_npu_runtime) {
        NpuConvTileConfig cfg = {
            sram_addr_ifm,
            sram_addr_weight,
            sram_addr_ofm,
            acc_addr_psum,
            c_in,
            k_h,
            k_w,
            stride,
            dilation,
            i_cin,
            t_cout,
            t_h_out,
            t_w_out,
            t_cin,
            quant_scale,
            quant_scaleshift,
            relu_enable,
            relu_type,
            bias_enable,
            is_group_conv
        };
        g_npu_runtime->run_conv_tile(cfg);
    }
}


void npu_gemm_run(
    bool dataflow, uint8_t int_type, uint8_t optype, bool accout_dest,
    uint16_t input_a_zeropoint, uint16_t input_b_zeropoint,
    uint32_t output_zeropoint, uint16_t output_scale, uint16_t output_scaleshift,
    uint32_t biaspsum_addr, uint16_t biaspsum_stride, uint8_t biaspsum_width, uint8_t biaspsum_height,
    uint32_t output_addr, uint16_t output_stride, bool isaccu, bool relu, uint8_t relu_type,
    bool is_bias,
    uint32_t input_a_addr, uint16_t input_a_col_num, uint8_t input_a_row_num, uint16_t input_a_stride,
    uint32_t input_b_addr, uint8_t input_b_col_num, uint16_t input_b_row_num, uint16_t input_b_stride,
    bool asymmetric_activations
) {
    NPU_CAPI_LOG(
        "npu_gemm_run(dataflow=%d, int_type=%u, optype=%u, accout_dest=%d, input_a_zeropoint=%u, input_b_zeropoint=%u, output_zeropoint=%u, output_scale=%u, output_scaleshift=%d, biaspsum_addr=0x%08X, biaspsum_stride=%u, biaspsum_width=%u, biaspsum_height=%u, output_addr=0x%08X, output_stride=%u, isaccu=%d, relu=%d, relu_type=%u, is_bias=%d, input_a_addr=0x%08X, input_a_col_num=%u, input_a_row_num=%u, input_a_stride=%u, input_b_addr=0x%08X, input_b_col_num=%u, input_b_row_num=%u, input_b_stride=%u, asymmetric_activations=%d)",
        (int)dataflow,
        (unsigned)int_type,
        (unsigned)optype,
        (int)accout_dest,
        (unsigned)input_a_zeropoint,
        (unsigned)input_b_zeropoint,
        output_zeropoint,
        (unsigned)output_scale,
        (int)static_cast<int16_t>(output_scaleshift),
        biaspsum_addr,
        (unsigned)biaspsum_stride,
        (unsigned)biaspsum_width,
        (unsigned)biaspsum_height,
        output_addr,
        (unsigned)output_stride,
        (int)isaccu,
        (int)relu,
        (unsigned)relu_type,
        (int)is_bias,
        input_a_addr,
        (unsigned)input_a_col_num,
        (unsigned)input_a_row_num,
        (unsigned)input_a_stride,
        input_b_addr,
        (unsigned)input_b_col_num,
        (unsigned)input_b_row_num,
        (unsigned)input_b_stride,
        (int)asymmetric_activations);
    if (g_npu_runtime) {
        GemmConfig cfg = {
            dataflow, int_type, optype, accout_dest, asymmetric_activations,
            input_a_zeropoint, input_b_zeropoint,
            output_zeropoint, output_scale, output_scaleshift,
            biaspsum_addr, biaspsum_stride, biaspsum_width, biaspsum_height,
            output_addr, output_stride, isaccu, relu, relu_type, is_bias,
            input_a_addr, input_a_col_num, input_a_row_num, input_a_stride,
            input_b_addr, input_b_col_num, input_b_row_num, input_b_stride
        };
        g_npu_runtime->run_gemm(cfg);
    }
}

void npu_matadd_run(
    uint32_t input_a_addr,
    uint32_t input_b_addr,
    uint32_t output_addr,
    uint8_t  col_num_m1,
    uint8_t  row_num_m1,
    uint32_t output_zeropoint,
    uint16_t output_scale,
    uint16_t output_scaleshift
) {
    NPU_CAPI_LOG(
        "npu_matadd_run(input_a_addr=0x%08X, input_b_addr=0x%08X, output_addr=0x%08X, col_num_m1=%u, row_num_m1=%u, output_zeropoint=%u, output_scale=%u, output_scaleshift=%d)",
        input_a_addr,
        input_b_addr,
        output_addr,
        (unsigned)col_num_m1,
        (unsigned)row_num_m1,
        output_zeropoint,
        (unsigned)output_scale,
        (int)static_cast<int16_t>(output_scaleshift));
    if (g_npu_runtime) {
        MataddConfig cfg = {
            input_a_addr,
            input_b_addr,
            output_addr,
            col_num_m1,
            row_num_m1,
            output_zeropoint,
            output_scale,
            output_scaleshift
        };
        g_npu_runtime->run_matadd(cfg);
    }
}

void npu_transpose_run(
    uint32_t input_sram_addr,
    uint32_t output_sram_addr,
    uint16_t col_num,
    uint16_t row_num,
    bool     out_padding_row,
    bool     out_padding_col
) {
    NPU_CAPI_LOG(
        "npu_transpose_run(input_sram_addr=0x%08X, output_sram_addr=0x%08X, col_num=%u, row_num=%u, out_padding_row=%d, out_padding_col=%d)",
        input_sram_addr,
        output_sram_addr,
        (unsigned)col_num,
        (unsigned)row_num,
        (int)out_padding_row,
        (int)out_padding_col);
    if (g_npu_runtime) {
        TransposeConfig cfg = {
            input_sram_addr,
            output_sram_addr,
            col_num,
            row_num,
            out_padding_row,
            out_padding_col
        };
        g_npu_runtime->run_transpose(cfg);
    }
}

void npu_resample_run(
    uint8_t  resample_type,
    uint8_t  resample_op,
    uint32_t input_sram_addr,
    uint32_t output_sram_addr,
    uint16_t input_col_num,
    uint16_t input_row_num
) {
    NPU_CAPI_LOG(
        "npu_resample_run(resample_type=%u, resample_op=%u, input_sram_addr=0x%08X, output_sram_addr=0x%08X, input_col_num=%u, input_row_num=%u)",
        (unsigned)resample_type,
        (unsigned)resample_op,
        input_sram_addr,
        output_sram_addr,
        (unsigned)input_col_num,
        (unsigned)input_row_num);
    if (g_npu_runtime) {
        ResampleConfig cfg = {
            resample_type,
            resample_op,
            input_sram_addr,
            output_sram_addr,
            input_col_num,
            input_row_num
        };
        g_npu_runtime->run_resample(cfg);
    }
}

void npu_layout_nchw_to_nchwc32(
    uint32_t sram_addr,
    uint32_t output_addr,
    uint16_t n,
    uint16_t c,
    uint16_t h,
    uint16_t w
) {
    NPU_CAPI_LOG(
        "npu_layout_nchw_to_nchwc32(sram_addr=0x%08X, output_addr=0x%08X, n=%u, c=%u, h=%u, w=%u)",
        sram_addr,
        output_addr,
        (unsigned)n,
        (unsigned)c,
        (unsigned)h,
        (unsigned)w);
    if (g_npu_runtime) {
        LayoutConvertConfig cfg = {sram_addr, output_addr, n, c, h, w};
        g_npu_runtime->run_nchw_to_nchwc32(cfg);
    }
}

void npu_layout_nchwc32_to_nchw(
    uint32_t sram_addr,
    uint32_t output_addr,
    uint16_t n,
    uint16_t c,
    uint16_t h,
    uint16_t w
) {
    NPU_CAPI_LOG(
        "npu_layout_nchwc32_to_nchw(sram_addr=0x%08X, output_addr=0x%08X, n=%u, c=%u, h=%u, w=%u)",
        sram_addr,
        output_addr,
        (unsigned)n,
        (unsigned)c,
        (unsigned)h,
        (unsigned)w);
    if (g_npu_runtime) {
        LayoutConvertConfig cfg = {sram_addr, output_addr, n, c, h, w};
        g_npu_runtime->run_nchwc32_to_nchw(cfg);
    }
}

void npu_dma_mvin_test(void* host_ptr, uint32_t sram_addr, uint32_t col_num, uint32_t row_num,
                       uint16_t sram_stride, uint32_t dram_stride, uint8_t precision, uint8_t input_type,
                       bool dest, bool is_bias, bool is_quant, uint32_t quant_zero, uint16_t quant_scale, uint16_t quant_shift) {
    NPU_CAPI_LOG(
        "npu_dma_mvin_test(host_ptr=%p, sram_addr=0x%08X, col_num=%u, row_num=%u, sram_stride=%u, dram_stride=%u, precision=%u, input_type=%u, dest=%d, is_bias=%d, is_quant=%d, quant_zero=%u, quant_scale=%u, quant_shift=%d)",
        host_ptr,
        sram_addr,
        (unsigned)col_num,
        (unsigned)row_num,
        (unsigned)sram_stride,
        dram_stride,
        (unsigned)precision,
        (unsigned)input_type,
        (int)dest,
        (int)is_bias,
        (int)is_quant,
        quant_zero,
        (unsigned)quant_scale,
        (int)static_cast<int16_t>(quant_shift));
    #ifdef NPU_DEBUG
    printf("[TEST] MVIN: Host=%p SRAM=0x%x Size=%dx%d\n", host_ptr, sram_addr, row_num, col_num);
    #endif
}

void npu_dma_mvout_test(void* host_ptr, uint32_t sram_addr, uint32_t col_num, uint32_t row_num,
                        uint16_t sram_stride, uint32_t dram_stride, uint8_t precision, uint8_t output_type,
                        bool source, bool is_quant, uint32_t quant_zero, uint32_t f32_scale) {
    NPU_CAPI_LOG(
        "npu_dma_mvout_test(host_ptr=%p, sram_addr=0x%08X, col_num=%u, row_num=%u, sram_stride=%u, dram_stride=%u, precision=%u, output_type=%u, source=%d, is_quant=%d, quant_zero=%u, f32_scale=0x%08X)",
        host_ptr,
        sram_addr,
        (unsigned)col_num,
        (unsigned)row_num,
        (unsigned)sram_stride,
        dram_stride,
        (unsigned)precision,
        (unsigned)output_type,
        (int)source,
        (int)is_quant,
        quant_zero,
        (unsigned)f32_scale);
    #ifdef NPU_DEBUG
    printf("[TEST] MVOUT: Host=%p SRAM=0x%x Size=%dx%d\n", host_ptr, sram_addr, row_num, col_num);
    #endif
}

} // extern "C"
