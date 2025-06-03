#include "func.h"

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

std::vector<char> loadEngine(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open engine");
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void saveDetections(const std::string& filename, const std::vector<std::vector<float>>& detections) {
    std::ofstream out(filename);
    for (const auto& det : detections) {
        for (float val : det)
            out << val << " ";
        out << "\n";
    }
}