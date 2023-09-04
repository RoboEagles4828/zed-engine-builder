#include "yolo.hpp"
#include "NvOnnxParser.h"

using namespace nvinfer1;

static Logger gLogger;

Yolo::Yolo() {
}

Yolo::~Yolo() {
    if (is_init) {
        // Destroy the engine
        engine->destroy();
    }
    is_init = false;
}


int Yolo::build_engine(std::string onnx_path, std::string engine_path, OptimDim dyn_dim_profile) {


    std::vector<uint8_t> onnx_file_content;
    if (readFile(onnx_path, onnx_file_content)) return 1;

    if ((!onnx_file_content.empty())) {

        ICudaEngine * engine;
        // Create engine (onnx)
        std::cout << "Creating engine from onnx model" << std::endl;

        gLogger.setReportableSeverity(Severity::kINFO);
        auto builder = nvinfer1::createInferBuilder(gLogger);
        if (!builder) {
            std::cerr << "createInferBuilder failed" << std::endl;
            return 1;
        }

        auto explicitBatch = 1U << static_cast<uint32_t> (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(explicitBatch);

        if (!network) {
            std::cerr << "createNetwork failed" << std::endl;
            return 1;
        }

        auto config = builder->createBuilderConfig();
        if (!config) {
            std::cerr << "createBuilderConfig failed" << std::endl;
            return 1;
        }

        ////////// Dynamic dimensions handling : support only 1 size at a time
        if (!dyn_dim_profile.tensor_name.empty()) {

            IOptimizationProfile* profile = builder->createOptimizationProfile();

            profile->setDimensions(dyn_dim_profile.tensor_name.c_str(), OptProfileSelector::kMIN, dyn_dim_profile.size);
            profile->setDimensions(dyn_dim_profile.tensor_name.c_str(), OptProfileSelector::kOPT, dyn_dim_profile.size);
            profile->setDimensions(dyn_dim_profile.tensor_name.c_str(), OptProfileSelector::kMAX, dyn_dim_profile.size);

            config->addOptimizationProfile(profile);
            builder->setMaxBatchSize(1);
        }

        auto parser = nvonnxparser::createParser(*network, gLogger);
        if (!parser) {
            std::cerr << "nvonnxparser::createParser failed" << std::endl;
            return 1;
        }

        bool parsed = false;
        unsigned char *onnx_model_buffer = onnx_file_content.data();
        size_t onnx_model_buffer_size = onnx_file_content.size() * sizeof (char);
        parsed = parser->parse(onnx_model_buffer, onnx_model_buffer_size);

        if (!parsed) {
            std::cerr << "onnx file parsing failed" << std::endl;
            return 1;
        }

        if (builder->platformHasFastFp16()) {
            std::cout << "FP16 enabled!" << std::endl;
            config->setFlag(BuilderFlag::kFP16);
        }

        //////////////// Actual engine building

        engine = builder->buildEngineWithConfig(*network, *config);

        onnx_file_content.clear();

        // write plan file if it is specified        
        if (engine == nullptr) return 1;
        IHostMemory* ptr = engine->serialize();
        assert(ptr);
        if (ptr == nullptr) return 1;

        FILE *fp = fopen(engine_path.c_str(), "wb");
        fwrite(reinterpret_cast<const char*> (ptr->data()), ptr->size() * sizeof (char), 1, fp);
        fclose(fp);

        parser->destroy();
        network->destroy();
        config->destroy();
        builder->destroy();

        engine->destroy();

        return 0;
    } else return 1;


}