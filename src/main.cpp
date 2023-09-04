#include "yolo.hpp"

using namespace nvinfer1;

int main(int argc, char** argv) {
    if (argc == 1) {
        std::cout << "Usage: \n 1. ./yolo_onnx_zed yolov8s.onnx yolov8s.engine\n 2. ./yolo_onnx_zed yolov8s.onnx yolov8s.engine images:1x3x512x512" << std::endl;
        return EXIT_SUCCESS;
    }
    
    // Check Optim engine first
    if ((argc >= 3)) {
        std::string onnx_path = std::string(argv[1]);
        std::string engine_path = std::string(argv[2]);
        OptimDim dyn_dim_profile;

        if (argc == 4) {
            std::string optim_profile = std::string(argv[3]);
            bool error = dyn_dim_profile.setFromString(optim_profile);
            if (error) {
                std::cerr << "Invalid dynamic dimension argument, expecting something like 'images:1x3x512x512'" << std::endl;
                return EXIT_FAILURE;
            }
        }

        Yolo::build_engine(onnx_path, engine_path, dyn_dim_profile);
        return EXIT_SUCCESS;
    } else {
        std::cerr << "Invalid argument list! (Expected 3-4 arguments, got " << argc << "!)" << std::endl;
        return EXIT_FAILURE;
    }
}
