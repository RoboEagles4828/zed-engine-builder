#ifndef YOLO_HPP
#define YOLO_HPP

#include <NvInfer.h>
#include "logging.h"
#include "utils.h"

enum class YOLO_MODEL_VERSION_OUTPUT_STYLE {
    YOLOV6,
    YOLOV8_V5
};

struct BBox {
    float x1, y1, x2, y2;
};

struct BBoxInfo {
    BBox box;
    int label;
    float prob;
};

inline std::vector<std::string> split_str(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
}

struct OptimDim {
    nvinfer1::Dims4 size;
    std::string tensor_name;

    bool setFromString(std::string &arg) {
        // "images:1x3x512x512"
        std::vector<std::string> v_ = split_str(arg, ":");
        if (v_.size() != 2) return true;

        std::string dims_str = v_.back();
        std::vector<std::string> v = split_str(dims_str, "x");

        size.nbDims = 4;
        // assuming batch is 1 and channel is 3
        size.d[0] = 1;
        size.d[1] = 3;

        if (v.size() == 2) {
            size.d[2] = stoi(v[0]);
            size.d[3] = stoi(v[1]);
        } else if (v.size() == 3) {
            size.d[2] = stoi(v[1]);
            size.d[3] = stoi(v[2]);
        } else if (v.size() == 4) {
            size.d[2] = stoi(v[2]);
            size.d[3] = stoi(v[3]);
        } else return true;

        if (size.d[2] != size.d[3]) std::cerr << "Warning only squared input are currently supported" << std::endl;

        tensor_name = v_.front();
        return false;
    }
};

class Yolo {
public:
    Yolo();
    ~Yolo();

    static int build_engine(std::string onnx_path, std::string engine_path, OptimDim dyn_dim_profile);

private:

    // Get input dimension size
    nvinfer1::ICudaEngine* engine;

    bool is_init = false;
};

#endif /* YOLO_HPP */