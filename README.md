# ZED Yolo Engine Builder
A stripped down version of [this ZED example](https://github.com/stereolabs/zed-sdk/tree/master/object%20detection/custom%20detector/cpp/tensorrt_yolov5-v6-v8_onnx) that is only meant to build the engine files used for running Yolo with the ZED camera.

Intended for use in headless environments with minimal libraries (e.g Docker Containers)

## Installation:
Install the [ZED SDK](https://www.stereolabs.com/developers/release/) and [CUDA](https://developer.nvidia.com/cuda-downloads), as well as the NVinfer runtime from TensorRT.

## Building:
Build the project with `cmake . && make` in order to create the `zed_engine_builder` executable.

## Running:
Take your onnx file and run
```
./zed_engine_builder model.onnx model.engine
```

You can also run it on its own for a help message.