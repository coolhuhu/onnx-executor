cmake_minimum_required(VERSION 3.14)
project(onnx-executor)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

option(ENABLE_ONNX_EXECUTOR_GPU "Whether to use GPU" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(onnxruntime)

add_subdirectory(onnx-executor)
add_subdirectory(example)