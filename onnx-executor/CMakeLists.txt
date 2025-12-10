set(onnx_executor_srcs
  onnx-executor.cc
  onnx-utils.cc
  provider.cc
  provider-config.cc
  utils.cc
)

add_library(onnx-executor SHARED ${onnx_executor_srcs})
target_link_libraries(onnx-executor PUBLIC onnxruntime)
target_include_directories(onnx-executor PRIVATE ${CMAKE_SOURCE_DIR})