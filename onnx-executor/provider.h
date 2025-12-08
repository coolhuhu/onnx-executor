#ifndef ONNX_EXECUTOR_PROVIDER_H_
#define ONNX_EXECUTOR_PROVIDER_H_

#include <string>

namespace onnx_executor {

// Please refer to
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/get_execution_providers.cc
// for a list of available providers
enum class Provider {
  kCPU = 0,   // CPUExecutionProvider
  kCUDA = 1,  // CUDAExecutionProvider
  kTRT = 2,   // TensorRTExecutionProvider
};

/**
 * Convert a string to an enum.
 *
 * @param s We will convert it to lowercase before comparing.
 * @return Return an instance of Provider.
 */
Provider StringToProvider(std::string s);

}  // namespace onnx_executor

#endif  // ONNX_EXECUTOR_PROVIDER_H_