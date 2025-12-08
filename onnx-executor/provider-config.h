#ifndef ONNX_EXECUTOR_PROVIDER_CONFIG_H_
#define ONNX_EXECUTOR_PROVIDER_CONFIG_H_

#include "onnx-executor/macros.h"
#include "onnxruntime_cxx_api.h"  // NOLINT

namespace onnx_executor {

struct CUDAProviderConfig {
  int32_t ort_cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;

  CUDAProviderConfig() = default;
  explicit CUDAProviderConfig(int32_t cudnn_conv_algo_search)
      : ort_cudnn_conv_algo_search(cudnn_conv_algo_search) {
    if (ort_cudnn_conv_algo_search < 1 || ort_cudnn_conv_algo_search > 3) {
      ONNX_EXECUTOR_LOGE(
          "cudnn_conv_algo_search: '%d' is not a valid option."
          "Options : [1,3]. Check OnnxRT docs",
          ort_cudnn_conv_algo_search);
      ONNX_EXECUTOR_LOGE("Fallback to OrtCudnnConvAlgoSearchHeuristic");
    }
  }
};

struct OnnxRuntimeProviderConfig {
  CUDAProviderConfig cuda_provider_config;
  int32_t device_id = 0;

  OnnxRuntimeProviderConfig() = default;
};

}  // namespace onnx_executor

#endif  // ONNX_EXECUTOR_PROVIDER_CONFIG_H_