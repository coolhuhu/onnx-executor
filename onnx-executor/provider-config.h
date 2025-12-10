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

struct TensorrtProviderConfig {
  int64_t trt_max_workspace_size = 2147483647;
  int32_t trt_max_partition_iterations = 10;
  int32_t trt_min_subgraph_size = 5;
  bool trt_fp16_enable = true;
  bool trt_detailed_build_log = false;
  bool trt_engine_cache_enable = true;
  bool trt_timing_cache_enable = true;
  std::string trt_engine_cache_path = ".";
  std::string trt_timing_cache_path = ".";
  bool trt_dump_subgraphs = false;
  std::string trt_profile_min_shapes;
  std::string trt_profile_max_shapes;
  std::string trt_profile_opt_shapes;

  TensorrtProviderConfig() = default;

  explicit TensorrtProviderConfig(
      int64_t trt_max_workspace_size, int32_t trt_max_partition_iterations,
      int32_t trt_min_subgraph_size, bool trt_fp16_enable,
      bool trt_detailed_build_log, bool trt_engine_cache_enable,
      bool trt_timing_cache_enable, const std::string &trt_engine_cache_path,
      const std::string &trt_timing_cache_path, bool trt_dump_subgraphs)
      : trt_max_workspace_size(trt_max_workspace_size),
        trt_max_partition_iterations(trt_max_partition_iterations),
        trt_min_subgraph_size(trt_min_subgraph_size),
        trt_fp16_enable(trt_fp16_enable),
        trt_detailed_build_log(trt_detailed_build_log),
        trt_engine_cache_enable(trt_engine_cache_enable),
        trt_timing_cache_enable(trt_timing_cache_enable),
        trt_engine_cache_path(trt_engine_cache_path),
        trt_timing_cache_path(trt_timing_cache_path),
        trt_dump_subgraphs(trt_dump_subgraphs),
        trt_profile_min_shapes(trt_profile_min_shapes),
        trt_profile_max_shapes(trt_profile_max_shapes),
        trt_profile_opt_shapes(trt_profile_opt_shapes) {}

  // TODO(lianghu) validate config.
};

struct OnnxRuntimeProviderConfig {
  CUDAProviderConfig cuda_provider_config;
  TensorrtProviderConfig tensorrt_provider_config;
  int32_t device_id = 0;

  OnnxRuntimeProviderConfig() = default;
};

}  // namespace onnx_executor

#endif  // ONNX_EXECUTOR_PROVIDER_CONFIG_H_