#include "onnx-executor/onnx-utils.h"

#include <sstream>

#include "onnx-executor/macros.h"
#include "onnx-executor/provider.h"

namespace onnx_executor {

Ort::SessionOptions GetSessionOptions(
    int32_t num_threads, const std::string &provider_str,
    const OnnxRuntimeProviderConfig *provider_config /* = nullptr */) {
  Provider p = StringToProvider(provider_str);

  Ort::SessionOptions sess_opts;
  sess_opts.SetIntraOpNumThreads(num_threads);

  sess_opts.SetInterOpNumThreads(num_threads);

  // TODO(lianghu): support other provider.
  std::vector<std::string> available_providers = Ort::GetAvailableProviders();
  std::ostringstream os;
  for (const auto &ep : available_providers) {
    os << ep << ", ";
  }

  switch (p) {
    case Provider::kCPU:
      // nothing to do
      break;
    case Provider::kTRT: {
      if (provider_config == nullptr) {
        ONNX_EXECUTOR_LOGE("trt provider shouble be set provider config");
        exit(-1);
      } else {
        auto trt_config = provider_config->tensorrt_provider_config;
        auto device_id = std::to_string(provider_config->device_id);
        auto trt_max_workspace_size =
            std::to_string(trt_config.trt_max_workspace_size);
        auto trt_max_partition_iterations =
            std::to_string(trt_config.trt_max_partition_iterations);
        auto trt_min_subgraph_size =
            std::to_string(trt_config.trt_min_subgraph_size);
        auto trt_fp16_enable = std::to_string(trt_config.trt_fp16_enable);
        auto trt_detailed_build_log =
            std::to_string(trt_config.trt_detailed_build_log);
        auto trt_engine_cache_enable =
            std::to_string(trt_config.trt_engine_cache_enable);
        auto trt_timing_cache_enable =
            std::to_string(trt_config.trt_timing_cache_enable);
        auto trt_dump_subgraphs = std::to_string(trt_config.trt_dump_subgraphs);

        std::vector<std::pair<const char *, const char *>> trt_options = {
            {"device_id", device_id.c_str()},
            {"trt_max_workspace_size", trt_max_workspace_size.c_str()},
            {"trt_max_partition_iterations",
             trt_max_partition_iterations.c_str()},
            {"trt_min_subgraph_size", trt_min_subgraph_size.c_str()},
            {"trt_fp16_enable", trt_fp16_enable.c_str()},
            {"trt_detailed_build_log", trt_detailed_build_log.c_str()},
            {"trt_engine_cache_enable", trt_engine_cache_enable.c_str()},
            {"trt_engine_cache_path", trt_config.trt_engine_cache_path.c_str()},
            {"trt_timing_cache_enable", trt_timing_cache_enable.c_str()},
            {"trt_timing_cache_path", trt_config.trt_timing_cache_path.c_str()},
            {"trt_dump_subgraphs", trt_dump_subgraphs.c_str()}};

        if (!trt_config.trt_profile_min_shapes.empty()) {
          trt_options.emplace_back("trt_profile_min_shapes",
                                   trt_config.trt_profile_min_shapes.c_str());
        }
        if (!trt_config.trt_profile_max_shapes.empty()) {
          trt_options.emplace_back("trt_profile_max_shapes",
                                   trt_config.trt_profile_max_shapes.c_str());
        }
        if (!trt_config.trt_profile_opt_shapes.empty()) {
          trt_options.emplace_back("trt_profile_opt_shapes",
                                   trt_config.trt_profile_opt_shapes.c_str());
        }

        // TODO: add other trt option

        std::vector<const char *> option_keys, option_values;
        for (const auto &pair : trt_options) {
          option_keys.emplace_back(pair.first);
          option_values.emplace_back(pair.second);
        }

        std::vector<std::string> available_providers =
            Ort::GetAvailableProviders();
        if (std::find(available_providers.begin(), available_providers.end(),
                      "TensorrtExecutionProvider") !=
            available_providers.end()) {
          const auto &api = Ort::GetApi();

          OrtTensorRTProviderOptionsV2 *tensorrt_options = nullptr;
          OrtStatus *statusC =
              api.CreateTensorRTProviderOptions(&tensorrt_options);
          OrtStatus *statusU = api.UpdateTensorRTProviderOptions(
              tensorrt_options, option_keys.data(), option_values.data(),
              option_keys.size());
          sess_opts.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);

          if (statusC) {
            const auto &api = Ort::GetApi();
            const char *msg = api.GetErrorMessage(statusC);
            ONNX_EXECUTOR_LOGE(
                "Failed to enable TensorRT : %s."
                "Available providers: %s. Fallback to cuda",
                msg, os.str().c_str());
            api.ReleaseStatus(statusC);
          }
          if (statusU) {
            const auto &api = Ort::GetApi();
            const char *msg = api.GetErrorMessage(statusU);
            ONNX_EXECUTOR_LOGE(
                "Failed to enable TensorRT : %s."
                "Available providers: %s. Fallback to cuda",
                msg, os.str().c_str());
            api.ReleaseStatus(statusU);
          }

          api.ReleaseTensorRTProviderOptions(tensorrt_options);
        }
      }
      // break; is omitted here intentionally so that
      // if TRT not available, CUDA will be used
    }
    case Provider::kCUDA: {
      if (std::find(available_providers.begin(), available_providers.end(),
                    "CUDAExecutionProvider") != available_providers.end()) {
        // The CUDA provider is available, proceed with setting the options
        OrtCUDAProviderOptions options;

        if (provider_config != nullptr) {
          options.device_id = provider_config->device_id;
          options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch(
              provider_config->cuda_provider_config.ort_cudnn_conv_algo_search);
        } else {
          options.device_id = 0;
          // Default OrtCudnnConvAlgoSearchExhaustive is extremely slow
          options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
          // set more options on need
        }
        sess_opts.AppendExecutionProvider_CUDA(options);
      } else {
        ONNX_EXECUTOR_LOGE(
            "Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON. Available "
            "providers: %s. Fallback to cpu!",
            os.str().c_str());
      }
      break;
    }
    default:
      break;
  }

  return sess_opts;
}

static std::string GetInputName(Ort::Session *sess, size_t index,
                                OrtAllocator *allocator) {
// Note(fangjun): We only tested 1.17.1 and 1.11.0
// For other versions, we may need to change it
#if ORT_API_VERSION >= 12
  auto v = sess->GetInputNameAllocated(index, allocator);
  return v.get();
#else
  auto v = sess->GetInputName(index, allocator);
  std::string ans = v;
  allocator->Free(allocator, v);
  return ans;
#endif
}

static std::string GetOutputName(Ort::Session *sess, size_t index,
                                 OrtAllocator *allocator) {
// Note(fangjun): We only tested 1.17.1 and 1.11.0
// For other versions, we may need to change it
#if ORT_API_VERSION >= 12
  auto v = sess->GetOutputNameAllocated(index, allocator);
  return v.get();
#else
  auto v = sess->GetOutputName(index, allocator);
  std::string ans = v;
  allocator->Free(allocator, v);
  return ans;
#endif
}

void GetInputNames(Ort::Session *sess, std::vector<std::string> *input_names,
                   std::vector<const char *> *input_names_ptr) {
  Ort::AllocatorWithDefaultOptions allocator;
  size_t node_count = sess->GetInputCount();
  input_names->resize(node_count);
  input_names_ptr->resize(node_count);
  for (size_t i = 0; i != node_count; ++i) {
    (*input_names)[i] = GetInputName(sess, i, allocator);
    (*input_names_ptr)[i] = (*input_names)[i].c_str();
  }
}

void GetOutputNames(Ort::Session *sess, std::vector<std::string> *output_names,
                    std::vector<const char *> *output_names_ptr) {
  Ort::AllocatorWithDefaultOptions allocator;
  size_t node_count = sess->GetOutputCount();
  output_names->resize(node_count);
  output_names_ptr->resize(node_count);
  for (size_t i = 0; i != node_count; ++i) {
    (*output_names)[i] = GetOutputName(sess, i, allocator);
    (*output_names_ptr)[i] = (*output_names)[i].c_str();
  }
}

void PrintModelInputInfo(Ort::Session *sess,
                         Ort::AllocatorWithDefaultOptions &allocator) {
  size_t num_input_nodes = sess->GetInputCount();
  ONNX_EXECUTOR_LOGE("number of inputs: %lu", num_input_nodes);

  for (size_t i = 0; i < num_input_nodes; ++i) {
    auto input_name = sess->GetInputNameAllocated(i, allocator);
    ONNX_EXECUTOR_LOGE("input_%lu's name is %s", i, input_name.get());

    Ort::TypeInfo type_info = sess->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_shape = tensor_info.GetShape();

    std::string shape_str;
    for (auto s : input_shape) {
      shape_str.append(std::to_string(s));
      shape_str.push_back(' ');
    }

    ONNX_EXECUTOR_LOGE("input_%lu's shape is: %s", i, shape_str.c_str());
  }
}

void PrintModelMetaData(Ort::ModelMetadata &meta_data) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<Ort::AllocatedStringPtr> v =
      meta_data.GetCustomMetadataMapKeysAllocated(allocator);
  for (const auto &key : v) {
    auto p = meta_data.LookupCustomMetadataMapAllocated(key.get(), allocator);
    ONNX_EXECUTOR_LOGE("%s: %s", key.get(), p.get());
  }
}

void GetModelMetaData(
    Ort::ModelMetadata &meta_data,
    std::unordered_map<std::string, std::string> *meta_data_map) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<Ort::AllocatedStringPtr> v =
      meta_data.GetCustomMetadataMapKeysAllocated(allocator);

  for (const auto &key : v) {
    auto p = meta_data.LookupCustomMetadataMapAllocated(key.get(), allocator);
    meta_data_map->emplace(key.get(), p.get());
  }
}

}  // namespace onnx_executor
