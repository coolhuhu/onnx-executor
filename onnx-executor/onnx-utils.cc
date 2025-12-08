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
    case Provider::kTRT: {
      // TODO(lianghu)
      ONNX_EXECUTOR_LOGE("TensorRt currently unsupported.");
      exit(-1);
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
