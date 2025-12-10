#ifndef ONNX_EXECUTOR_ONNX_UTILS_H_
#define ONNX_EXECUTOR_ONNX_UTILS_H_

#include <unordered_map>

#include "onnx-executor/provider-config.h"
#include "onnxruntime_cxx_api.h"  // NOLINT

namespace onnx_executor {

Ort::SessionOptions GetSessionOptions(
    int32_t num_threads, const std::string &provider_str,
    const OnnxRuntimeProviderConfig *provider_config = nullptr);

/**
 * Get the input names of a model.
 *
 * @param sess An onnxruntime session.
 * @param input_names. On return, it contains the input names of the model.
 * @param input_names_ptr. On return, input_names_ptr[i] contains
 *                         input_names[i].c_str()
 */
void GetInputNames(Ort::Session *sess, std::vector<std::string> *input_names,
                   std::vector<const char *> *input_names_ptr);

/**
 * Get the output names of a model.
 *
 * @param sess An onnxruntime session.
 * @param output_names. On return, it contains the output names of the model.
 * @param output_names_ptr. On return, output_names_ptr[i] contains
 *                         output_names[i].c_str()
 */
void GetOutputNames(Ort::Session *sess, std::vector<std::string> *output_names,
                    std::vector<const char *> *output_names_ptr);

template <typename T>
Ort::Value CreateTensor(T *data, size_t data_len, int64_t *shape,
                        size_t shape_len) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  return Ort::Value::CreateTensor(memory_info, data, data_len, shape,
                                  shape_len);
}

void PrintModelInputInfo(Ort::Session *sess,
                         Ort::AllocatorWithDefaultOptions &allocator);

void PrintModelMetaData(Ort::ModelMetadata &meta_data);

void GetModelMetaData(
    Ort::ModelMetadata &meta_data,
    std::unordered_map<std::string, std::string> *meta_data_map);

}  // namespace onnx_executor

#endif  // ONNX_EXECUTOR_ONNX_UTILS_H_