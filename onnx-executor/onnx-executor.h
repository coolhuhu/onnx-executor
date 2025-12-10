#ifndef ONNX_EXECUTOR_H_
#define ONNX_EXECUTOR_H_

#include <memory>
#include <string>

#include "onnx-executor/provider-config.h"
#include "onnxruntime_cxx_api.h"  // NOLINT

namespace onnx_executor {

struct OnnxRuntimeConfig {
  std::string model;
  std::string provider = "cpu";
  int32_t num_threads = 1;

  OnnxRuntimeProviderConfig provider_config;
};

class OnnxExecutor {
 public:
  explicit OnnxExecutor(const OnnxRuntimeConfig &config);

  ~OnnxExecutor();

  std::vector<Ort::Value> Forward(const std::vector<Ort::Value> &inputs);

  std::vector<Ort::Value> Forward(std::vector<Ort::Value> &&inputs);

  std::string LookupCustomMetaData(const char *key);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace onnx_executor

#endif  // ONNX_EXECUTOR_H_