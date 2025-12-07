#include "onnx-executor/onnx-executor.h"

#include <chrono>

#include "onnx-executor/macros.h"
#include "onnx-executor/onnx-utils.h"
#include "onnx-executor/utils.h"

namespace onnx_executor {

class OnnxExecutor::Impl {
 public:
  Impl(const OnnxRuntimeConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config.num_threads, config.provider)),
        allocator_{} {
    Init();
  }

  std::vector<Ort::Value> Forward(const std::vector<Ort::Value> &inputs) {
    return sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                      output_names_ptr_.data(), output_names_ptr_.size());
  }

  std::vector<Ort::Value> Forward(std::vector<Ort::Value> &&inputs) {
    return sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                      output_names_ptr_.data(), output_names_ptr_.size());
  }

  std::string LookupCustomMetaData(const char *key) {
    if (meta_data_.find(key) != meta_data_.end()) {
      return meta_data_[key];
    }
    return "";
  }

 private:
  void Init() {
    sess_ =
        std::make_unique<Ort::Session>(env_, config_.model.c_str(), sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
    GetModelMetaData(meta_data, &meta_data_);
    PrintMetaData();
  }

  void PrintMetaData() {
    for (auto &kv : meta_data_) {
      ONNX_EXECUTOR_LOGE("%s: %s", kv.first.c_str(), kv.second.c_str());
    }
  }

 private:
  OnnxRuntimeConfig config_;

  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;
  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  std::unordered_map<std::string, std::string> meta_data_;
};

OnnxExecutor::OnnxExecutor(const OnnxRuntimeConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OnnxExecutor::~OnnxExecutor() = default;

std::vector<Ort::Value> OnnxExecutor::Forward(
    const std::vector<Ort::Value> &inputs) {
  return impl_->Forward(inputs);
}

std::vector<Ort::Value> OnnxExecutor::Forward(
    std::vector<Ort::Value> &&inputs) {
  return impl_->Forward(std::forward<std::vector<Ort::Value>>(inputs));
}

std::string OnnxExecutor::LookupCustomMetaData(const char *key) {
  return impl_->LookupCustomMetaData(key);
}

}  // namespace onnx_executor
