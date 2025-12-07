#include "onnx-executor/onnx-executor.h"

#include <unordered_map>

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

 private:
  void Init() {
    auto buf = ReadFile(config_.model);
    sess_ = std::make_unique<Ort::Session>(env_, buf.data(), buf.size(),
                                           sess_opts_);
    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);
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
};

OnnxExecutor::OnnxExecutor(const OnnxRuntimeConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

std::vector<Ort::Value> OnnxExecutor::Forward(
    const std::vector<Ort::Value> &inputs) {
  return impl_->Forward(inputs);
}

std::vector<Ort::Value> OnnxExecutor::Forward(
    std::vector<Ort::Value> &&inputs) {
  return impl_->Forward(std::forward<std::vector<Ort::Value>>(inputs));
}

}  // namespace onnx_executor
