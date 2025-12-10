#include <cassert>
#include <future>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "onnx-executor/onnx-executor.h"
#include "onnx-executor/onnx-utils.h"

class SileroVadModel {
 public:
  explicit SileroVadModel(const onnx_executor::OnnxRuntimeConfig &config)
      : executor_(std::make_shared<onnx_executor::OnnxExecutor>(config)) {}

  SileroVadModel(const SileroVadModel &other) { executor_ = other.executor_; }

  std::unique_ptr<SileroVadModel> Clone() {
    return std::make_unique<SileroVadModel>(*this);
  }

  float Forward(float *input, int32_t n) {
    // n shouble be 576 for silero vad chunk size.
    std::vector<int64_t> input_shape{1, n};
    std::vector<float> state(256, 0);
    std::array<int64_t, 3> state_shape{2, 1, 128};
    int64_t sr = 16000;
    int64_t sr_shape = 1;

    Ort::Value input_tensor = onnx_executor::CreateTensor(
        input, n, input_shape.data(), input_shape.size());
    Ort::Value state_tensor = onnx_executor::CreateTensor(
        state.data(), state.size(), state_shape.data(), state_shape.size());
    Ort::Value sr_tensor =
        onnx_executor::CreateTensor<int64_t>(&sr, 1, &sr_shape, 1);

    std::vector<Ort::Value> inputs(3);
    inputs[0] = std::move(input_tensor);
    inputs[1] = std::move(state_tensor);
    inputs[2] = std::move(sr_tensor);

    std::vector<Ort::Value> output = executor_->Forward(std::move(inputs));
    float logit = output[0].GetTensorMutableData<float>()[0];
    return logit;
  }

 private:
  std::shared_ptr<onnx_executor::OnnxExecutor> executor_;
  // other state for model.
};

std::vector<float> generate_random_data(int size) {
  std::vector<float> data(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1.0);
  for (auto &x : data) {
    x = dis(gen);
  }
  return data;
}

std::vector<std::vector<std::vector<float>>> PrepareMultiThreadInputs(
    int num_threads, int num_inputs_per_thread) {
  assert(num_threads > 0 && num_inputs_per_thread > 0);

  std::vector<std::vector<std::vector<float>>> inputs(num_threads);
  inputs[0].resize(num_inputs_per_thread);
  for (int j = 0; j < num_inputs_per_thread; ++j) {
    inputs[0][j] = generate_random_data(576);
  }
  for (int i = 1; i < num_threads; ++i) {
    inputs[i] = inputs[0];
  }
  return inputs;
}

std::vector<float> ThreadFunc(std::vector<std::vector<float>> &inputs,
                              SileroVadModel *model) {
  int n = inputs.size();
  std::vector<float> logits(n);
  for (int i = 0; i < n; ++i) {
    logits[i] = model->Forward(inputs[i].data(), inputs[i].size());
  }
  return logits;
}

int main(int argc, char **argv) {
  // ./multi_thread_load. /to/path/silero_vad.onnx
  std::string model_path = argv[1];
  int num_threads = 1;
  num_threads = std::atoi(argv[2]);
  if (num_threads <= 0) {
    std::cerr << "num_threads should be >= 1" << std::endl;
    std::exit(-1);
  }

  onnx_executor::OnnxRuntimeConfig config;
  config.model = model_path;
  config.provider = "cpu";

  std::vector<std::unique_ptr<SileroVadModel>> models(num_threads);
  models[0] = std::make_unique<SileroVadModel>(config);
  for (int i = 1; i < num_threads; ++i) {
    models[i] = models[0]->Clone();
  }
  std::cout << "multi thread model loaded." << std::endl;

  std::vector<std::vector<std::vector<float>>> datas =
      PrepareMultiThreadInputs(num_threads, 10);

  std::vector<std::promise<std::vector<float>>> promises(num_threads);
  std::vector<std::future<std::vector<float>>> futures(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    futures[i] = promises[i].get_future();
  }

  std::vector<std::thread> threads(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    threads[i] = std::thread([i, &datas, &models, &promises, &futures]() {
      auto res = ThreadFunc(datas[i], models[i].get());
      promises[i].set_value(res);
    });
  }

  for (int i = 0; i < num_threads; ++i) {
    threads[i].join();
  }

  for (int i = 0; i < num_threads; ++i) {
    auto res = futures[i].get();
    std::cout << "thread " << i << ":" << std::endl;
    for (auto x : res) {
      std::cout << x << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
  }
}
