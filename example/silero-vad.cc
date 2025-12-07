#include <iostream>
#include <random>

#include "onnx-executor/onnx-executor.h"
#include "onnx-executor/onnx-utils.h"

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

int main(int argc, char **argv) {
  std::string model = argv[1];
  onnx_executor::OnnxRuntimeConfig config;
  config.model = model;

  onnx_executor::OnnxExecutor executor(config);

  std::vector<float> input = generate_random_data(576);
  std::vector<int64_t> input_shape{1, 576};
  std::vector<float> state(256, 0);
  std::vector<int64_t> state_shape{2, 1, 128};
  int64_t sr = 16000;
  int64_t sr_shape = 1;

  Ort::Value input_tensor = onnx_executor::CreateTensor(
      input.data(), input.size(), input_shape.data(), input_shape.size());
  Ort::Value state_tensor = onnx_executor::CreateTensor(
      state.data(), state.size(), state_shape.data(), state_shape.size());
  Ort::Value sr_tensor =
      onnx_executor::CreateTensor<int64_t>(&sr, 1, &sr_shape, 1);

  std::vector<Ort::Value> inputs(3);
  inputs[0] = std::move(input_tensor);
  inputs[1] = std::move(state_tensor);
  inputs[2] = std::move(sr_tensor);

  std::vector<Ort::Value> output = executor.Forward(std::move(inputs));
  float logit = output[0].GetTensorMutableData<float>()[0];

  std::cout << "logit: " << logit << std::endl;
}