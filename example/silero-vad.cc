#include <iostream>
#include <random>

#include "onnx-executor/onnx-executor.h"

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

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor =
      Ort::Value::CreateTensor(memory_info, input.data(), input.size(),
                               input_shape.data(), input_shape.size());

  Ort::Value state_tensor =
      Ort::Value::CreateTensor(memory_info, state.data(), state.size(),
                               state_shape.data(), state_shape.size());
  Ort::Value sr_tensor =
      Ort::Value::CreateTensor<int64_t>(memory_info, &sr, 1, &sr_shape, 1);

  std::vector<Ort::Value> inputs;
  inputs.emplace_back(std::move(input_tensor));
  inputs.emplace_back(std::move(state_tensor));
  inputs.emplace_back(std::move(sr_tensor));

  std::vector<Ort::Value> output = executor.Forward(std::move(inputs));
  float logit = output[0].GetTensorMutableData<float>()[0];

  std::cout << "logit: " << logit << std::endl;
}