
#include <iostream>
#include <random>
#include <thread>

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
  std::cout << "starting initialize onnx executor." << std::endl;
  onnx_executor::OnnxRuntimeConfig config;

  // fireredasr encoder
  config.model = argv[1];
  config.provider = "cuda";
  config.provider_config.device_id = 0;

  std::cout << config.model << std::endl;

  onnx_executor::OnnxExecutor onnx_executor(config);

  std::this_thread::sleep_for(std::chrono::seconds(10));

  std::cout << "onnx executor initialize done." << std::endl;

  // x'shape (1, T, 80)
  // x_len'shape (1)
  std::vector<float> x = generate_random_data(400);
  std::vector<int64_t> x_shape{1, 5, 80};
  int64_t x_len = 1;
  int64_t x_len_shape = 1;

  Ort::Value x_tensor = onnx_executor::CreateTensor(
      x.data(), x.size(), x_shape.data(), x_shape.size());
  Ort::Value x_len_tensor =
      onnx_executor::CreateTensor(&x_len, 1, &x_len_shape, 1);
  std::vector<Ort::Value> inputs(2);
  inputs[0] = std::move(x_tensor);
  inputs[1] = std::move(x_len_tensor);

  // outputs[0]: [16,Concatn_layer_cross_k_dim_1,T,1280]
  // outputs[1]: [16,Concatn_layer_cross_v_dim_1,T,1280]
  auto outputs = onnx_executor.Forward(inputs);

  Ort::ConstMemoryInfo memory_info = outputs[0].GetTensorMemoryInfo();
  OrtMemoryInfoDeviceType tensor_device_type = memory_info.GetDeviceType();
  if (tensor_device_type == OrtMemoryInfoDeviceType_CPU) {
    std::cout << "cpu" << std::endl;
  } else if (tensor_device_type == OrtMemoryInfoDeviceType_GPU) {
    std::cout << "gpu" << std::endl;
  } else {
    std::cout << "other device" << std::endl;
  }

  std::cout << "Forward done." << std::endl;
}