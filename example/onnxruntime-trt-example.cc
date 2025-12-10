
#include <chrono>
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

std::vector<Ort::Value> PrepareMnistModelInput() {
  std::vector<float> x = generate_random_data(784);
  std::vector<int64_t> x_shape{1, 1, 28, 28};
  Ort::Value x_tensor = onnx_executor::CreateTensor(
      x.data(), x.size(), x_shape.data(), x_shape.size());
  std::vector<Ort::Value> inputs;
  inputs.emplace_back(std::move(x_tensor));

  return inputs;
}

std::vector<Ort::Value> PrepareFireredasrEncoderInput() {
  // x'shape (1, T, 80)
  // x_len'shape (1)
  // T = 100
  std::vector<float> x = generate_random_data(800);
  std::vector<int64_t> x_shape{1, 10, 80};
  int64_t x_len = 1;
  int64_t x_len_shape = 1;

  Ort::AllocatorWithDefaultOptions allocator;
  Ort::Value x_tensor = Ort::Value::CreateTensor<float>(
      allocator, x_shape.data(), x_shape.size());
  float *p_x_tensor = x_tensor.GetTensorMutableData<float>();
  std::copy(x.begin(), x.end(), p_x_tensor);

  Ort::Value x_len_tensor =
      Ort::Value::CreateTensor<int64_t>(allocator, &x_len, 1);
  float *p_x_len_tensor = x_len_tensor.GetTensorMutableData<float>();
  *p_x_len_tensor = 1;

  // Ort::Value x_tensor = onnx_executor::CreateTensor(
  //     x.data(), x.size(), x_shape.data(), x_shape.size());
  // Ort::Value x_len_tensor =
  //     onnx_executor::CreateTensor(&x_len, 1, &x_len_shape, 1);

  std::vector<Ort::Value> inputs(2);
  inputs[0] = std::move(x_tensor);
  inputs[1] = std::move(x_len_tensor);

  return inputs;
}

int main(int argc, char **argv) {
  std::cout << "starting initialize onnx executor." << std::endl;
  onnx_executor::OnnxRuntimeConfig config;

  config.model = argv[1];
  config.provider = "cpu";
  config.provider_config.device_id = 0;
  config.provider_config.tensorrt_provider_config.trt_detailed_build_log = true;
  config.provider_config.tensorrt_provider_config.trt_profile_min_shapes =
      "x:1x10x80,x_len:1";
  config.provider_config.tensorrt_provider_config.trt_profile_max_shapes =
      "x:1x500x80,x_len:1";
  config.provider_config.tensorrt_provider_config.trt_profile_opt_shapes =
      "x:1x10x80,x_len:1";

  std::cout << config.model << std::endl;

  auto start_time = std::chrono::steady_clock::now();
  onnx_executor::OnnxExecutor onnx_executor(config);
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - start_time)
                      .count();
  std::cout << "onnx executor initialize cost " << duration << " ms"
            << std::endl;

  std::cout << "onnx executor initialize done." << std::endl;

  // std::vector<Ort::Value> inputs = PrepareMnistModelInput();
  std::vector<Ort::Value> inputs = PrepareFireredasrEncoderInput();
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