# ONNX Executor

A lightweight C++ library for running ONNX models using ONNX Runtime. This library is essentially a wrapper around the onnxruntime library. This library provides a simple and clean interface for loading ONNX models and performing inference.

## Features

- Simple C++ API for ONNX model inference
- Custom metadata extraction from ONNX models
- Thread-safe execution

## Build Requirements

- CMake 3.14 or higher
- C++17 compatible compiler
- ONNX Runtime (automatically downloaded by CMake or load files downloaded in advance to the third-part directory.)

## Building

```bash
# Clone the repository
git clone https://github.com/coolhuhu/onnx-executor.git
cd onnx-executor

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
# or use GPU
# cmake -DENABLE_ONNX_EXECUTOR_GPU=ON ..
make -j$(nproc)
```

## Usage

### Basic Example

```cpp
#include <iostream>
#include "onnx-executor/onnx-executor.h"
#include "onnx-executor/onnx-utils.h"

int main(int argc, char **argv) {
    // Configure the executor
    onnx_executor::OnnxRuntimeConfig config;
    config.model = "path/to/your/model.onnx";
    config.provider = "cpu";  // Use CPU execution provider
    config.num_threads = 1;   // Number of threads to use

    // Create executor
    onnx_executor::OnnxExecutor executor(config);

    // Prepare input data
    std::vector<float> input_data = { /* your input data */ };
    std::vector<int64_t> input_shape = {1, input_data.size()};

    // Create input tensor
    Ort::Value input_tensor = onnx_executor::CreateTensor(
        input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size()
    );

    // Run inference
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(std::move(input_tensor));

    std::vector<Ort::Value> outputs = executor.Forward(std::move(inputs));

    // Process outputs
    float* output_data = outputs[0].GetTensorMutableData<float>();
    std::cout << "Output: " << output_data[0] << std::endl;

    return 0;
}
```

## API Reference

### OnnxRuntimeConfig

Configuration structure for the ONNX Runtime executor:

- `model` (std::string): Path to the ONNX model file
- `provider` (std::string): Execution provider (default: "cpu", currently supported: "cpu", "cuda")
- `num_threads` (int32_t): Number of threads to use (default: 1)
- `provider_config` : Onnxruntime execution provider configuration

### OnnxExecutor

Main executor class:

```cpp
class OnnxExecutor {
public:
    explicit OnnxExecutor(const OnnxRuntimeConfig &config);
    ~OnnxExecutor();

    // Run inference with input tensors
    std::vector<Ort::Value> Forward(const std::vector<Ort::Value> &inputs);
    std::vector<Ort::Value> Forward(std::vector<Ort::Value> &&inputs);

    // Extract custom metadata from the model
    std::string LookupCustomMetaData(const char *key);
};
```

### Utility Functions

`onnx_executor::Create<T>()`: Template function to create ONNX tensors from C++ data:

```cpp
// For float tensors
Ort::Value tensor = onnx_executor::CreateTensor(
    data_ptr, data_size, shape_ptr, shape_size
);

// For int64_t tensors
Ort::Value tensor = onnx_executor::CreateTensor<int64_t>(
    data_ptr, data_size, shape_ptr, shape_size
);
```

## Project Structure

```
onnx-executor/
├── onnx-executor/           # Main library source code
│   ├── onnx-executor.h/.cc  # Main executor class
│   ├── onnx-utils.h/.cc     # Utility functions
│   ├── provider.h/.cc       # Execution provider management
│   └── utils.h/.cc          # Internal utilities
├── example/                 # Example usage
│   └── silero-vad.cc        # Silero VAD example
│   ├── silero_vad.onnx      # Silero VAD onnx model
│   ├── multi_thread_load.cc # Multithreading inference example
├── cmake/                   # CMake modules
├── third-part/             # Third-party dependencies
└── CMakeLists.txt          # Root CMake configuration
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Dependencies

- [ONNX Runtime](https://onnxruntime.ai/) - High performance cross-platform inference engine for ONNX models