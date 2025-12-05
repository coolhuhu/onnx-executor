#ifndef ONNX_EXECUTOR_UTILS_H_
#define ONNX_EXECUTOR_UTILS_H_

#include <string>
#include <vector>

namespace onnx_executor {

std::vector<char> ReadFile(const std::string &filename);

}  // namespace onnx_executor

#endif  // ONNX_EXECUTOR_UTILS_H_