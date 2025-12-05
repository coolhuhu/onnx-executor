#include "onnx-executor/utils.h"

#include <fstream>

namespace onnx_executor {

std::vector<char> ReadFile(const std::string &filename) {
  std::ifstream input(filename, std::ios::binary);
  std::vector<char> buffer(std::istreambuf_iterator<char>(input), {});
  return buffer;
}

}  // namespace onnx_executor
