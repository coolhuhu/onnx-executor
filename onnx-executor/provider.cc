#include "onnx-executor/provider.h"

#include <algorithm>

#include "onnx-executor/macros.h"

namespace onnx_executor {

Provider StringToProvider(std::string s) {
  std::transform(s.cbegin(), s.cend(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (s == "cpu") {
    return Provider::kCPU;
  } else if (s == "cuda") {
    return Provider::kCUDA;
  } else if (s == "trt") {
    return Provider::kTRT;
  } else {
    ONNX_EXECUTOR_LOGE(
        "Currently, only cpu, cuda, and trt are supported.Unsupported string: "
        "%s. Fallback to cpu",
        s.c_str());
    return Provider::kCPU;
  }
}

}  // namespace onnx_executor