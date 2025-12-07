#ifndef ONNX_EXECUTOR_MACROS_H_
#define ONNX_EXECUTOR_MACROS_H_

#include <cstdio>
#include <cstdlib>

#define ONNX_EXECUTOR_LOGE(...)                      \
  do {                                               \
    fprintf(stderr, "%s:%s:%d ", __FILE__, __func__, \
            static_cast<int>(__LINE__));             \
    fprintf(stderr, ##__VA_ARGS__);                  \
    fprintf(stderr, "\n");                           \
  } while (0)

#endif  // ONNX_EXECUTOR_MACROS_H_