#ifndef ONNX_EXECUTOR_ONNX_UTILS_H_
#define ONNX_EXECUTOR_ONNX_UTILS_H_

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace onnx_executor {

Ort::SessionOptions GetSessionOptions(int32_t num_threads,
                                      const std::string &provider_str);

/**
 * Get the input names of a model.
 *
 * @param sess An onnxruntime session.
 * @param input_names. On return, it contains the input names of the model.
 * @param input_names_ptr. On return, input_names_ptr[i] contains
 *                         input_names[i].c_str()
 */
void GetInputNames(Ort::Session *sess, std::vector<std::string> *input_names,
                   std::vector<const char *> *input_names_ptr);

/**
 * Get the output names of a model.
 *
 * @param sess An onnxruntime session.
 * @param output_names. On return, it contains the output names of the model.
 * @param output_names_ptr. On return, output_names_ptr[i] contains
 *                         output_names[i].c_str()
 */
void GetOutputNames(Ort::Session *sess, std::vector<std::string> *output_names,
                    std::vector<const char *> *output_names_ptr);

}  // namespace onnx_executor

#endif  // ONNX_EXECUTOR_ONNX_UTILS_H_