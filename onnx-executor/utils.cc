#include "onnx-executor/utils.h"

#include <fstream>

namespace onnx_executor {

std::vector<char> ReadFile(const std::string &filename) {
  std::ifstream input(filename, std::ios::binary);
  std::vector<char> buffer(std::istreambuf_iterator<char>(input), {});
  return buffer;
}

void SplitStringToVector(const std::string &full, const char *delim,
                         bool omit_empty_strings,
                         std::vector<std::string> *out) {
  size_t start = 0, found = 0, end = full.size();
  out->clear();
  while (found != std::string::npos) {
    found = full.find_first_of(delim, start);
    // start != end condition is for when the delimiter is at the end
    if (!omit_empty_strings || (found != start && start != end))
      out->push_back(full.substr(start, found - start));
    start = found + 1;
  }
}

template <class F>
bool SplitStringToFloats(const std::string &full, const char *delim,
                         bool omit_empty_strings,  // typically false
                         std::vector<F> *out) {
  assert(out != nullptr);
  if (*(full.c_str()) == '\0') {
    out->clear();
    return true;
  }
  std::vector<std::string> split;
  SplitStringToVector(full, delim, omit_empty_strings, &split);
  out->resize(split.size());
  for (size_t i = 0; i < split.size(); ++i) {
    // assume atof never fails
    F f = 0;
    if (!ConvertStringToReal(split[i], &f)) return false;
    (*out)[i] = f;
  }
  return true;
}

// Instantiate the template above for float and double.
template bool SplitStringToFloats(const std::string &full, const char *delim,
                                  bool omit_empty_strings,
                                  std::vector<float> *out);
template bool SplitStringToFloats(const std::string &full, const char *delim,
                                  bool omit_empty_strings,
                                  std::vector<double> *out);

}  // namespace onnx_executor
