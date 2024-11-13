#include <string>

#include "third_party/nanobind/include/nanobind/nanobind.h"

namespace flaxlib {
std::string sum_as_string(int a, int b) {
  return std::to_string(a + b);
}

NB_MODULE(flaxlib_cc, m) {
  m.def("sum_as_string", &sum_as_string);
}
}  // namespace flaxlib