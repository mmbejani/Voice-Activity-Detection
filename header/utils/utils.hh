#include <arrayfire.h>

namespace za {
    af::array cat_array(const int dim, const std::vector<af::array>& arrays);
    af::array pad_array(const int dim, const std::vector<af::array>& arrays);
}