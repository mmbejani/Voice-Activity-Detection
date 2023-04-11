#include <arrayfire.h>

#include <glog/logging.h>

using namespace std;

namespace za {
    af::array cat_array(const int dim, const vector<af::array>& arrays);
    af::array pad_cat_array(const int dim, const vector<af::array>& arrays, int max_size);
}