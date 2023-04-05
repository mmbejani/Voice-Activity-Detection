#include "utils.hh"

namespace za {
    af::array cat_array(const int dim, const std::vector<af::array>& arrays){
        if (arrays.size() == 1) [[unlikely]]
            return arrays[0];

        auto cat = af::join(dim, arrays[0], arrays[1]);
        for (unsigned int i = 2; i < arrays.size(); i++)
            cat = af::join(dim, cat, arrays[i]);
        
        [[likely]]return cat;
    }

    af::array pad_cat_array(const int dim, const vector<af::array>& arrays, int max_size){
        if (max_size == -1) {
            for (int i = 0; i < arrays.size();i++)
                if (max_size > arrays[i].dims(dim))
                    max_size = arrays[i].dims(dim);
        }

        vector<af::array> pad_arrays;
        af::dim4 begin_pad(0,0,0,0);
        for (int i = 0; i < arrays.size(); i++)
        {
            af::dim4 end_pad(0,0,0,0);
            end_pad[dim] = max_size - arrays[i].dims(dim);
            pad_arrays.push_back(
                af::pad(arrays[i], begin_pad, end_pad, af::borderType::AF_PAD_ZERO)
            );
        }
        
        return cat_array(dim, pad_arrays);       
    }
}