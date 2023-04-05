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

    af::array pad_array(const int dim, const std::vector<af::array>& arrays){
        int max_size = 0;
        for (int i = 0; i < arrays.size();i++)
        {
            /* code */
        }
        
    }
}