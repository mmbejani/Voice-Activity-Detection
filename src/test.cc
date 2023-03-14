#include <iostream>
#include "dataset.hh"
#include <arrayfire.h>
#include <flashlight/fl/dataset/BatchDataset.h>

void print_dim_af(const af::array& a){
    for (int i = 0; i < 4; i++)
        std::cout << "Dim " << i << " is : " << a.dims(i) << std::endl;
}

int main(int argc, char **argv){
    //auto dataset = std::make_shared<za::VADDataset>("/home/mahdi/Projects/Voice-Activity-");
    //fl::BatchDataset b_data(,);
    
    
    return 0;
}