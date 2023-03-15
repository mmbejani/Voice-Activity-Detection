#include <iostream>
#include "dataset.hh"

void print_dim_af(const af::array& a){
    std::cout << "The dims of input tensor is ( ";
    for (int i = 0; i < 4; i++)
        std::cout << a.dims(i) << ", ";
    std::cout << ")" << std::endl;
}

int main(int argc, char **argv){
    google::InitGoogleLogging(*argv);
    google::ParseCommandLineFlags(&argc, &argv, false);
    
      

    return 0;
}