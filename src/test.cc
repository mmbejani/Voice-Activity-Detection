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

    auto dataset = za::getDataset();
    auto data = dataset->get(2);
    for (int i = 0; i < data.size(); i++)
    {
        print_dim_af(data[i]);
    }

    af::print("Label data is : ", data[1]);
    
    return 0;
}