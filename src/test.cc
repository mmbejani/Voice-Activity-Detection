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
    auto loader = fl::BatchDataset(dataset, FLAGS_batch_size,
                                   fl::BatchDatasetPolicy::INCLUDE_LAST,
                                   {dataset->audioCollator});

    for (auto &&batch : loader)
    {
        print_dim_af(batch[0]);
        af::print("Audio size",batch[1]);
        af::print("Audio label", batch[2]);
        std::cout << "==========================" << std::endl;
    }
    

    return 0;
}