#include <iostream>
#include "dataset.hh"
#include "inference.hh"

void print_dim_af(const af::array& a){
    std::cout << "The dims of input tensor is ( ";
    for (int i = 0; i < 4; i++)
        std::cout << a.dims(i) << ", ";
    std::cout << ")" << std::endl;
}

int main(int argc, char **argv){
    google::InitGoogleLogging(*argv);
    google::ParseCommandLineFlags(&argc, &argv, false);
    
    /*auto dataset = za::getDataset();
    auto loader = fl::BatchDataset(dataset, FLAGS_batch_size,
                                   fl::BatchDatasetPolicy::INCLUDE_LAST,
                                   {dataset->audioCollator});*/

    auto infer = new za::Inference("/home/mahdi/Project/Voice-Activity-Detection/src/model.conf");
    
    

    return 0;
}