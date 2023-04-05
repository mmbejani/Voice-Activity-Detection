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

    auto model = new za::Inference("/home/mahdi/Project/Voice-Activity-Detection/src/model.conf");
    
    float data[16000];
    for (int i = 0; i < 16000; i++)
        data[i] = 0;
    
    fl::Variable tensor(af::array(af::dim4(1,16000), data), false);
    auto output = model->infer(tensor);

    print_dim_af(output.array());
    
    return 0;
}