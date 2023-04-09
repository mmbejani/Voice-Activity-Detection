#include <iostream>
#include "dataset.hh"
#include "inference.hh"

using namespace za;

void print_dim_af(const af::array& a){
    std::cout << "The dims of input tensor is ( ";
    for (int i = 0; i < 4; i++)
        std::cout << a.dims(i) << ", ";
    std::cout << ")" << std::endl;
}

void f(int argc, char **argv){
    /*google::InitGoogleLogging(*argv);
    google::ParseCommandLineFlags(&argc, &argv, false);
    fl::init();
    
    auto dataset = za::getDataset();
    auto loader = fl::BatchDataset(dataset, FLAGS_batch_size,
                                   fl::BatchDatasetPolicy::INCLUDE_LAST,
                                   {dataset->audioCollator, dataset->lengthCollator});

    auto model = new za::Inference("/home/mahdi/Project/Voice-Activity-Detection/src/model.conf");
    auto tic = std::chrono::system_clock::now();
    for (auto& b: loader)
    {
        auto input = fl::Variable(b[0], false).linear();
        auto input_size = b[1];
        model->inferBatch(input, input_size);
    }
    auto toc = std::chrono::system_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    
    
    
    auto output = model->infer("/home/mahdi/Downloads/a.wav");
    int times = 0;
    for(int i = 0; i <100;i++){
        auto tic = std::chrono::steady_clock::now();
        auto output = model->infer("/home/mahdi/Downloads/a.wav");
        auto toc = std::chrono::steady_clock::now();
        times += std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
        print_dim_af(output.array());
    }

    cout << times / 100;*/
    
    //return 0;
}