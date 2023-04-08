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
                                   {dataset->audioCollator});

    auto tic = std::chrono::system_clock::now();
    for (auto& b: loader)
    {
        print_dim_af(b[0]);
    }
    auto toc = std::chrono::system_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();*/
    
    auto model = new za::Inference("/home/mahdi/Project/Voice-Activity-Detection/src/model.conf");
    
    //auto r = af::randn(af::dim4(400, 1, 13, 1));
    //fl::Variable tensor(r, false);
    
    auto output = model->infer("/home/mahdi/Downloads/a.wav");
    int times = 0;
    for(int i = 0; i <100;i++){
        auto tic = std::chrono::steady_clock::now();
        auto output = model->infer("/home/mahdi/Downloads/a.wav");
        auto toc = std::chrono::steady_clock::now();
        times += std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
        print_dim_af(output.array());
    }

    cout << times / 100;
    
    
    return 0;
}