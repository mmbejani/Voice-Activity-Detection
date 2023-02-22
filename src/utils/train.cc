#include "train.hh"
#include "dataset.hh"

using fl::pkg::runtime::buildSequentialModule;
using fl::lib::audio::Mfcc;
using fl::lib::audio::FeatureParams;
namespace za{
    DEFINE_string(model_config, "model.conf", "Path to Config of Model");
    DEFINE_uint32(num_feature, 13, "Number of extracted features");

    Train::Train(std::shared_ptr<fl::Sequential>& model, 
                  std::shared_ptr<VADDataset>& dataset,
                  std::shared_ptr<fl::BinaryCrossEntropy>& loss_function,
                  std::shared_ptr<fl::FirstOrderOptimizer>& optimizer,
                  uint16_t max_epochs) {
                    LOG(INFO) << "Initializing Traning procedure ...";
                    this->model = std::make_shared<fl::Sequential>(model);
                    this->batch_dataset = std::make_shared<fl::BatchDataset>
                                            (dataset, FLAGS_batch_size, dataset->audioCollator);
                    this->loss_function = std::make_shared<fl::BinaryCrossEntropy>(loss_function);
                    this->optimizer = std::make_shared<fl::FirstOrderOptimizer>(optimizer);
                    this->max_epochs = max_epochs;
                }

    void Train::start_train_process(){
        for (size_t epoch = 0; epoch < this->max_epochs; epoch++)
        {
            this->start_of_epoch();
            for (auto &batch : *this->batch_dataset)
                this->step(batch);
            this->end_of_epoch();
        }
    }

    af::array& Train::step(std::vector<af::array>& batch){
        auto inputs = fl::Variable(batch[0], false);
        auto targets = fl::Variable(batch[1], false);

        //forward path
        auto outputs = this->model->forward(inputs);                                    
        auto loss = this->loss_function->forward(outputs, targets);

        //backward path
        loss.backward();

        //optimization step
        this->optimizer->zeroGrad();
        this->optimizer->step();

        return loss.array();
    }

    void Train::start_of_epoch(){

    }

    void Train::end_of_epoch(){
        fl::save("vad.bin", this->model);
    }
}

int main(int argc, char **argv){
    fl::init();
    google::InitGoogleLogging(argv[0]);
    google::LogToStderr();
    LOG(INFO) << "Parsing command line flags ...";
    gflags::ParseCommandLineFlags(&argc, &argv, false);

    LOG(INFO) << "Loading Model ...";
    auto model = buildSequentialModule(za::FLAGS_model_config, za::FLAGS_num_feature, 2);
    
    LOG(INFO) << "Creating Feature Params ...";
    auto featureParam = FeatureParams();
    auto mfcc = std::make_shared<Mfcc>(featureParam);

    LOG(INFO) << "Creating Dataset ...";
    
    return 0;
}