#include "train.hh"
#include "dataset.hh"
#include "inference.hh"

using fl::lib::audio::Mfcc;
using fl::lib::audio::FeatureParams;
namespace za{
    DEFINE_string(model_config, "model.conf", "Path to Config of Model");
    DEFINE_uint32(num_feature, 13, "Number of extracted features");

    Train::Train(std::shared_ptr<Vad> vad, 
                 std::shared_ptr<fl::BatchDataset> dataset,
                 std::shared_ptr<fl::BinaryCrossEntropy> loss_function,
                 std::shared_ptr<fl::FirstOrderOptimizer> optimizer,
                 uint16_t max_epochs) : vad(vad),
                                         dataset(dataset),
                                         loss_function(loss_function),
                                         optimizer(optimizer),
                                         max_epochs(max_epochs){
                    LOG(INFO) << "Initializing Traning procedure ...";
                    
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
        
        auto outputs = (*this->vad)(inputs);
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