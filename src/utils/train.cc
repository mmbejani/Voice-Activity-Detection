#include "train.hh"
#include "dataset.hh"
#include "inference.hh"

using fl::lib::audio::Mfcc;
using fl::lib::audio::FeatureParams;
namespace za{
    DEFINE_string(model_config, "model.conf", "Path to Config of Model");
    DEFINE_uint32(num_feature, 13, "Number of extracted features");
    DEFINE_string(checkpoint_path, "", "The root file that all of the checkpoint is going to be saved there");

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
                    LOG(INFO) << "Try to loading the last checkpoint";
                }

    void Train::start_train_process(){
        for (size_t epoch = 0; epoch < this->max_epochs; epoch++)
        {
            this->start_of_epoch(epoch);
            for (auto &batch : *this->batch_dataset)
                this->step(batch);
            this->end_of_epoch(epoch);
        }
    }

    af::array& Train::step(std::vector<af::array>& batch){
        auto inputs = fl::Variable(batch[0], false);
        auto input_lengths = batch[1];
        auto targets = fl::Variable(batch[2], false);

        //forward path
        auto outputs = (*this->vad)(inputs, input_lengths);
        auto loss = this->loss_function->forward(outputs, targets);

        //backward path
        this->optimizer->zeroGrad();
        loss.backward();

        //optimization step
        this->optimizer->step();

        return loss.array();
    }

    void Train::start_of_epoch(const size_t num_epoch){
        LOG(INFO) << "Epoch " << num_epoch << " is started";
    }

    void Train::end_of_epoch(const size_t num_epoch){
        LOG(INFO) << "Epoch " << num_epoch << " is ended";
        LOG(INFO) << "The model is going to be saved in checkpoint directory ...";
        auto checkpoint_save_fmt = boost::format("%1%/vad-%2%.bin") % 
                                                FLAGS_checkpoint_path % 
                                                num_epoch;
        fl::save(checkpoint_save_fmt.str(), this->vad);
    }
}