#include "train.hh"
#include "dataset.hh"
#include "inference.hh"

using fl::lib::audio::Mfcc;
using fl::lib::audio::FeatureParams;
using namespace boost::filesystem;

namespace za{

    DEFINE_string(model_config, "model.conf", "Path to Config of Model");
    DEFINE_uint32(num_feature, 13, "Number of extracted features");
    DEFINE_string(checkpoint_path, "/home/mahdi/Desktop", "The root file that all of the checkpoint is going to be saved there");

    Train::Train(std::shared_ptr<Vad> vad, 
                 std::shared_ptr<fl::BatchDataset> dataset,
                 std::shared_ptr<fl::MSE> loss_function,
                 std::shared_ptr<fl::FirstOrderOptimizer> optimizer,
                 uint16_t max_epochs) : vad(vad),
                                        dataset(dataset),
                                        loss_function(loss_function),
                                        optimizer(optimizer),
                                        max_epochs(max_epochs){
                    LOG(INFO) << "Initializing Traning procedure ...";
                    LOG(INFO) << "Try to loading the last checkpoint";
                    try{
                        //load_model();
                    }catch(...){
                        LOG(WARNING) << "Cannot load model from checkpoint";
                    }
                }

    void Train::start_train_process(){
        this->vad->model->train();
        for (size_t epoch = 0; epoch < this->max_epochs; epoch++)
        {
            unsigned int iteration = 1;
            this->start_of_epoch(epoch);
            for (auto &batch : *this->dataset){
                auto loss_array = this->step(batch);
                auto loss = loss_array.host<float>();
                LOG(INFO) << "In iteration " << iteration << " loss value becomes " << *loss;
                iteration++;
            }
            //this->end_of_epoch(epoch);
        }
    }

    af::array Train::step(std::vector<af::array>& batch){
        auto inputs = fl::noGrad(batch[0]).linear();
        auto input_lengths = batch[1];
        auto targets = fl::noGrad(batch[2]).linear();

        //forward path
        cout << inputs.dims() << endl;
        auto tic = std::chrono::system_clock::now();
        auto outputs = (*this->vad)(inputs, input_lengths);
        auto toc = std::chrono::system_clock::now();
        cout << "Duration of inference : " << std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count() << endl;
        outputs = outputs(af::span, 1);
        this->optimizer->zeroGrad();
        auto loss = this->loss_function->forward(outputs, targets);
        //backward path
        loss.backward();
        //optimization step
        fl::clipGradNorm(this->vad->model->params(), 100.0);
        this->optimizer->step();
        return loss.array();
    }

    void Train::start_of_epoch(const size_t num_epoch){
        LOG(INFO) << "Epoch " << num_epoch << " is started";
    }

    void Train::end_of_epoch(const size_t num_epoch) const {
        LOG(INFO) << "Epoch " << num_epoch << " is ended";
        auto checkpoint_save_fmt = boost::format("%1%/vad-epoch-%2%.bin") % 
                                                FLAGS_checkpoint_path % 
                                                num_epoch;
        LOG(INFO) << "The model is going to be saved in checkpoint directory with name " <<
                        checkpoint_save_fmt.str();
        
        Serializer::save(checkpoint_save_fmt.str(), "v0.0.0", this->vad->model);
        return;
    }

    void Train::load_model(){
        auto checkpoint_path = find_last_checkpoint();
        Serializer::load(checkpoint_path, this->vad->model);
        return;
    }

    std::string Train::find_last_checkpoint() const{
        path last_checkpoint_path(FLAGS_checkpoint_path);
        for(auto i = directory_iterator(path(FLAGS_checkpoint_path));
            i != directory_iterator(); i++){
                if (last_checkpoint_path < i->path())
                    last_checkpoint_path = i->path();
            }
        return (boost::format("%1%/%2%")
                    % FLAGS_checkpoint_path 
                    % last_checkpoint_path.filename().string())
                    .str();
    }
}