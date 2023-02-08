#include "train.hh"
#include "dataset.hh"

using fl::pkg::runtime::buildSequentialModule;
using fl::lib::audio::Mfcc;
using fl::lib::audio::FeatureParams;
namespace za{
    DEFINE_string(model_config, "model.conf", "Path to Config of Model");
    DEFINE_uint32(num_feature, 13, "Number of extracted features");

    Train::Train(std::shared_ptr<fl::Sequential>& model, 
                  std::unique_ptr<VADDataset>& dataset,
                  std::unique_ptr<fl::CategoricalCrossEntropy>& loss_function,
                  std::unique_ptr<fl::FirstOrderOptimizer>& optimizer,
                  const uint16_t max_epochs){
                    
                  }

    void Train::start_train_process(){

    }

    void Train::step(){

    }

    void Train::start_of_epoch(){

    }

    void Train::end_of_epoch(){

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