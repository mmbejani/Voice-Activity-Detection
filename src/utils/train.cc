#include "train.hh"

using fl::pkg::runtime::buildSequentialModule;
using fl::lib::audio::Mfcc;
using fl::lib::audio::FeatureParams;

int main(int argc, char **argv){
    fl::init();
    google::InitGoogleLogging(argv[0]);
    google::LogToStderr();
    LOG(INFO) << "Parsing command line flags";
    gflags::ParseCommandLineFlags(&argc, &argv, false);

    LOG(INFO) << "Loading Model";
    auto model = buildSequentialModule(FLAGS_model_config, FLAGS_num_feature, 2);
    
    LOG(INFO) << "Create Feature Params";
    auto featureParam = FeatureParams();
    auto mfcc = std::make_shared<Mfcc>(featureParam);

    
    return 0;
}