#include <flashlight/fl/flashlight.h>
#include <flashlight/lib/audio/feature/Mfcc.h>
#include <flashlight/lib/audio/feature/FeatureParams.h>

#include <flashlight/pkg/runtime/common/SequentialBuilder.h>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include "dataset.hh"

#include <chrono>


namespace za{
    DECLARE_string(model_config);
    DECLARE_uint32(num_feature);

    class Train {

        public:
            Train(std::shared_ptr<fl::Sequential>& model, 
                  std::shared_ptr<VADDataset>& dataset,
                  std::shared_ptr<fl::BinaryCrossEntropy>& loss_function,
                  std::shared_ptr<fl::FirstOrderOptimizer>& optimizer,
                  uint16_t max_epochs);

            void start_train_process();

        private:
            af::array& step(std::vector<af::array>&);
            void train();
            void start_of_epoch();
            void end_of_epoch();

            std::shared_ptr<fl::Sequential>& model;
            std::shared_ptr<fl::BatchDataset>& batch_dataset;
            std::shared_ptr<fl::BinaryCrossEntropy>& loss_function;
            std::shared_ptr<fl::FirstOrderOptimizer>& optimizer;
            uint16_t max_epochs;
    };
}