#include <flashlight/fl/flashlight.h>
#include <flashlight/lib/audio/feature/Mfcc.h>
#include <flashlight/lib/audio/feature/FeatureParams.h>

#include <flashlight/pkg/runtime/common/SequentialBuilder.h>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include "dataset.hh"

#include <chrono>
#include <boost/format.hpp>

namespace za{
    DECLARE_string(model_config);
    DECLARE_uint32(num_feature);
    DECLARE_string(checkpoint_path);

    class Train {

        public:
            Train(std::shared_ptr<Vad> vad, 
                  std::shared_ptr<fl::BatchDataset> dataset,
                  std::shared_ptr<fl::BinaryCrossEntropy> loss_function,
                  std::shared_ptr<fl::FirstOrderOptimizer> optimizer,
                  uint16_t max_epochs);

            void start_train_process();

        private:
            af::array& step(std::vector<af::array>&);
            void train();
            void start_of_epoch(const size_t);
            void end_of_epoch(const size_t);

            std::shared_ptr<Vad> vad;
            std::shared_ptr<fl::BatchDataset> dataset;
            std::shared_ptr<fl::BatchDataset> batch_dataset;
            std::shared_ptr<fl::BinaryCrossEntropy> loss_function;
            std::shared_ptr<fl::FirstOrderOptimizer> optimizer;
            uint16_t max_epochs;
    };
}