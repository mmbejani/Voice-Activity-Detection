#include <flashlight/fl/flashlight.h>
#include <flashlight/lib/audio/feature/Mfcc.h>
#include <flashlight/lib/audio/feature/FeatureParams.h>

#include <flashlight/pkg/runtime/common/SequentialBuilder.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <dataset.hh>


namespace za{
    DECLARE_string(model_config);
    DECLARE_uint32(num_feature);

    class Train {

        public:
            Train(std::shared_ptr<fl::Sequential>& model, 
                  std::unique_ptr<VADDataset>& dataset,
                  std::unique_ptr<fl::CategoricalCrossEntropy>& loss_function,
                  std::unique_ptr<fl::FirstOrderOptimizer>& optimizer,
                  const uint16_t max_epochs);

            void start_train_process();

        private:
            void step();
            void start_of_epoch();
            void end_of_epoch();
    };
}