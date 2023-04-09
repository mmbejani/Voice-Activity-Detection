#include <flashlight/fl/flashlight.h>
#include <flashlight/lib/audio/feature/Mfcc.h>
#include <flashlight/lib/audio/feature/FeatureParams.h>
#include <flashlight/pkg/runtime/common/Serializer.h>
#include <flashlight/pkg/runtime/common/SequentialBuilder.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <chrono>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include "dataset.hh"
#include "vad.hh"

using fl::pkg::runtime::Serializer;

namespace za{
    DECLARE_string(model_config);
    DECLARE_uint32(num_feature);
    DECLARE_string(checkpoint_path);

    class Train {

        public:
            Train(std::shared_ptr<Vad> vad, 
                  std::shared_ptr<fl::BatchDataset> dataset,
                  std::shared_ptr<fl::MSE> loss_function,
                  std::shared_ptr<fl::FirstOrderOptimizer> optimizer,
                  uint16_t max_epochs);

            void start_train_process();

        private:
            af::array step(std::vector<af::array>&);
            void train();
            void start_of_epoch(const size_t);
            void end_of_epoch(const size_t) const;
            std::string find_last_checkpoint() const;
            void load_model();

            std::shared_ptr<Vad> vad;
            std::shared_ptr<fl::BatchDataset> dataset;
            std::shared_ptr<fl::MSE> loss_function;
            std::shared_ptr<fl::FirstOrderOptimizer> optimizer;
            uint16_t max_epochs;
    };
}