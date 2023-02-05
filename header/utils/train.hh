#include <flashlight/fl/flashlight.h>
#include <flashlight/lib/audio/feature/Mfcc.h>
#include <flashlight/lib/audio/feature/FeatureParams.h>

#include <flashlight/pkg/runtime/common/SequentialBuilder.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_string(model_config, "model.conf", "Path to Config of Model");

DEFINE_uint32(num_feature, 13, "Number of extracted features");