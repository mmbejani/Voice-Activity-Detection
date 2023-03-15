#pragma once
#include <flashlight/fl/flashlight.h>
#include <flashlight/fl/contrib/contrib.h>
#include <flashlight/fl/contrib/modules/modules.h>
#include <flashlight/lib/audio/feature/Mfcc.h>
#include <flashlight/lib/audio/feature/FeatureParams.h>

using namespace fl::lib::audio;

namespace za{

    class Vad
    {
    private:
        std::shared_ptr<Mfcc> MfccFeature;

        std::shared_ptr<fl::Sequential> model;

    public:
        Vad(std::shared_ptr<fl::Sequential> model);
        
        fl::Variable operator()(const fl::Variable& input_signals, const af::array& input_sizes) const;
        fl::Variable featureExtractor(const fl::Variable& input_signal) const;
        fl::Variable forward(const fl::Variable&,const af::array&) const;
    };
}