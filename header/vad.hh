#pragma once
#include <flashlight/fl/flashlight.h>
#include <flashlight/fl/contrib/contrib.h>
#include <flashlight/fl/contrib/modules/modules.h>
#include <flashlight/lib/audio/feature/Mfcc.h>
#include <flashlight/lib/audio/feature/FeatureParams.h>

using namespace fl::lib::audio;
using namespace std;

namespace za{

    class Vad
    {
    private:
        std::shared_ptr<Mfcc> mfccFeature;

    public:
        std::shared_ptr<fl::Sequential> model;

        Vad(std::shared_ptr<fl::Sequential> model);
        
        fl::Variable operator()(const vector<vector<float>>& input_signals, const af::array& input_sizes) const;
        af::array featureExtractor(const vector<float>& input_signal) const;
        fl::Variable forward(const fl::Variable&,const af::array&) const;
    };
}