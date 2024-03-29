#include "vad.hh"

namespace za{
    
    Vad::Vad(std::shared_ptr<fl::Sequential> model): model(model){
      /**
       * int64_t samplingfreq = 16000L,
       *  int64_t framesizems = 25L,
       *  int64_t framestridems = 10L,
       *  int64_t numfilterbankchans = 23L,
       *  int64_t lowfreqfilterbank = 0L,
       *  int64_t highfreqfilterbank = -1L,
       *  int64_t numcepstralcoeffs = 13L,
       *  int64_t lifterparam = 22L,
       *  int64_t deltawindow = 2L,
       *  int64_t accwindow = 2L,
       *  fl::lib::audio::WindowType windowtype = fl::lib::audio::WindowType::HAMMING,
       *  float preemcoef = (0.9700000286F),
       *  float melfloor = (1.0F),
       *  float ditherval = (0.0F),
       *  bool usepower = true,
       *  bool usenergy = true,
       *  bool rawenergy = true,
       *  bool zeromeanframe = true
      */
      auto featureParams = FeatureParams();

      featureParams.useEnergy = false;
      featureParams.usePower = false;
      featureParams.zeroMeanFrame = false;
      this->mfccFeature = std::make_shared<Mfcc>(featureParams);
    }

    fl::Variable Vad::operator()(const std::vector<std::vector<float>>& input_signals,
                                 const af::array& input_sizes) const {
        vector<af::array> features;
        for (int i = 0; i < input_signals.size(); i++)
          features.push_back(featureExtractor(input_signals[i]));

        auto mfccFeatures = fl::Variable(za::pad_cat_array(1, features, -1), false);
        return forward(mfccFeatures, input_sizes);
    }

    fl::Variable Vad::operator()(const fl::Variable& input_mfccs,
                                 const af::array& input_sizes) const {
      return forward(input_mfccs, input_sizes);
    }

    af::array Vad::featureExtractor(const vector<float>& input_signal) const {
      auto output_dim = af::dim4(mfccFeature->getFeatureParams().numFrames(input_signal.size()),
                                 1, mfccFeature->getFeatureParams().numCepstralCoeffs);
      auto feature = af::array(output_dim,
                        this->mfccFeature->apply(input_signal).data());
      return feature;
    }

    fl::Variable Vad::forward(
        const fl::Variable& mfccFeature,
        const af::array& input_sizes) const {

      // expected input dims T x C x 1 x B
      auto output = mfccFeature;
      for (auto& module : this->model->modules()) {
          output = module->forward({output}).front();
        }
        return output;
      }
}