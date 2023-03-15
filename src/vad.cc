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
      this->MfccFeature = std::make_shared<Mfcc>(FeatureParams());
    }

    fl::Variable Vad::operator()(const fl::Variable& input_signals,
                                 const af::array& input_sizes) const {
        auto mfccFeatures = featureExtractor(input_signals);
        return forward(mfccFeatures, input_sizes);
    }


    fl::Variable Vad::forward(
        const fl::Variable& mfccFeature,
        const af::array& input_sizes) const {

      // expected input dims T x C x 1 x B
      int T = mfccFeature.dims(0), B = mfccFeature.dims(3);
      int B = mfccFeature.dims(3);
      auto inputMaxSize = af::tile(af::max(input_sizes), 1, B);
      af::array inputNotPaddedSize = af::ceil(input_sizes * T / inputMaxSize);
      auto padMask = af::iota(af::dim4(T, 1), af::dim4(1, B)) <
          af::tile(inputNotPaddedSize, T, 1);
      af::array inputNotPaddedSizeRatio = input_sizes / inputMaxSize;
      auto output = mfccFeature;
      for (auto& module : this->model->modules()) {
        auto tr = std::dynamic_pointer_cast<fl::Transformer>(module);
        auto cfr = std::dynamic_pointer_cast<fl::Conformer>(module);
        if (tr != nullptr || cfr != nullptr) {
          /* input dims of Transformer module should be CxTxBx1 */
          int T = output.dims(1);
          auto inputNotPaddedSize = af::ceil(T * inputNotPaddedSizeRatio);
          auto padMask = af::iota(af::dim4(T, 1), af::dim4(1, B)) <
                     af::tile(inputNotPaddedSize, T, 1);
          output = module->forward({output, fl::noGrad(padMask)}).front();
        } else {
          output = module->forward({output}).front();
        }
      }
      return output;
    }
}