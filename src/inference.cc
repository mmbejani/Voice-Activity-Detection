#include "inference.hh"

namespace za {
    Inference::Inference(const string& path_to_model){
        auto model = buildSequentialModule(path_to_model, 13, 2);
        this->vad = std::make_shared<Vad>(model);
    }
}