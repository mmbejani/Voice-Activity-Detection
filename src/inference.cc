#include "inference.hh"
#include <iostream>

using namespace std;

namespace za {
    Inference::Inference(const string& path_to_model){
        auto model = buildSequentialModule(path_to_model, 13, 2);
        model->eval();
        this->vad = std::make_shared<Vad>(model);
    }

    fl::Variable Inference::infer(const fl::Variable& tensor){
        auto tensor_length = af::array(1, new float[1]{1.0f});
        auto result = (*this->vad)(tensor,tensor_length);
        return result;
    }

    fl::Variable Inference::infer(ifstream& buffer){
        auto vector_data = loadSound<float>(buffer);
        buffer.seekg(0);
        auto audio_info = loadSoundInfo(buffer);
        // Compute Mfcc
        auto mfcc = vad->featureExtractor(vector_data);
        auto tensor = fl::Variable(mfcc, false);
        return infer(tensor);
    }

    fl::Variable Inference::infer(string&& path_to_utterance){
        ifstream input_stream(path_to_utterance);
        return infer(input_stream);
    }
}