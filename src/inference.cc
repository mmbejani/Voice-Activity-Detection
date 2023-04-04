#include "inference.hh"

namespace za {
    Inference::Inference(const string& path_to_model){
        auto model = buildSequentialModule(path_to_model, 13, 2);
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

        auto data = static_cast<float*>(vector_data.data());
        auto tensor = fl::Variable(af::array(af::dim4(audio_info.channels, audio_info.frames), data), false);

        return infer(tensor);
    }

    fl::Variable Inference::infer(const string& path_to_utterance){
        ifstream input_stream(path_to_utterance);
        return infer(input_stream);
    }
}