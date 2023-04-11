#include "inference.hh"

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
        // Compute Mfcc
        auto mfcc = vad->featureExtractor(vector_data);
        auto tensor = fl::Variable(mfcc, false);
        return infer(tensor);
    }

    fl::Variable Inference::infer(string&& path_to_utterance){
        ifstream input_stream(path_to_utterance);
        return infer(input_stream);
    }

    fl::Variable Inference::inferBatch(const vector<string>& path_to_utterances){
        vector<istream> streams;
        for (auto& path: path_to_utterances)
            streams.push_back(ifstream(path));
        return inferBatch(streams);
    }

    fl::Variable Inference::inferBatch(vector<istream>& streams){
        float max_length = 0;
        vector<float>lengths;
        vector<vector<float>> signals;
        for(auto& stream: streams){
            auto data = loadSound<float>(stream);
            lengths.push_back(data.size());
            if (max_length < data.size())
                max_length = data.size();

            signals.push_back(data);
        }

        for(auto& length: lengths)
            length /= max_length;

        auto af_lengths = af::array(af::dim4(1, lengths.size()), lengths.data());

        auto result = (*this->vad)(signals, af_lengths); 
        return result;
    }

    fl::Variable Inference::inferBatch(const fl::Variable& input, const af::array& input_size){
        return (*this->vad)(input, input_size);
    }
}