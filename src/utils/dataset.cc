#include "dataset.hh"

namespace za{

    DEFINE_uint32(batch_size, 4, "Size of batch size");
    DEFINE_string(manifest_path, "/home/mahdi/data/with-noise.json", "Path to tsv (TAB Seperated Values) meta data for training");

    VADDataset::VADDataset(std::string &manifest_path, uint32_t batch_size){
        auto featureParams = FeatureParams();
        featureParams.useEnergy = false;
        featureParams.usePower = false;
        featureParams.zeroMeanFrame = false;
        this->preprocessor = inputFeatures(featureParams,
                                            fl::pkg::speech::FeatureType::MFCC,
                                            {-1, -1});
        this->audioLoader = [=](const std::string &audioPath){
                   std::vector<float>data = fl::pkg::speech::loadSound<float>(audioPath);
                   auto vec_data = static_cast<void*>(data.data());
                   auto feature = this->preprocessor(
                        vec_data, af::dim4(1, data.size()), af::dtype::f32);
                   return feature;
                };

        this->audioCollator = [](const std::vector<af::array> &dataList){
            unsigned int max_len_data = 0;
            const int cat_dim = 2;
            const af::dim4 zero_dim(0,0,0,0);
            for (size_t i = 0; i < dataList.size(); i++)
                if (max_len_data < dataList[i].dims(0))
                    max_len_data = dataList[i].dims(0);

            std::vector<af::array> pad_arrays;
            for (size_t i = 0; i < dataList.size(); i++){
                auto array = dataList[i];
                af::dim4 pad_size(max_len_data - array.dims(0), 0, 0, 0);
                auto pad_array = af::pad(dataList[i], 
                                         pad_size, 
                                         zero_dim,
                                         af::borderType::AF_PAD_ZERO);
                pad_arrays.push_back(pad_array);
            }

            af::array pad_array = za::cat_array(cat_dim, pad_arrays);
            pad_array = af::reorder(pad_array, 0, 3, 1, 2);

            return pad_array;
        };

        this->lengthCollator = [](const std::vector<af::array> &labelList){
            af::array pad_array = af::array(za::cat_array(1, labelList));
            return pad_array;
        };
        
        LOG(INFO) << "Reading Manifest From " << FLAGS_manifest_path << " ...";
        try{
            std::unique_ptr<std::ifstream> reader = 
                std::make_unique<std::ifstream>(FLAGS_manifest_path);
            Json::Value root;
            Json::CharReaderBuilder builder;

            builder["collectComments"] = true;
            JSONCPP_STRING errs;
            Json::parseFromStream(builder, *reader, &root, &errs);
            for (unsigned int i = 0; i < root.size(); i++)
            {
                this->audio_paths.push_back(root[i]["audio_path"].asString());
                this->audio_labels.push_back(root[i]["audio_label"].asInt());
            }

            reader->close();
        }catch(...){
            LOG(INFO) << "Manifest file is not readable ...";
            exit(-1);
        }
    }

    int64_t VADDataset::size() const { return this->audio_labels.size(); }

    std::vector<af::array> VADDataset::get(const int64_t idx) const {
        constexpr float one[1] = {1};
        constexpr float zero[1] = {1};
        auto audioData = this->audioLoader(this->audio_paths[idx]);
        auto audioLength = af::array(1, new float[1]{(float)audioData.dims(0)});
        
        return {audioData,
                audioLength,
                this->audio_labels[idx] == 1 ? af::array(1, one) : af::array(1, zero)};
    }

    std::shared_ptr<VADDataset> getDataset(){
        return std::make_shared<VADDataset>();
    }
}