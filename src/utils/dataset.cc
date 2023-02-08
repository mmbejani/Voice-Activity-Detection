#include "dataset.hh"

DEFINE_uint32(batch_size, 32, "Size of batch size");
DEFINE_string(manifest_path, "/train_manifest.json", "Path to tsv (TAB Seperated Values) meta data for training");

namespace za{

    VADDataset::VADDataset(std::string &manifest_path, uint32_t batch_size){
        this->audioLoader = [](const std::string &audioPath){
                   std::vector<float>data = fl::pkg::speech::loadSound<float>(audioPath);
                   af::array af_data(data.size(), af::dtype::f32);
                   af_data.write(data.data(), data.size() * sizeof(float));
                   return af_data;
                };

        this->audioCollator = [](const std::vector<af::array> &dataList){
            unsigned int max_len_data = 0;
            for (size_t i = 0; i < dataList.size(); i++)
                if (max_len_data < dataList[i].dims(0))
                    max_len_data = dataList[i].dims(0);

            std::vector<af::array> pad_arrays;
            for (size_t i = 0; i < dataList.size(); i++){
                auto array = dataList[i];
                af::dim4 pad_size(max_len_data - array.dims(0), 1, 1, 1);
                auto pad_array = af::pad(dataList[i], 
                                         pad_size, 
                                         af::dim4(0,0,0,0),
                                         af::borderType::AF_PAD_ZERO);
                pad_arrays.push_back(pad_array);
            }

            af::array pad_array = cat_array(pad_arrays);
            return pad_array;
        };

        LOG(INFO) << "Reading Manifest ...";
        auto reader = std::make_unique<std::ifstream>(manifest_path, std::ios_base::openmode::_S_in);


        std::string line;
        while (std::getline(*reader, line))
        {
            std::vector<std::string> elements;
            boost::algorithm::split(elements, line, boost::is_any_of("\t"));

            this->audio_paths.push_back(elements[0]);
            //FIXME: read labels and turn them to arrayfire
            std::vector<unsigned int> labels;

            for (size_t i = 0; i < elements.size(); i++)
                labels.push_back(boost::lexical_cast<unsigned int>(elements[i]));

            auto labels_array = std::make_unique<af::array>(labels.size(), af::dtype::u16);
            labels_array->write(labels.data(), labels.size() * sizeof(unsigned int));
        }

        reader->close();

    }

     int64_t VADDataset::size() const { return this->audio_labels.size(); }

     std::vector<af::array> VADDataset::get(const int64_t idx) const {
        return {this->audioLoader(this->audio_paths[idx]),
            this->audio_labels[idx]};
    }

    std::unique_ptr<VADDataset> getDataset(){
        return std::make_unique<VADDataset>();
    }

    af::array cat_array(std::vector<af::array>& arrays){
        return arrays[0];
    }    
}