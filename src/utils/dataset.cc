#include "dataset.hh"


za::VADDataset::VADDataset(std::string &manifest_path, uint32_t batch_size){
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

void za::VADDataset::readAudioFile(std::string &audioPath, std::unique_ptr<af::array> af_data){
    std::vector<float>data = fl::pkg::speech::loadSound<float>(audioPath);
    af_data->write(data.data(), data.size() * sizeof(float));
}


std::vector<af::array> za::VADDataset::get(const int64_t idx){
    return std::vector<af::array>();
}

int64_t za::VADDataset::size(){
    return 0;
}

std::shared_ptr<za::VADDataset> za::getDataset(){
    return nullptr;
}
