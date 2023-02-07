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
    }

    reader->close();

}


std::vector<af::array> za::VADDataset::get(const int64_t idx){

}

int64_t za::VADDataset::size(){
    return 0;
}

std::shared_ptr<za::VADDataset> za::get_dataset(){
    return nullptr;
}
