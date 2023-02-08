#include <flashlight/fl/dataset/BatchDataset.h>
#include <flashlight/fl/dataset/Dataset.h>
#include <flashlight/pkg/speech/data/Sound.h>


#include <gflags/gflags.h>
#include <glog/logging.h>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <string>
#include <fstream>

DEFINE_uint32(batch_size, 32, "Size of batch size");
namespace za{
    class VADDataset: fl::Dataset {
        public:
            VADDataset(std::string &manifest_path, uint32_t batch_size=FLAGS_batch_size);

        std::vector<af::array> get(const int64_t idx);
        int64_t size();

        private:
            std::vector<std::string> audio_paths;
            std::vector<af::array> audio_labels;
            LoadFunction loader;
            BatchFunction collator;

            void readAudioFile(std::string &audioPath, std::unique_ptr<af::array> af_data);

    };

    std::shared_ptr<za::VADDataset> getDataset();
}