#pragma once

#include <flashlight/fl/dataset/BatchDataset.h>
#include <flashlight/fl/dataset/Dataset.h>
#include <flashlight/pkg/speech/data/Sound.h>
#include <flashlight/fl/flashlight.h>
#include <flashlight/pkg/speech/common/Flags.h>


#include <gflags/gflags.h>
#include <glog/logging.h>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <json/json.h>

#include <string>
#include <fstream>

DECLARE_uint32(batch_size);
DECLARE_string(manifest_path);


namespace za{
    class VADDataset: fl::Dataset {
        public:
            VADDataset(std::string &manifest_path=FLAGS_manifest_path, uint32_t batch_size=FLAGS_batch_size);
            std::vector<af::array> get(const int64_t idx) const override;
            int64_t size() const override;

            std::vector<af::array> getBatch();

        private:
            std::vector<std::string> audio_paths;
            std::vector<int16_t> audio_labels;
            LoadFunction audioLoader;
            BatchFunction audioCollator, labelCollator;
    };

    std::unique_ptr<za::VADDataset> getDataset();

    af::array cat_array(std::vector<af::array>& arrays);
}