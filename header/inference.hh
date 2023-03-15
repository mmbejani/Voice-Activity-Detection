#pragma once

#include <flashlight/fl/flashlight.h>
#include <flashlight/pkg/runtime/common/SequentialBuilder.h>

using namespace std;
using namespace fl::pkg::runtime;

namespace za{

    class Inference
    {
    private:
        std::shared_ptr<fl::Sequential> model;

    public:
        Inference(const string&);

        fl::Variable infer(const string&);
        fl::Variable infer(const fl::Variable&);
        
        fl::Variable inferBatch(const vector<string>&);
        fl::Variable inferBatch(const fl::Variable&);
    };   
}