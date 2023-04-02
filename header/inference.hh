#pragma once

#include <flashlight/fl/flashlight.h>
#include <flashlight/pkg/runtime/common/SequentialBuilder.h>

#include <fstream>

#include "vad.hh"

using namespace std;
using namespace fl::pkg::runtime;

namespace za{

    class Inference
    {
    private:
        std::shared_ptr<Vad> vad;

    public:
        Inference(const string&);

        //For inferencing from a single utterance file
        fl::Variable infer(const string&);

        //For inferencing from single stream of utterance bytes
        fl::Variable infer(const ifstream&);

        //For inferencing from single tensor respect to an utterance
        fl::Variable infer(const fl::Variable&);
        
        //Inference on some utterances as file paths (batch manner)
        fl::Variable inferBatch(const vector<string>&);

        //Inference on some utterances as byte stream (batch manner)
        fl::Variable inferBatch(const vector<ifstream>&);

        /**Inference on some utterances a tensor (batch manner) where
          the shape of the tensor is BxL (B and L stand for batch-size and
          max length of utterances).

          @param input_signals: It is a flashlight Variable which
                                carries out a tensor of utterace signal
          @param input_lengths: It is a flashlight Variable which
                                carries out the lengths of signals of the 
                                first param.

          @return this method return a vector with length B which is $\{0,1\}^B$ 
                  where B is batch-size and one stands for normal human utterance
                  and zero represents abnormal utterance or voice (The defination is
                  different from VAD. We name it SAD or Speech Anomaly Detection).
                  Zarebin Assistant members know the meaning of SAD. To become familiar
                  with the concept be in touch with ZA squad ;).
          
        */
        fl::Variable inferBatch(const fl::Variable& input_signals, 
                                const fl::Variable& input_lengths);
    };   
}