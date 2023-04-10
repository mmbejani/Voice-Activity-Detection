#pragma once

#include "service.pb.h"
#include "service.grpc.pb.h"

#include <vector>
#include <thread>
#include <mutex>

using namespace grpc;
using namespace std;

namespace za {

    mutex result;
    mutex add;

    class ServiceImpl : public SAD::Service {
        public:
            ServiceImpl();

            Status validate(ServerContext* context, 
                            const AudioRequest* request, 
                            AnomalyReply* response) override;
            
        private:
            void inferenceLoop();

            vector<AudioRequest*> request_queue;
    };
}