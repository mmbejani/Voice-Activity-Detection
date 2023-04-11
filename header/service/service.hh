#pragma once

#include <grpc++/grpc++.h>

#include "service.pb.h"
#include "service.grpc.pb.h"
#include "inference.hh"

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <deque>
#include <map>
#include <thread>
#include <mutex>
#include <chrono>
#include <random>

using namespace grpc;
using namespace std;

DECLARE_string(path_to_model);
DECLARE_double(determination_threshold);
DECLARE_uint32(batch_size);
DECLARE_uint32(wait_time);

namespace za {

    mutex result_mutex;
    mutex validation_mutex;

    class ServiceImpl : public SAD::Service {
        public:
            ServiceImpl(unsigned short max_batch_size,
                        unsigned short wait_time);

            Status validate(ServerContext* context, 
                            const AudioRequest* request, 
                            AnomalyReply* response) override;
            
        private:
            [[noreturn]] void inferenceLoop();
            void inferenceOnBatch();
            inline long generateId() const;

            const unsigned short max_batch_size;
            const unsigned short wait_time;
            unsigned short number_current_request_in_queue;
            shared_ptr<Inference> model;
            shared_ptr<deque<pair<const long, const AudioRequest*>>> request_queue;
            condition_variable preparing_result;
            map<const long, bool> results;
            thread* inferenceThread;
    };
}