#include "service.hh"

#include <semaphore>
namespace za {

    DEFINE_string(path_to_model, "/path/to/model", "Path to Model file (bin)");
    DEFINE_double(determination_threshold, 0.5, "Whether or not accept the output of network");

    ServiceImpl::ServiceImpl(unsigned short max_batch_size,
                             unsigned short wait_time) :
                                max_batch_size(max_batch_size),
                                wait_time(wait_time),
                                number_current_request_in_queue(0),
                                request_queue(make_shared<deque<pair<const long, const AudioRequest*>>>()),
                                model(make_shared<Inference>(FLAGS_path_to_model)) 
    {
        this->inferenceThread = new thread(&ServiceImpl::inferenceLoop, this);
    }

    Status ServiceImpl::validate(ServerContext* context, 
                                 const AudioRequest* request, 
                                 AnomalyReply* response){
        unique_lock<mutex>(validation_mutex);
        const long id = this->generateId();
        this->request_queue->push_back(pair<const long, const AudioRequest*>(id, request));
        number_current_request_in_queue++;
        validation_mutex.unlock();


        unique_lock<mutex> lock(result_mutex);
        this->preparing_result.wait(lock);
        response->set_isvalid(this->results[id]);
    }

    void ServiceImpl::inferenceLoop(){
        while (true) {
            LOG(INFO) << "Thread is wating for filling queue";
            this_thread::sleep_for(chrono::milliseconds(this->wait_time));
            unique_lock<mutex>(validation_mutex);
            this->inferenceOnBatch();
            validation_mutex.unlock();
        }
    }

    void ServiceImpl::inferenceOnBatch(){
        for (size_t i = 0;i <= this->request_queue->size() / max_batch_size;++i) {
            vector<const long> ids;
            vector<istream> streams;
            for (size_t j = i * max_batch_size; j < min((i + 1) * this->max_batch_size, this->request_queue->size()); j++)
            {
                const long id = request_queue->front().first;
                istringstream audio(request_queue->front().second->audiobytes());

                ids.push_back(id);
                streams.push_back(audio);

                request_queue->pop_front();
            }

            auto batch_result = model->inferBatch(streams).array().host<float>();

            for (size_t j = 0; j < max_batch_size; j++)
            {
                if (batch_result[i] > FLAGS_determination_threshold)
                    results[ids[i]] = true;
                else
                    results[ids[i]] = false;
            } 
        }
        number_current_request_in_queue = 0;

        this->preparing_result.notify_all();
        return;
    }

    inline long ServiceImpl::generateId() const {
        long time = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
        return time + rand();
    }
}