#include "train.hh"

#include <flashlight/pkg/runtime/common/SequentialBuilder.h>

using namespace za;
using namespace fl::pkg::runtime;

int main(int argc, char **argv){
    google::InitGoogleLogging(*argv);
    google::ParseCommandLineFlags(&argc, &argv, false);
    google::SetStderrLogging(0);
    fl::init();
    // Creating dataset
    auto dataset = getDataset();
    auto loader_var = fl::BatchDataset(dataset, (int)FLAGS_batch_size,
                                   fl::BatchDatasetPolicy::INCLUDE_LAST,
                                   {dataset->audioCollator, dataset->lengthCollator});
    auto loader = make_shared<fl::BatchDataset>(loader_var);

    // Creating model
    auto model = buildSequentialModule("/home/mahdi/Project/Voice-Activity-Detection/src/model.conf", 39, 1);
    auto vad = make_shared<Vad>(model);

    //Creating optimizer
    auto optimizer = static_pointer_cast<fl::FirstOrderOptimizer> 
                    (make_shared<fl::SGDOptimizer>(vad->model->params(), 1e-5));

    //Creating loss function
    auto loss = make_shared<fl::MSE>();

    //Creating trainer
    auto trainer = Train(vad, loader, loss, optimizer, 100);
    trainer.start_train_process();
    return 0;
}