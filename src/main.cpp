#include "train.hh"

using namespace za;

int main(int argc, char **argv){
    google::InitGoogleLogging(*argv);
    google::ParseCommandLineFlags(&argc, &argv, false);
    fl::init();
    return 0;
}