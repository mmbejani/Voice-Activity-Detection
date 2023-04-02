# Speech Anomaly Detection (Normal Human Voice Detection)
This library has been implemented for voice activity detection (VAD) by human voice detection aspect. Therefore, the model is classified voices into two class, no-human (noise) and human voice. The library is based on C/C++17 (Clang compiler is preferred).


# Pre-requirements
This library is highly depends on Flashlight library (url!!!!!!). Then take to following step to install it:
 1) Install libraries `boost`, `glog`, `gflag`, and `json`, 
 2) Install Flashlight from following link (version 0.3):

 3) Install `GRPC` to build it as service.

# How to install
Run following commands to build executable file:
```
git clone https://github.com/mmbejani/Voice-Activity-Detection

cd Voice-Activity-Detection

mkdir build && cd build

cmake .. [OPTION]

make -j 2

./run_[corrosponding built file]
```

Instead of `OPTION` use the following option 







choose between `-DTRAIN=ON` to build project for training and `-DINFER=ON` to build project for inference on local data,. Beside, corresponding to building option in the last step a run file is maked. `