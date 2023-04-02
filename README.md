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


| OPTION | Description | Result File |
| :---:  | :---------: | :---------: |
| `TRAIN`| Compile the files for training phase | `run_train`
| `TEST` | Compile the corresponding files for test some utterance locally | `run_test`
| `INFER`| Compile the files for making the model as library (compile all dependency in shared manner). Then the result can be installed (run `make install`). | `sad.so`
| `SERV` | Compile the corresponding files for GRPC server | `run_server`

One can choose one the above options.