cmake_minimum_required(VERSION 3.10)

project(VAD LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(header/utils)
include_directories(header)

if(MKL_PATH)
    message(STATUS "-- Your path has been set as MKL home directory")
else()
    set(MKL_PATH /opt/intel/mkl)
    message(WARNING "The default path is considered as MKL home directory ${MKL_PATH}")
endif()

list(APPEND CMAKE_PREFIX_PATH /opt/intel/mkl)

if(CUDA)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    set(CUDNN_INCLUDE_DIR /usr/local/cuda/include)
    set(CUDNN_LIBRARY /usr/local/cuda/lib64/libcudnn.so)
endif()


find_package(flashlight CONFIG REQUIRED)
find_package(gflags REQUIRED)
find_package(glog REQUIRED)
find_package(jsoncpp REQUIRED)

if(BUILD_TRAIN)
    file(GLOB SRC_FILES src/utils/*.cc)
    file(GLOB HEADER_FILES header/utils/*.hh)
    set(RUN_FILE run_train)
    add_executable(${RUN_FILE} ${SRC_FILES} ${HEADER_FILES})
    target_link_libraries(${RUN_FILE} PRIVATE flashlight::flashlight
                                              flashlight::fl_pkg_runtime
                                              flashlight::fl_lib_audio
                                              flashlight::fl_pkg_speech
                                              glog::glog
                                              jsoncpp_lib)

elseif(BUILD_INFER)
    file(GLOB SRC_FILES src/*.cc)
    file(GLOB HEADER_FILES header/*.hh)
    set(RUN_FILE run_infer)    
    add_executable(${RUN_FILE} ${SRC_FILES} ${HEADER_FILES})
    target_link_libraries(${RUN_FILE} PRIVATE flashlight::flashlight)
elseif(BUILD_TEST)
    file(GLOB SRC_FILES src/utils/* src/*)
    file(GLOB HEADER_FILES header/utils/* header/*)
    set(RUN_FILE run_test)    
    add_executable(${RUN_FILE} ${SRC_FILES} ${HEADER_FILES})
    target_link_libraries(${RUN_FILE} PRIVATE flashlight::flashlight
                                              flashlight::fl_pkg_runtime
                                              flashlight::fl_pkg_speech
                                              flashlight::fl_lib_audio
                                              glog::glog
                                              jsoncpp_lib)
else()
    message(FATAL_ERROR "Select one of the options 1. BUILD_INFER 2.BUILD_TRAIN 3.BUILD_TEST")
endif()

message(STATUS ${RUN_FILE})