cmake_minimum_required(VERSION 3.14)
project(MiniTorchGPU LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(include)

add_library(minitorch_gpu STATIC
    src/core/tensor.cpp
    src/core/function.cpp
    src/core/tensor.cu
    src/core/function.cu
    src/nn/linear.cpp
    src/nn/activations.cpp
    src/nn/sequential.cpp
    src/nn/conv.cpp
    src/nn/pool.cpp
    src/nn/flatten.cpp
    src/optim/sgd.cpp
)

add_executable(main ${CMAKE_SOURCE_DIR}/examples/main.cpp)

# ✨ FIX: Add cublas and curand to the list of linked libraries
target_link_libraries(main PRIVATE minitorch_gpu CUDA::cudart CUDA::cublas CUDA::curand)

target_include_directories(minitorch_gpu PUBLIC include)