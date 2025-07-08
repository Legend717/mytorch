cmake_minimum_required(VERSION 3.14)
project(MiniTorchCPU)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===== 手动设置 Python3 路径（来自 Conda） =====
set(Python3_EXECUTABLE "/home/rogers/miniconda3/envs/pytorch/bin/python")
set(Python3_INCLUDE_DIR "/home/rogers/miniconda3/envs/pytorch/include/python3.12")
set(Python3_LIBRARY "/home/rogers/miniconda3/envs/pytorch/lib/libpython3.12.so")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

include_directories(include)
include_directories(${Python3_INCLUDE_DIRS})

add_library(minitorch_cpu STATIC
    src/core/tensor.cpp
    src/core/function.cpp
    src/nn/linear.cpp
    src/nn/activations.cpp
    src/nn/sequential.cpp
    src/nn/conv.cpp
    src/nn/pool.cpp
    src/nn/flatten.cpp
    src/optim/sgd.cpp
)

add_executable(main examples/luotest.cpp)

target_link_libraries(main
    minitorch_cpu
    pybind11::embed
    ${Python3_LIBRARIES}
)

target_include_directories(minitorch_cpu PUBLIC include)
