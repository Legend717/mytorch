cmake_minimum_required(VERSION 3.14)
project(MiniTorchCPU)

# 设置 C++ 标准为 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加头文件目录，以便 #include "core/tensor.h" 能够工作
include_directories(include)

# 定义我们的框架库
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

# 定义我们的示例可执行文件
add_executable(main examples/main.cpp)

# 将框架库链接到示例程序
target_link_libraries(main minitorch_cpu)

# 方便在 IDE 中看到头文件
target_include_directories(minitorch_cpu PUBLIC include)