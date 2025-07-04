# MiniTorch CPU

这是一个用 C++ 实现的、纯 CPU 的迷你深度学习框架，用于教学和理解深度学习框架的内部工作原理。

## 功能
- 动态计算图
- 自动微分 (反向传播)
- 模块化的神经网络层 (Linear, ReLU, Sequential)
- SGD 优化器

## 如何构建和运行

确保你已经安装了 `cmake` 和一个 C++ 编译器 (如 `g++` 或 `clang++`)。

```bash
# 1. 创建一个 build 目录
mkdir build
cd build

# 2. 运行 CMake 来配置项目
cmake ..

# 3. 编译项目
make

# 4. 运行示例程序
./train_example