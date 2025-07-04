#include "core/tensor.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <numeric>

// --- CUDA 错误检查宏 ---
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)


// --- 私有辅助函数的 CUDA 部分 ---
// 这个函数由构造函数调用，负责根据设备分配内存
void Tensor::allocate_data() {
    size_t total_size = this->size();
    if (total_size == 0) {
        _data = nullptr;
        return;
    }

    if (_device == Device::CPU) {
        _data = new std::vector<float>(total_size, 0.0f);
    } else { // _device == Device::CUDA
        CUDA_CHECK(cudaMalloc(&_data, total_size * sizeof(float)));
        // 确保新分配的GPU内存被清零，这对于zeros()等操作很重要
        CUDA_CHECK(cudaMemset(_data, 0, total_size * sizeof(float)));
    }
}

// 析构函数的实现，因为它需要调用 cudaFree
Tensor::~Tensor() {
    if (_data) {
        if (_device == Device::CPU) {
            delete static_cast<std::vector<float>*>(_data);
        } else { // _device == Device::CUDA
            // cudaFree 在内部会检查指针是否为空，但我们为了保持良好习惯也进行检查
            CUDA_CHECK(cudaFree(_data));
        }
    }
}


// --- 数据访问与设备转移 ---

// 从任何设备获取数据的CPU拷贝
std::vector<float> Tensor::data_cpu() const {
    size_t total_size = this->size();
    if (total_size == 0) return {};

    std::vector<float> host_data(total_size);
    if (_device == Device::CPU) {
        const auto& vec = *static_cast<const std::vector<float>*>(_data);
        host_data = vec;
    } else { // _device == Device::CUDA
        CUDA_CHECK(cudaMemcpy(host_data.data(), _data, total_size * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return host_data;
}

// 获取单个元素的值，针对GPU做了优化
float Tensor::item() const {
    if (this->size() != 1) {
        throw std::runtime_error("item() can only be called on tensors with a single element.");
    }
    if (_device == Device::CPU) {
        return (*static_cast<std::vector<float>*>(_data))[0];
    } else { // _device == Device::CUDA
        float host_val;
        CUDA_CHECK(cudaMemcpy(&host_val, _data, sizeof(float), cudaMemcpyDeviceToHost));
        return host_val;
    }
}

// 将张量移动到另一个设备
std::shared_ptr<Tensor> Tensor::to(Device device) {
    if (this->_device == device) {
        return shared_from_this();
    }
    
    // 创建一个位于目标设备的新张量
    auto new_tensor = std::make_shared<Tensor>(_shape, _requires_grad, device);
    size_t data_size = this->size() * sizeof(float);
    if (data_size == 0) return new_tensor;

    // 根据方向进行拷贝
    if (device == Device::CUDA) { // CPU -> CUDA
        CUDA_CHECK(cudaMemcpy(new_tensor->mutable_data_ptr(), this->data_ptr(), data_size, cudaMemcpyHostToDevice));
    } else { // CUDA -> CPU
        CUDA_CHECK(cudaMemcpy(new_tensor->mutable_data_ptr(), this->data_ptr(), data_size, cudaMemcpyDeviceToHost));
    }
    return new_tensor;
}


// --- 工厂方法 (randn, ones, zeros) ---

// 用于填充GPU张量的辅助核函数
__global__ void fill_kernel(float* data, float value, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

// 主机端函数，用于启动 fill_kernel
void fill_value_gpu(float* data, float value, size_t n) {
    if (n == 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fill_kernel<<<blocks, threads>>>(data, value, n);
    CUDA_CHECK(cudaPeekAtLastError());
}


// randn 的设备分发实现
std::shared_ptr<Tensor> Tensor::randn(const std::vector<size_t>& shape, bool requires_grad, Device device) {
    auto t = std::make_shared<Tensor>(shape, requires_grad, device);
    size_t n = t->size();
    if (n == 0) return t;

    if (device == Device::CPU) {
        std::vector<float>& data_vec = *static_cast<std::vector<float>*>(t->mutable_data_ptr());
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 1);
        for(auto& val : data_vec) val = d(gen);
    } else { // device == Device::CUDA
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        // 使用时间和时钟周期组合作为种子，增加随机性
        curandSetPseudoRandomGeneratorSeed(gen, time(NULL) + clock());
        curandGenerateNormal(gen, static_cast<float*>(t->mutable_data_ptr()), n, 0.0f, 1.0f);
        // FIX: Corrected function name from curandDestroy to curandDestroyGenerator
        curandDestroyGenerator(gen);
    }
    return t;
}

// ones 的设备分发实现
std::shared_ptr<Tensor> Tensor::ones(const std::vector<size_t>& shape, bool requires_grad, Device device) {
    auto t = std::make_shared<Tensor>(shape, requires_grad, device);
    if (t->size() == 0) return t;

    if (device == Device::CPU) {
        std::vector<float>& data_vec = *static_cast<std::vector<float>*>(t->mutable_data_ptr());
        std::fill(data_vec.begin(), data_vec.end(), 1.0f);
    } else { // device == Device::CUDA
        fill_value_gpu(static_cast<float*>(t->mutable_data_ptr()), 1.0f, t->size());
    }
    return t;
}

// zeros 的设备分发实现
std::shared_ptr<Tensor> Tensor::zeros(const std::vector<size_t>& shape, bool requires_grad, Device device) {
    // 构造函数已经为我们分配了内存，并将其清零
    auto t = std::make_shared<Tensor>(shape, requires_grad, device);
    return t;
}


// --- Transpose 的设备分发实现 ---

// 2D 矩阵转置的 CUDA 核函数
__global__ void transpose_kernel(float* out, const float* in, size_t H, size_t W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < H && x < W) {
        out[x * H + y] = in[y * W + x];
    }
}

std::shared_ptr<Tensor> Tensor::transpose() const {
    if (_shape.size() != 2) throw std::runtime_error("Transpose is only supported for 2D tensors.");
    
    std::vector<size_t> new_shape = {_shape[1], _shape[0]};
    auto new_tensor = std::make_shared<Tensor>(new_shape, _requires_grad, _device);

    if (this->size() == 0) return new_tensor;

    if (_device == Device::CPU) {
        const auto& in_data = *static_cast<const std::vector<float>*>(_data);
        auto& out_data = *static_cast<std::vector<float>*>(new_tensor->mutable_data_ptr());
        size_t H = _shape[0];
        size_t W = _shape[1];
        for(size_t i=0; i < W; ++i) { // new height
            for(size_t j=0; j < H; ++j) { // new width
                out_data[i * H + j] = in_data[j * W + i];
            }
        }
    } else { // _device == Device::CUDA
        dim3 threads(16, 16); // 2D 线程块
        dim3 blocks(
            (new_shape[1] + threads.x - 1) / threads.x,
            (new_shape[0] + threads.y - 1) / threads.y
        );
        transpose_kernel<<<blocks, threads>>>(
            static_cast<float*>(new_tensor->mutable_data_ptr()),
            static_cast<const float*>(_data),
            _shape[0], _shape[1]
        );
        CUDA_CHECK(cudaPeekAtLastError());
    }
    return new_tensor;
}