#include "core/tensor.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <numeric>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

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
        } else {
            // cudaFree 在内部会检查指针是否为空，以防万一检查一下
            CUDA_CHECK(cudaFree(_data));
        }
    }
}

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
        throw std::runtime_error("item() 只能用于标量Tensor");
    }
    if (_device == Device::CPU) {
        return (*static_cast<std::vector<float>*>(_data))[0];
    } else {
        float host_val;
        CUDA_CHECK(cudaMemcpy(&host_val, _data, sizeof(float), cudaMemcpyDeviceToHost));
        return host_val;
    }
}

// 将张量移动到另一个设备
std::shared_ptr<Tensor> Tensor::to(Device device) {
    // 如果已经在目标设备上，直接返回自身的共享指针
    if (this->_device == device) {
        return shared_from_this();
    }
    
    // 创建一个位于目标设备的新张量，它会自动分配好目标设备的内存
    auto new_tensor = std::make_shared<Tensor>(_shape, _requires_grad, device);
    size_t data_size = this->size() * sizeof(float);
    if (data_size == 0) {
        return new_tensor;
    }

    // 根据数据转移的方向，选择正确的指针和拷贝方式
    if (device == Device::CUDA) { // 方向: CPU -> CUDA
        // 源(this)在CPU上, _data 是 std::vector<float>*
        // 目标(new_tensor)在GPU上, mutable_data_ptr() 返回 float* (GPU地址)

        // 1. 从源CPU张量中获取 std::vector<float> 对象
        auto& src_vector = *static_cast<std::vector<float>*>(this->mutable_data_ptr());
        
        // 2. 使用 .data() 方法获取指向vector底层连续数据的裸指针 (float*)
        const float* src_ptr = src_vector.data();

        // 3. 执行从主机到设备的内存拷贝
        CUDA_CHECK(cudaMemcpy(
            new_tensor->mutable_data_ptr(), // 目标: GPU地址 (void*)
            src_ptr,                        // 源: CPU上原始数据的地址 (const float*)
            data_size, 
            cudaMemcpyHostToDevice
        ));

    } else { // 方向: CUDA -> CPU
        // 源(this)在GPU上, _data 是 float* (GPU地址)
        // 目标(new_tensor)在CPU上, _data 是 std::vector<float>*
        
        // 1. 从目标CPU张量中获取 std::vector<float> 对象
        auto& dst_vector = *static_cast<std::vector<float>*>(new_tensor->mutable_data_ptr());
        
        // 2. 使用 .data() 方法获取指向vector底层连续数据的裸指针 (float*)
        float* dst_ptr = dst_vector.data();
        
        // 3. 执行从设备到主机的内存拷贝
        CUDA_CHECK(cudaMemcpy(
            dst_ptr,                        // 目标: CPU上原始数据的地址 (float*)
            this->data_ptr(),               // 源: GPU地址 (const void*)
            data_size, 
            cudaMemcpyDeviceToHost
        ));
    }
    return new_tensor;
}

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
        curandDestroyGenerator(gen);
    }
    return t;
}

// ones 的设备分发实现
std::shared_ptr<Tensor> Tensor::ones(const std::vector<size_t>& shape, bool requires_grad, Device device) {
    auto t = std::make_shared<Tensor>(shape, requires_grad, device);
    if (t->size() == 0) {
        return t;
    }

    if (device == Device::CPU) {
        // 对于 CPU, _data 是 std::vector<float>*, 转换是安全的
        auto& data_vec = *static_cast<std::vector<float>*>(t->mutable_data_ptr());
        std::fill(data_vec.begin(), data_vec.end(), 1.0f);
    } else { // device == Device::CUDA
        // 对于 CUDA, _data 是 float* (GPU地址), 调用 CUDA 核函数填充
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

// 2D 矩阵转置的 CUDA 核函数
__global__ void transpose_kernel(float* out, const float* in, size_t H, size_t W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < H && x < W) {
        out[x * H + y] = in[y * W + x];
    }
}

std::shared_ptr<Tensor> Tensor::transpose() const {
    if (_shape.size() != 2) throw std::runtime_error("转置操作仅支持二维Tensor");
    
    std::vector<size_t> new_shape = {_shape[1], _shape[0]};
    auto new_tensor = std::make_shared<Tensor>(new_shape, _requires_grad, _device);

    if (this->size() == 0) return new_tensor;

    if (_device == Device::CPU) {
        const auto& in_data = *static_cast<const std::vector<float>*>(_data);
        auto& out_data = *static_cast<std::vector<float>*>(new_tensor->mutable_data_ptr());
        size_t H = _shape[0];
        size_t W = _shape[1];
        for(size_t i = 0; i < W; ++i) { // new height
            for(size_t j = 0; j < H; ++j) { // new width
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

// 在Tensor中的第一个维度切出从start到end的部分
std::shared_ptr<Tensor> Tensor::slice(size_t start, size_t end) const {
    if(_shape.size() < 1 || start >= end || end > _shape[0]) {
        throw std::runtime_error("切片参数无效");
    }
    
    size_t feature_size = 1;
    for(size_t i = 1; i < _shape.size(); ++i) {
        feature_size *= _shape[i];
    }
    
    std::vector<size_t> new_shape = _shape;
    new_shape[0] = end - start;
    
    // 创建新张量
    auto new_tensor = std::make_shared<Tensor>(new_shape, _requires_grad, _device);
    
    if (this->size() == 0) return new_tensor;
    
    if (_device == Device::CPU) {
        // CPU 内存处理
        const auto& src_data = *static_cast<const std::vector<float>*>(_data);
        auto& dst_data = *static_cast<std::vector<float>*>(new_tensor->mutable_data_ptr());
        
        auto start_sl = src_data.begin() + start * feature_size;
        auto end_sl = src_data.begin() + end * feature_size;
        dst_data.assign(start_sl, end_sl);
    } else {
        // GPU 内存处理
        size_t copy_size = (end - start) * feature_size;
        const float* src_start = static_cast<const float*>(_data) + start * feature_size;
        
        CUDA_CHECK(cudaMemcpy(
            new_tensor->mutable_data_ptr(),
            src_start,
            copy_size * sizeof(float),
            cudaMemcpyDeviceToDevice
        ));
    }
    
    return new_tensor;
}

