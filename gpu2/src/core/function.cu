#include "core/tensor.h"
#include "core/function.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cfloat>
#include <stdexcept>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUBLAS_CHECK(call) \
do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "CUBLAS error in file '%s' in line %i: %d\n", \
                __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// SGD
__global__ void sgd_update_kernel(float* params, const float* grads, float lr, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        params[idx] -= lr * grads[idx];
    }
}

void sgd_update_cuda(float* params, const float* grads, float lr, size_t n) {
    if (n == 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(params, grads, lr, n);
    CUDA_CHECK(cudaPeekAtLastError());
}

// kernel
__global__ void add_kernel(float* out, const float* a, const float* b, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void sub_kernel(float* out, const float* a, const float* b, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void add_broadcast_4d_kernel(float* out, const float* a, const float* b,
                                     size_t a_n, size_t a_c, size_t a_h, size_t a_w,
                                     size_t b_c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < a_n * a_c * a_h * a_w) {
        int c_idx = (idx / (a_w * a_h)) % a_c;
        out[idx] = a[idx] + b[c_idx];
    }
}

__global__ void add_broadcast_2d_kernel(float* out, const float* a, const float* b, size_t M, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int col = idx % N;
        out[idx] = a[idx] + b[col];
    }
}


__global__ void mul_kernel(float* out, const float* a, const float* b, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void mul_scalar_kernel(float* out, const float* a, float b, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b;
    }
}


__global__ void relu_kernel(float* out, const float* a, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] > 0 ? a[idx] : 0.0f;
    }
}

std::shared_ptr<Tensor> add_forward_cuda(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    if (a->size() < b->size()) {
        return add_forward_cuda(b, a);
    }
    auto output = Tensor::zeros(a->shape(), false, Device::CUDA);
    size_t n = a->size();
    if (n == 0) return output;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    const auto& a_shape = a->shape();
    const auto& b_shape = b->shape();

    if (a_shape == b_shape) {
        add_kernel<<<blocks, threads>>>(
            static_cast<float*>(output->mutable_data_ptr()),
            static_cast<const float*>(a->data_ptr()),
            static_cast<const float*>(b->data_ptr()), n);
    } else if (a_shape.size() == 4 && b_shape.size() == 4 &&
               b_shape[0] == 1 && b_shape[2] == 1 && b_shape[3] == 1 &&
               a_shape[1] == b_shape[1]) {
        add_broadcast_4d_kernel<<<blocks, threads>>>(
            static_cast<float*>(output->mutable_data_ptr()),
            static_cast<const float*>(a->data_ptr()),
            static_cast<const float*>(b->data_ptr()),
            a_shape[0], a_shape[1], a_shape[2], a_shape[3],
            b_shape[1]);
    } else if (a_shape.size() == 2 && b_shape.size() == 2 &&
               b_shape[0] == 1 && a_shape[1] == b_shape[1]) {
        add_broadcast_2d_kernel<<<blocks, threads>>>(
            static_cast<float*>(output->mutable_data_ptr()),
            static_cast<const float*>(a->data_ptr()),
            static_cast<const float*>(b->data_ptr()),
            a_shape[0], a_shape[1]);
    } else {
        throw std::runtime_error("GPU代码：还没支持这种shape的broadcast");
    }
    CUDA_CHECK(cudaPeekAtLastError());
    return output;
}

std::shared_ptr<Tensor> sub_forward_cuda(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    if (a->shape() != b->shape()) {
        throw std::runtime_error("GPU减法：shape不匹配");
    }
    auto output = Tensor::zeros(a->shape(), false, Device::CUDA);
    size_t n = a->size();
    if (n == 0) return output;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sub_kernel<<<blocks, threads>>>(
        static_cast<float*>(output->mutable_data_ptr()),
        static_cast<const float*>(a->data_ptr()),
        static_cast<const float*>(b->data_ptr()), n);
    CUDA_CHECK(cudaPeekAtLastError());
    return output;
}

std::shared_ptr<Tensor> mul_forward_cuda(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    const std::shared_ptr<Tensor> *tensor_ptr, *scalar_ptr;
    if (a->size() == 1 && b->size() > 1) {
        tensor_ptr = &b;
        scalar_ptr = &a;
    } else if (b->size() == 1 && a->size() > 1) {
        tensor_ptr = &a;
        scalar_ptr = &b;
    } else if (a->shape() == b->shape()) {
        auto output = Tensor::zeros(a->shape(), false, Device::CUDA);
        size_t n = a->size();
        if (n == 0) return output;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        mul_kernel<<<blocks, threads>>>(static_cast<float*>(output->mutable_data_ptr()), static_cast<const float*>(a->data_ptr()), static_cast<const float*>(b->data_ptr()), n);
        CUDA_CHECK(cudaPeekAtLastError());
        return output;
    } else {
        throw std::runtime_error("GPU乘法：shape不匹配");
    }

    // 标量乘法
    auto output = Tensor::zeros((*tensor_ptr)->shape(), false, Device::CUDA);
    float scalar_val = (*scalar_ptr)->item(); // .item() copies from GPU to CPU
    size_t n = (*tensor_ptr)->size();
    if (n == 0) return output;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mul_scalar_kernel<<<blocks, threads>>>(static_cast<float*>(output->mutable_data_ptr()), static_cast<const float*>((*tensor_ptr)->data_ptr()), scalar_val, n);
    CUDA_CHECK(cudaPeekAtLastError());
    return output;
}

std::shared_ptr<Tensor> matmul_forward_cuda(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    if (a->shape().size() != 2 || b->shape().size() != 2 || a->shape()[1] != b->shape()[0]) { throw std::runtime_error("MatMul shape mismatch."); }
    size_t M = a->shape()[0];
    size_t K = a->shape()[1];
    size_t N = b->shape()[1];
    auto output = Tensor::zeros({M, N}, false, Device::CUDA);
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         N, M, K, // cuBLAS参数 m, n, k
                         &alpha,
                         static_cast<const float*>(b->data_ptr()), N, // cuBLAS 的矩阵 A 是 b
                         static_cast<const float*>(a->data_ptr()), K, // cuBLAS 的矩阵 B 是 a
                         &beta,
                         static_cast<float*>(output->mutable_data_ptr()), N));    cublasDestroy(handle);
    return output;
}

std::shared_ptr<Tensor> relu_forward_cuda(const std::shared_ptr<Tensor>& a) {
    auto output = Tensor::zeros(a->shape(), false, Device::CUDA);
    size_t n = a->size();
    if (n == 0) return output;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>( static_cast<float*>(output->mutable_data_ptr()), static_cast<const float*>(a->data_ptr()), n);
    CUDA_CHECK(cudaPeekAtLastError());
    return output;
}

std::shared_ptr<Tensor> reshape_forward_cuda(const std::shared_ptr<Tensor>& input, const std::vector<size_t>& new_shape) {
    auto new_tensor = Tensor::zeros(new_shape, false, Device::CUDA);
    if (input->size() > 0) {
        CUDA_CHECK(cudaMemcpy( new_tensor->mutable_data_ptr(), input->data_ptr(), input->size() * sizeof(float), cudaMemcpyDeviceToDevice ));
    }
    return new_tensor;
}