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

__global__ void reduce_sum_columns_kernel(float* out, const float* grad_output, size_t M, size_t N) {
    // 每个线程块负责计算输出向量中的一个元素（一列的和）
    int j = blockIdx.x; // 列索引

    if (j < N) {
        float sum = 0.0f;
        // 遍历该列的所有行并累加
        for (size_t i = 0; i < M; ++i) {
            sum += grad_output[i * N + j];
        }
        out[j] = sum;
    }
}

// 启动 reduce_sum_columns_kernel 的主机端函数
std::shared_ptr<Tensor> reduce_sum_columns_cuda(const std::shared_ptr<Tensor>& grad_output, const std::vector<size_t>& target_shape) {
    // 创建一个新的Tensor用于存储结果，确保它在GPU上
    auto grad_b = Tensor::zeros(target_shape, false, Device::CUDA);
    
    const auto& grad_shape = grad_output->shape();
    size_t M = grad_shape[0]; // batch_size
    size_t N = grad_shape[1]; // features

    if (N == 0) return grad_b;

    // 启动核函数：每个线程块计算一列的和，因此需要 N 个块
    reduce_sum_columns_kernel<<<N, 1>>>(
        static_cast<float*>(grad_b->mutable_data_ptr()),
        static_cast<const float*>(grad_output->data_ptr()),
        M,
        N
    );
    CUDA_CHECK(cudaPeekAtLastError());

    return grad_b;
}

// 新的包装函数：Add 运算在 CUDA 上的反向传播实现
// 这个函数将被 C++ 代码调用
std::shared_ptr<Tensor> add_backward_cuda(const std::shared_ptr<Tensor>& grad_output, const std::shared_ptr<Tensor>& b) {
    // 创建一个新的Tensor用于存储广播后的梯度结果，确保它和 b 在同一设备(GPU)上
    auto grad_b = Tensor::zeros(b->shape(), false, Device::CUDA);
    
    const auto& grad_shape = grad_output->shape();
    size_t M = grad_shape[0]; // batch_size
    size_t N = grad_shape[1]; // features

    if (N == 0) return grad_b;

    // 启动核函数：每个线程块计算一列的和，因此需要 N 个块
    reduce_sum_columns_kernel<<<N, 1>>>(
        static_cast<float*>(grad_b->mutable_data_ptr()),
        static_cast<const float*>(grad_output->data_ptr()),
        M,
        N
    );
    CUDA_CHECK(cudaPeekAtLastError());

    return grad_b;
}

// relu
__global__ void relu_backward_kernel(float* grad_input, const float* grad_output, const float* input, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = (input[idx] > 0) ? grad_output[idx] : 0.0f;
    }
}

std::shared_ptr<Tensor> relu_backward_cuda(const std::shared_ptr<Tensor>& grad_output, const std::shared_ptr<Tensor>& input) {
    auto grad_input = Tensor::zeros(input->shape(), false, Device::CUDA);
    size_t n = input->size();
    if (n == 0) return grad_input;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(
        static_cast<float*>(grad_input->mutable_data_ptr()),
        static_cast<const float*>(grad_output->data_ptr()),
        static_cast<const float*>(input->data_ptr()),
        n
    );
    CUDA_CHECK(cudaPeekAtLastError());
    return grad_input;
}


//conv
__global__ void im2col_kernel(const float* data_im, float* data_col,
                            int N, int C, int H, int W,
                            int K, int S, int P,
                            int H_out, int W_out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int col_size = C * K * K;
    int num_kernels = N * H_out * W_out;

    if (index < num_kernels * col_size) {
        int col_idx = index % col_size;
        int row_idx = index / col_size;

        // 分解列索引以找到核的位置
        int k_w = col_idx % K;
        int k_h = (col_idx / K) % K;
        int c_in = col_idx / (K * K);

        // 分解行索引以找到输出像素的位置
        int w_out = row_idx % W_out;
        int h_out = (row_idx / W_out) % H_out;
        int n = row_idx / (H_out * W_out);

        // 计算输入图像的相应坐标
        int h_in = h_out * S - P + k_h;
        int w_in = w_out * S - P + k_w;

        // 如果坐标在图像内，则将相应像素的值添加到输出列矩阵中
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            data_col[index] = data_im[(n * C + c_in) * H * W + h_in * W + w_in];
        } else {
            data_col[index] = 0.0f;
        }
    }
}

// 主机端函数，将输入图像转换为列矩阵，并使用 im2col_kernel 核函数进行处理
std::shared_ptr<Tensor> im2col_cuda(const std::shared_ptr<Tensor>& input,
                                  size_t K, size_t S, size_t P) {
    const auto& shape = input->shape();
    int N = shape[0];
    int C = shape[1];
    int H = shape[2];
    int W = shape[3];

    int H_out = (H + 2 * P - K) / S + 1;
    int W_out = (W + 2 * P - K) / S + 1;

    // 在GPU上创建输出列Tensor
    auto col_tensor = Tensor::zeros({(size_t)C * K * K, (size_t)N * H_out * W_out}, false, Device::CUDA);
    size_t n = col_tensor->size();
    if (n == 0) return col_tensor;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    im2col_kernel<<<blocks, threads>>>(
        static_cast<const float*>(input->data_ptr()),
        static_cast<float*>(col_tensor->mutable_data_ptr()),
        N, C, H, W, K, S, P, H_out, W_out
    );
    CUDA_CHECK(cudaPeekAtLastError());

    return col_tensor;
}


__global__ void col2im_kernel(const float* data_col, float* data_im,
                            int N, int C, int H, int W,
                            int K, int S, int P,
                            int H_out, int W_out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int col_size = C * K * K;
    int num_kernels = N * H_out * W_out;

    if (index < num_kernels * col_size) {
        // Calculate the position in the column matrix
        int col_idx = index % col_size;
        int row_idx = index / col_size;

        // Decompose the column index to find kernel position
        int k_w = col_idx % K;
        int k_h = (col_idx / K) % K;
        int c_in = col_idx / (K * K);

        // Decompose the row index to find output pixel position
        int w_out = row_idx % W_out;
        int h_out = (row_idx / W_out) % H_out;
        int n = row_idx / (H_out * W_out);

        // Calculate the corresponding input coordinates
        int h_in = h_out * S - P + k_h;
        int w_in = w_out * S - P + k_w;

        // Add to the image buffer if within bounds
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            atomicAdd(&data_im[(n * C + c_in) * H * W + h_in * W + w_in], data_col[index]);
        }
    }
}

std::vector<std::shared_ptr<Tensor>> conv2d_backward_cuda(
    const std::shared_ptr<Tensor>& grad_output,
    const std::shared_ptr<Tensor>& input,
    const std::shared_ptr<Tensor>& weight,
    size_t stride, size_t padding
) {
    const auto& in_shape = input->shape();
    const auto& w_shape = weight->shape();
    const auto& grad_out_shape = grad_output->shape();
    size_t N = in_shape[0], C_in = in_shape[1], H_in = in_shape[2], W_in = in_shape[3];
    size_t C_out = w_shape[0], kH = w_shape[2], kW = w_shape[3];
    size_t H_out = grad_out_shape[2], W_out = grad_out_shape[3];

    // --- 1. Calculate grad_weight ---
    // This is a convolution between input and grad_output.
    // grad_weight = matmul(grad_output_reshaped, input_col.T)

    // Reshape grad_output from (N, C_out, H_out, W_out) to (C_out, N * H_out * W_out)
    auto grad_output_reshaped = grad_output->reshape({C_out, N * H_out * W_out});

    // Get input as a column matrix
    auto input_col = im2col_cuda(input, kH, stride, padding); // Shape: (C_in*kH*kW, N*H_out*W_out)

    // Transpose the input column matrix
    auto input_col_t = input_col->transpose(); // Shape: (N*H_out*W_out, C_in*kH*kW)

    // Perform GEMM
    auto grad_weight_reshaped = grad_output_reshaped->matmul(input_col_t); // Shape: (C_out, C_in*kH*kW)

    // Reshape back to the original weight shape
    auto grad_weight = grad_weight_reshaped->reshape(w_shape);


    // --- 2. Calculate grad_input ---
    // This is a "full" convolution between grad_output and a rotated weight kernel.
    // It can be implemented as matmul(weight.T, grad_output_reshaped) followed by col2im.

    // Reshape weight from (C_out, C_in, kH, kW) to (C_out, C_in*kH*kW)
    auto weight_reshaped = weight->reshape({C_out, C_in * kH * kW});
    
    // Transpose the reshaped weight
    auto weight_reshaped_t = weight_reshaped->transpose(); // Shape: (C_in*kH*kW, C_out)

    // Perform GEMM to get the column matrix for the input gradient
    auto grad_input_col = weight_reshaped_t->matmul(grad_output_reshaped); // Shape: (C_in*kH*kW, N*H_out*W_out)

    // Create the final gradient tensor for the input, initialized to zeros
    auto grad_input = Tensor::zeros(in_shape, false, Device::CUDA);

    // Perform col2im to get the final grad_input
    size_t n = grad_input_col->size();
    if (n > 0) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        col2im_kernel<<<blocks, threads>>>(
            static_cast<const float*>(grad_input_col->data_ptr()),
            static_cast<float*>(grad_input->mutable_data_ptr()),
            N, C_in, H_in, W_in, kH, stride, padding, H_out, W_out
        );
        CUDA_CHECK(cudaPeekAtLastError());
    }

    return {grad_input, grad_weight};
}

// ----------------------------------------------------------------------------
// MaxPool2D Forward (CUDA Kernel and Host Function)
// ----------------------------------------------------------------------------
__global__ void maxpool2d_forward_kernel(const float* input_data, float* output_data, float* max_indices_data,
                                       int N, int C, int H, int W,
                                       int k, int s,
                                       int H_out, int W_out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N * C * H_out * W_out) {
        // Decompose index to find the output pixel location
        int w_out = index % W_out;
        int h_out = (index / W_out) % H_out;
        int c = (index / W_out / H_out) % C;
        int n = index / W_out / H_out / C;

        // Find the top-left corner of the pooling window in the input
        int h_start = h_out * s;
        int w_start = w_out * s;

        // Find the maximum value in the window
        float max_val = -FLT_MAX;
        int max_idx = -1;

        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                int h_in = h_start + i;
                int w_in = w_start + j;
                if (h_in < H && w_in < W) {
                    int input_idx = n * C * H * W + c * H * W + h_in * W + w_in;
                    float current_val = input_data[input_idx];
                    if (current_val > max_val) {
                        max_val = current_val;
                        max_idx = input_idx;
                    }
                }
            }
        }
        output_data[index] = max_val;
        max_indices_data[index] = static_cast<float>(max_idx); // Store index as float
    }
}

std::shared_ptr<Tensor> maxpool2d_forward_cuda(const std::shared_ptr<Tensor>& input, size_t kernel_size, size_t stride, std::shared_ptr<Tensor>& max_indices_tensor) {
    const auto& shape = input->shape();
    size_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];

    size_t H_out = (H - kernel_size) / stride + 1;
    size_t W_out = (W - kernel_size) / stride + 1;

    auto output = Tensor::zeros({N, C, H_out, W_out}, false, Device::CUDA);
    max_indices_tensor = Tensor::zeros({N, C, H_out, W_out}, false, Device::CUDA); // To store indices

    size_t n_out = output->size();
    if (n_out == 0) return output;

    int threads = 256;
    int blocks = (n_out + threads - 1) / threads;

    maxpool2d_forward_kernel<<<blocks, threads>>>(
        static_cast<const float*>(input->data_ptr()),
        static_cast<float*>(output->mutable_data_ptr()),
        static_cast<float*>(max_indices_tensor->mutable_data_ptr()),
        N, C, H, W, kernel_size, stride, H_out, W_out
    );
    CUDA_CHECK(cudaPeekAtLastError());

    return output;
}


// ----------------------------------------------------------------------------
// MaxPool2D Backward (CUDA Kernel and Host Function)
// ----------------------------------------------------------------------------
__global__ void maxpool2d_backward_kernel(float* grad_input, const float* grad_output, const float* max_indices_data, size_t n_grad_out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n_grad_out) {
        float grad_out_val = grad_output[index];
        int input_idx = static_cast<int>(max_indices_data[index]);
        if (input_idx != -1) {
            atomicAdd(&grad_input[input_idx], grad_out_val);
        }
    }
}

std::shared_ptr<Tensor> maxpool2d_backward_cuda(const std::shared_ptr<Tensor>& grad_output, const std::shared_ptr<Tensor>& max_indices_tensor, const std::vector<size_t>& input_shape) {
    auto grad_input = Tensor::zeros(input_shape, false, Device::CUDA);

    size_t n_grad_out = grad_output->size();
    if (n_grad_out == 0) return grad_input;

    int threads = 256;
    int blocks = (n_grad_out + threads - 1) / threads;

    maxpool2d_backward_kernel<<<blocks, threads>>>(
        static_cast<float*>(grad_input->mutable_data_ptr()),
        static_cast<const float*>(grad_output->data_ptr()),
        static_cast<const float*>(max_indices_tensor->data_ptr()),
        n_grad_out
    );
    CUDA_CHECK(cudaPeekAtLastError());

    return grad_input;
}

__global__ void rearrange_output_kernel(const float* matmul_data, float* output_data,
                                     int N, int C_out, int H_out, int W_out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * H_out * W_out;

    if (index < total_elements) {
        // 从线性索引分解出 N, C, H, W 坐标
        int w = index % W_out;
        int h = (index / W_out) % H_out;
        int c = (index / (W_out * H_out)) % C_out;
        int n = index / (W_out * H_out * C_out);

        // 计算源数据 (matmul_result) 中的索引
        // matmul_result 的形状是 [C_out, N*H_out*W_out]
        // N*H_out*W_out 是内循环
        int H_W_out = H_out * W_out;
        int src_idx = c * (N * H_W_out) + n * H_W_out + h * W_out + w;
        
        output_data[index] = matmul_data[src_idx];
    }
}

void rearrange_output_kernel_launcher(const float* matmul_data, float* output_data,
                                     int N, int C_out, int H_out, int W_out) {
    size_t n_out = N * C_out * H_out * W_out;
    if (n_out == 0) return;
    int threads = 256;
    int blocks = (n_out + threads - 1) / threads;
    rearrange_output_kernel<<<blocks, threads>>>(matmul_data, output_data, N, C_out, H_out, W_out);
    CUDA_CHECK(cudaPeekAtLastError());
}