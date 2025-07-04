#include "core/tensor.h"
#include "core/function.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cfloat>

// --- CUDA Error Checking Macro ---
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error in %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// --- SGD Update Kernel ---
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

// --- Kernels ---

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

// ✨ NEW ✨: Kernel for multiplying a tensor by a scalar
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

__global__ void im2col_kernel(const float* data_im, int channels, int height, int width,
                              int kernel_h, int kernel_w, int pad_h, int pad_w,
                              int stride_h, int stride_w,
                              int height_col, int width_col, float* data_col) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int num_kernels = channels * height_col * width_col;
    if (index >= num_kernels) return;

    int h_index = index / width_col;
    int w_out = index % width_col;
    int c_in = h_index / height_col;
    int h_out = h_index % height_col;

    int h_in_start = h_out * stride_h - pad_h;
    int w_in_start = w_out * stride_w - pad_w;

    int out_start_col = (c_in * height_col + h_out) * width_col + w_out;

    for (int i = 0; i < kernel_h; ++i) {
        for (int j = 0; j < kernel_w; ++j) {
            int h_in = h_in_start + i;
            int w_in = w_in_start + j;
            int out_row = (i * kernel_w) + j;
            
            float val = 0;
            if (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) {
                val = data_im[(c_in * height + h_in) * width + w_in];
            }
            data_col[out_row * (height_col * width_col) + out_start_col] = val;
        }
    }
}

__global__ void max_pool_forward_kernel(const float* input, float* output, size_t* max_indices,
                                        size_t N, size_t C, size_t H, size_t W,
                                        size_t H_out, size_t W_out,
                                        size_t kernel_size, size_t stride) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * C * H_out * W_out) {
        return;
    }

    int w_out = index % W_out;
    int h_out = (index / W_out) % H_out;
    int c = (index / (W_out * H_out)) % C;
    int n = index / (C * W_out * H_out);

    int h_start = h_out * stride;
    int w_start = w_out * stride;

    float max_val = -FLT_MAX;
    size_t max_idx = 0;

    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int h_in = h_start + i;
            int w_in = w_start + j;
            if (h_in < H && w_in < W) {
                size_t input_idx = n * (C * H * W) + c * (H * W) + h_in * W + w_in;
                float val = input[input_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = input_idx;
                }
            }
        }
    }
    output[index] = max_val;
    if (max_indices != nullptr) {
        max_indices[index] = max_idx;
    }
}


// --- Host-side launch functions ---

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
        throw std::runtime_error("CUDA Add broadcast shape not supported yet.");
    }
    CUDA_CHECK(cudaPeekAtLastError());
    return output;
}

std::shared_ptr<Tensor> sub_forward_cuda(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    if (a->shape() != b->shape()) {
        throw std::runtime_error("CUDA Sub requires same shapes");
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

// ✨ MODIFIED: mul_forward_cuda now supports scalar multiplication
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
        throw std::runtime_error("CUDA Mul requires same shapes or one scalar operand");
    }

    // Handle scalar multiplication
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

std::shared_ptr<Tensor> conv2d_forward_cuda(const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& weight, size_t stride, size_t padding) {
    size_t N = input->shape()[0];
    size_t C = input->shape()[1];
    size_t H = input->shape()[2];
    size_t W = input->shape()[3];

    size_t K_out = weight->shape()[0];
    size_t K_in = weight->shape()[1];
    size_t KH = weight->shape()[2];
    size_t KW = weight->shape()[3];

    if (C != K_in) { throw std::runtime_error("Conv2D input channels and weight channels mismatch."); }
    
    size_t H_out = (H + 2 * padding - KH) / stride + 1;
    size_t W_out = (W + 2 * padding - KW) / stride + 1;

    auto output = Tensor::zeros({N, K_out, H_out, W_out}, input->requires_grad(), Device::CUDA);

    size_t col_buffer_rows = C * KH * KW;
    size_t col_buffer_cols = H_out * W_out;

    float* col_buffer_gpu;
    CUDA_CHECK(cudaMalloc(&col_buffer_gpu, col_buffer_rows * col_buffer_cols * sizeof(float)));

    int M = K_out;
    int K = col_buffer_rows;
    int N_gemm = col_buffer_cols;
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    for(size_t i = 0; i < N; ++i) {
        const float* input_im_ptr = static_cast<const float*>(input->data_ptr()) + i * (C * H * W);
        
        int num_kernels = C * H_out * W_out;
        int threads_per_block = 512;
        int blocks = (num_kernels + threads_per_block - 1) / threads_per_block;
        im2col_kernel<<<blocks, threads_per_block>>>(input_im_ptr, C, H, W, KH, KW, padding, padding, stride, stride, H_out, W_out, col_buffer_gpu);
        CUDA_CHECK(cudaPeekAtLastError());

        float* output_im_ptr = static_cast<float*>(output->mutable_data_ptr()) + i * (K_out * H_out * W_out);

        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N_gemm, M, K,
                                 &alpha,
                                 col_buffer_gpu, N_gemm,
                                 static_cast<const float*>(weight->data_ptr()), K,
                                 &beta,
                                 output_im_ptr, N_gemm));
    }
    
    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(col_buffer_gpu));
    
    return output;
}

// --- Conv2D Backward Kernels ---

// col2im Kernel: The inverse of im2col. It "scatters" the columns back into an image.
__global__ void col2im_kernel(const float* data_col, int channels, int height, int width,
                              int kernel_h, int kernel_w, int pad_h, int pad_w,
                              int stride_h, int stride_w,
                              int height_col, int width_col, float* data_im) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int num_kernels = channels * height * width;
    if (index >= num_kernels) return;

    int w_im = index % width;
    int h_im = (index / width) % height;
    int c_im = index / (width * height);

    float val = 0;
    for (int h_col = 0; h_col < height_col; ++h_col) {
        for (int w_col = 0; w_col < width_col; ++w_col) {
            int h_k = h_im + pad_h - h_col * stride_h;
            int w_k = w_im + pad_w - w_col * stride_w;

            if (h_k >= 0 && h_k < kernel_h && w_k >= 0 && w_k < kernel_w) {
                 int data_col_c = c_im / (kernel_h * kernel_w);
                 int data_col_h = h_k;
                 int data_col_w = w_k;

                 int col_index = (((data_col_c * kernel_h + data_col_h) * kernel_w + data_col_w) * height_col + h_col) * width_col + w_col;
                 val += data_col[col_index];
            }
        }
    }
    data_im[index] = val;
}

// Kernel to sum gradients for the bias term
__global__ void sum_bias_kernel(const float* grad_output, float* bias_grad,
                                size_t N, size_t K_out, size_t H_out, size_t W_out) {
    extern __shared__ float sdata[];
    int k = blockIdx.x; // Each block computes sum for one output channel k

    float my_sum = 0;
    for (int i = threadIdx.x; i < N * H_out * W_out; i += blockDim.x) {
        int n = i / (H_out * W_out);
        int h = (i / W_out) % H_out;
        int w = i % W_out;
        my_sum += grad_output[(n * K_out + k) * H_out * W_out + h * W_out + w];
    }

    sdata[threadIdx.x] = my_sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        bias_grad[k] = sdata[0];
    }
}


// --- Conv2D Backward Host-side Launcher ---

std::vector<std::shared_ptr<Tensor>> conv2d_backward_cuda(
    const std::shared_ptr<Tensor>& grad_output,
    const std::shared_ptr<Tensor>& input,
    const std::shared_ptr<Tensor>& weight,
    size_t stride, size_t padding
) {
    // Get shapes
    size_t N = input->shape()[0], C = input->shape()[1], H = input->shape()[2], W = input->shape()[3];
    size_t K_out = weight->shape()[0], K_in = weight->shape()[1], KH = weight->shape()[2], KW = weight->shape()[3];
    size_t H_out = grad_output->shape()[2], W_out = grad_output->shape()[3];

    // Create gradient tensors to be returned
    auto grad_input = Tensor::zeros(input->shape(), false, Device::CUDA);
    auto grad_weight = Tensor::zeros(weight->shape(), false, Device::CUDA);
    auto grad_bias = Tensor::zeros({1, K_out, 1, 1}, false, Device::CUDA);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // --- 1. Calculate grad_weight ---
    // grad_weight = im2col(input)^T * grad_output
    size_t col_buffer_rows = C * KH * KW;
    size_t col_buffer_cols = H_out * W_out;
    float* col_buffer_gpu;
    CUDA_CHECK(cudaMalloc(&col_buffer_gpu, col_buffer_rows * col_buffer_cols * sizeof(float)));

    // We process one image at a time from the batch
    for (size_t i = 0; i < N; ++i) {
        const float* input_im_ptr = static_cast<const float*>(input->data_ptr()) + i * (C * H * W);
        const float* grad_output_ptr = static_cast<const float*>(grad_output->data_ptr()) + i * (K_out * H_out * W_out);

        // im2col for the input image
        int num_kernels_im2col = C * H_out * W_out;
        int threads_per_block_im2col = 512;
        int blocks_im2col = (num_kernels_im2col + threads_per_block_im2col - 1) / threads_per_block_im2col;
        im2col_kernel<<<blocks_im2col, threads_per_block_im2col>>>(input_im_ptr, C, H, W, KH, KW, padding, padding, stride, stride, H_out, W_out, col_buffer_gpu);
        CUDA_CHECK(cudaPeekAtLastError());

        // GEMM: grad_weight += col_buffer^T * grad_output
        // M = K_out, K = H_out * W_out, N = C*KH*KW
        const float beta_add = 1.0f; // Use beta=1 to accumulate gradients over the batch
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 col_buffer_rows, K_out, col_buffer_cols,
                                 &alpha,
                                 col_buffer_gpu, col_buffer_cols,
                                 grad_output_ptr, col_buffer_cols,
                                 &beta_add,
                                 static_cast<float*>(grad_weight->mutable_data_ptr()), col_buffer_rows));
    }


    // --- 2. Calculate grad_input ---
    // grad_input = col2im(weight^T * grad_output)
    // First, GEMM: col_buffer = weight^T * grad_output
    // M = C*KH*KW, K = K_out, N = H_out*W_out
    int M_grad_in = C * KH * KW;
    int K_grad_in = K_out;
    int N_grad_in = H_out * W_out;
    for (size_t i = 0; i < N; ++i) {
        const float* grad_output_ptr = static_cast<const float*>(grad_output->data_ptr()) + i * (K_out * H_out * W_out);
        float* grad_input_ptr = static_cast<float*>(grad_input->mutable_data_ptr()) + i * (C * H * W);

        // GEMM
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 N_grad_in, M_grad_in, K_grad_in,
                                 &alpha,
                                 grad_output_ptr, N_grad_in,
                                 static_cast<const float*>(weight->data_ptr()), M_grad_in,
                                 &beta,
                                 col_buffer_gpu, N_grad_in));

        // col2im to get grad_input
        int num_kernels_col2im = C * H * W;
        int threads_per_block_col2im = 256;
        int blocks_col2im = (num_kernels_col2im + threads_per_block_col2im - 1) / threads_per_block_col2im;
        col2im_kernel<<<blocks_col2im, threads_per_block_col2im>>>(col_buffer_gpu, C, H, W, KH, KW, padding, padding, stride, stride, H_out, W_out, grad_input_ptr);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    // --- 3. Calculate grad_bias ---
    int threads_bias = 256;
    int blocks_bias = K_out;
    sum_bias_kernel<<<blocks_bias, threads_bias, threads_bias * sizeof(float)>>>(
        static_cast<const float*>(grad_output->data_ptr()),
        static_cast<float*>(grad_bias->mutable_data_ptr()),
        N, K_out, H_out, W_out
    );
    CUDA_CHECK(cudaPeekAtLastError());

    // Cleanup
    CUDA_CHECK(cudaFree(col_buffer_gpu));
    cublasDestroy(handle);

    return {grad_input, grad_weight, grad_bias};
}



std::shared_ptr<Tensor> maxpool2d_forward_cuda(const std::shared_ptr<Tensor>& input, size_t kernel_size, size_t stride, std::vector<size_t>& max_indices) {
    size_t N = input->shape()[0];
    size_t C = input->shape()[1];
    size_t H = input->shape()[2];
    size_t W = input->shape()[3];

    size_t H_out = (H - kernel_size) / stride + 1;
    size_t W_out = (W - kernel_size) / stride + 1;
    
    auto output = Tensor::zeros({N, C, H_out, W_out}, input->requires_grad(), Device::CUDA);
    
    size_t* d_max_indices;
    size_t total_output_size = N * C * H_out * W_out;
    CUDA_CHECK(cudaMalloc(&d_max_indices, total_output_size * sizeof(size_t)));

    int threads = 256;
    int blocks = (total_output_size + threads - 1) / threads;
    max_pool_forward_kernel<<<blocks, threads>>>( static_cast<const float*>(input->data_ptr()), static_cast<float*>(output->mutable_data_ptr()), d_max_indices, N, C, H, W, H_out, W_out, kernel_size, stride );
    CUDA_CHECK(cudaPeekAtLastError());

    max_indices.resize(total_output_size);
    CUDA_CHECK(cudaMemcpy(max_indices.data(), d_max_indices, total_output_size * sizeof(size_t), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_max_indices));
    
    return output;
}

__global__ void maxpool2d_backward_kernel(const float* grad_output, float* grad_input, const size_t* max_indices, size_t n_output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_output) {
        return;
    }

    // 获取输出梯度值
    float grad_val = grad_output[index];

    // 获取该梯度对应的、在前向传播时的原始输入位置
    size_t input_idx = max_indices[index];

    // 使用原子操作将梯度值加到输入梯度张量的对应位置
    // atomicAdd 对于scatter操作是安全的
    atomicAdd(&grad_input[input_idx], grad_val);
}

// --- MaxPool2D Backward Host-side Launcher ---

std::shared_ptr<Tensor> maxpool2d_backward_cuda(const std::shared_ptr<Tensor>& grad_output, const std::shared_ptr<Tensor>& input, const std::vector<size_t>& max_indices) {
    // 创建一个新的输入梯度张量，其形状与原始输入相同，并初始化为0
    auto grad_input = Tensor::zeros(input->shape(), false, Device::CUDA);

    size_t output_size = grad_output->size();
    if (output_size == 0) {
        return grad_input; // 如果没有输出，则无需计算
    }

    // 将 max_indices 从主机内存(std::vector)拷贝到设备内存
    size_t* d_max_indices;
    CUDA_CHECK(cudaMalloc(&d_max_indices, max_indices.size() * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_max_indices, max_indices.data(), max_indices.size() * sizeof(size_t), cudaMemcpyHostToDevice));

    // 配置并启动CUDA核函数
    int threads = 256;
    int blocks = (output_size + threads - 1) / threads;
    maxpool2d_backward_kernel<<<blocks, threads>>>(
        static_cast<const float*>(grad_output->data_ptr()),
        static_cast<float*>(grad_input->mutable_data_ptr()),
        d_max_indices,
        output_size
    );
    CUDA_CHECK(cudaPeekAtLastError());

    // 释放为indices分配的设备内存
    CUDA_CHECK(cudaFree(d_max_indices));

    return grad_input;
}