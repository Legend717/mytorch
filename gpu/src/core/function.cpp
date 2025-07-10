#include "core/function.h"
#include "core/tensor.h"
#include <stdexcept>
#include <numeric>
#include <omp.h>
#include <algorithm>
#include <iostream>

// 基类
std::shared_ptr<Tensor> Function::apply(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    _saved_inputs = inputs;
    auto output = _forward(inputs);

    bool requires_grad = false;
    for (const auto& input: inputs) {
        if (input->requires_grad()) {
            requires_grad = true;
            break;
        }
    }

    if (requires_grad) {
        output->set_requires_grad(true);
        output->set_ctx(shared_from_this());
    }
    return output;
}


std::vector<std::shared_ptr<Tensor>> Function::backward(const std::shared_ptr<Tensor>& grad_output) {
    return _backward(grad_output);
}

void Function::release_saved_inputs() {
    _saved_inputs.clear(); // <-- 清空保存的输入
}

// 加法
std::shared_ptr<Tensor> Add::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    //std::cout<<"Add"<<"\n";
    if (inputs[0]->device() == Device::CUDA) {
        return add_forward_cuda(inputs[0], inputs[1]);
    }

    auto a = inputs[0];
    auto b = inputs[1];

    // 如果维度不同，则进行广播加法
    if (a->shape() != b->shape() && b->shape()[0] == 1 && a->shape()[1] == b->shape()[1]) {
        std::vector<float> result_data(a->size());
        size_t batch_size = a->shape()[0];
        size_t features = a->shape()[1];
        #pragma omp parallel for
        for(size_t i = 0; i < batch_size; ++i) {
            for(size_t j = 0; j < features; ++j) {
                result_data[i * features + j] = a->data_cpu()[i * features + j] + b->data_cpu()[j];
            }
        }
        return Tensor::create(result_data, a->shape());
    }
    else { // 维度相同，直接加法
        const auto& a_data = a->data_cpu();
        const auto& b_data = b->data_cpu();
        std::vector<float> result_data(a_data.size());
        #pragma omp parallel for
        for (size_t i = 0; i < a_data.size(); ++i) {
            result_data[i] = a_data[i] + b_data[i];
        }
        return Tensor::create(result_data, a->shape());
    }
    // TODO: 维度不同，但不能广播的情况（暂时假设不存在此错误）
}

std::vector<std::shared_ptr<Tensor>> Add::_backward(const std::shared_ptr<Tensor>& grad_output) {
    //std::cout<<"Add_backward begin"<<std::endl;
    // 加法操作的梯度是 1，因此 grad_a 和 grad_b 直接等于 grad_output
    auto a = _saved_inputs[0];
    auto b = _saved_inputs[1];
    auto grad_a = grad_output;
    auto grad_b = grad_output;

    // 处理维度不同的情况，此时 grad_b 需要进行广播
    if (a->shape() != b->shape()) {
        if(b->device() == Device::CUDA){
            grad_b = add_backward_cuda(grad_output, b);
        }else{
            // 由于 b 被广播到 a 的每一行，反向传播时需要将 grad_output 的所有行梯度累加到 b 的对应位置

            size_t batch_size = grad_output->shape()[0];
            size_t features = grad_output->shape()[1];
            std::vector<float> sum_grad_data(features, 0.0f);

            // 优化循环顺序：外循环features，内循环batch_size
            #pragma omp parallel for
            for(size_t j = 0; j < features; ++j) {
                float sum = 0.0f;
                for(size_t i = 0; i < batch_size; ++i) {
                    sum += grad_output->data_cpu()[i * features + j];
                }
                sum_grad_data[j] = sum;
            }
            grad_b = Tensor::create(sum_grad_data, b->shape());
        }

    }
    return {grad_a, grad_b};
}

// 减法，暂不考虑广播情况
std::shared_ptr<Tensor> Sub::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    //std::cout<<"Sub"<<"\n";
    if (inputs[0]->device() == Device::CUDA) {
        // Create a tensor of -1s and multiply
        auto neg_b = inputs[1]->mul(Tensor::create({-1.0f}, {1}, false)->to(Device::CUDA));
        return add_forward_cuda(inputs[0], neg_b);
    }
    const auto& a = inputs[0]->data_cpu();
    const auto& b = inputs[1]->data_cpu();
    std::vector<float> result_data(a.size());
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); ++i) {
        result_data[i] = a[i] - b[i];
    }
    return Tensor::create(result_data, inputs[0]->shape());
}

std::vector<std::shared_ptr<Tensor>> Sub::_backward(const std::shared_ptr<Tensor>& grad_output) {
    //std::cout<<"Sub_backward begin"<<std::endl;
    // 减法的b梯度是-1，因此 grad_b 等于 -grad_output
    auto neg_grad = grad_output->mul(Tensor::create({-1.0f}, {1}))->to(grad_output->device());
    return {grad_output, neg_grad};
}

// 乘法（逐位）
std::shared_ptr<Tensor> Mul::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    //std::cout<<"Mul"<<"\n";
    if (inputs[0]->device() == Device::CUDA) {
        return mul_forward_cuda(inputs[0], inputs[1]);
    }
    const auto& a = inputs[0]->data_cpu();
    const auto& b = inputs[1]->data_cpu();
    std::vector<float> result_data(a.size());
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); ++i) {
        result_data[i] = a[i] * b[i];
    }
    return Tensor::create(result_data, inputs[0]->shape());
}

std::vector<std::shared_ptr<Tensor>> Mul::_backward(const std::shared_ptr<Tensor>& grad_output) {
    //std::cout<<"Mul_backward begin"<<std::endl;
    // z = a * b, dl/da = dl/dz * b, dl/db = dl/dz * a
    auto a = _saved_inputs[0];
    auto b = _saved_inputs[1];
    return {grad_output->mul(b), grad_output->mul(a)};
}

// 矩阵乘法
std::shared_ptr<Tensor> MatMul::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    //std::cout<<"MatMul"<<"\n";
    if (inputs[0]->device() == Device::CUDA) {
        return matmul_forward_cuda(inputs[0], inputs[1]);
    }
    const auto& a = inputs[0]->data_cpu();
    const auto& b = inputs[1]->data_cpu();
    if (a->shape().size() != 2 || b->shape().size() != 2 || a->shape()[1] != b->shape()[0]) {
        throw std::runtime_error("矩阵乘法输入维度不匹配");
    }
    size_t M = a->shape()[0];
    size_t K = a->shape()[1];
    size_t N = b->shape()[1];
    std::vector<float> result_data(M * N, 0.0f);
    #pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t k = 0; k < K; ++k) {
                result_data[i * N + j] += a[i * K + k] * b[k * N + j];
            }
        }
    }
    return Tensor::create(result_data, {M, N});
}

std::vector<std::shared_ptr<Tensor>> MatMul::_backward(const std::shared_ptr<Tensor>& grad_output) {
    //std::cout<<"MatMul_backward begin"<<std::endl;
    auto a = _saved_inputs[0];
    auto b = _saved_inputs[1];
    // grad_a = grad_output @ b.T
    // grad_b = a.T @ grad_output
    auto grad_a = grad_output->matmul(b->transpose());
    auto grad_b = a->transpose()->matmul(grad_output);
    return {grad_a, grad_b};
}

// 求和
std::shared_ptr<Tensor> Sum::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    //std::cout<<"Sum"<<"\n";
    const auto& x = inputs[0];
    if (x->device() == Device::CUDA) {
        // Placeholder for a proper CUDA reduction kernel
        float sum_val = 0.0f;
        auto x_cpu = x->data_cpu();
        for(const auto& val : x_cpu) sum_val += val;
        return Tensor::create({sum_val}, {1}, x->requires_grad())->to(Device::CUDA);
    }
    // float sum_val = std::accumulate(inputs[0]->data_cpu().begin(), inputs[0]->data_cpu().end(), 0.0f);
    const auto& data = inputs[0]->data_cpu();
    float sum_val = 0.0f;
    #pragma omp parallel for reduction(+:sum_val)
    for (size_t i = 0; i < data.size(); ++i) {
        sum_val += data[i];
    }
    return Tensor::create({sum_val}, {1});
}

std::vector<std::shared_ptr<Tensor>> Sum::_backward(const std::shared_ptr<Tensor>& grad_output) {
    //std::cout<<"Sum_backward begin"<<std::endl;
    auto original_input = _saved_inputs[0];
    auto original_shape = original_input->shape();
    auto device = original_input->device();

    // 从输入的梯度张量中获取标量梯度值 (grad_output可能在GPU上)
    float grad_scalar_value = grad_output->item();

    // 创建一个填充了梯度值的 std::vector
    size_t total_size = original_input->size();
    std::vector<float> grad_data(total_size, grad_scalar_value);

    // 先在CPU上创建梯度张量
    auto grad_tensor = Tensor::create(grad_data, original_shape, false);
    
    // 如果原始输入在GPU上，则将梯度也移动到GPU
    if (device == Device::CUDA) {
        return {grad_tensor->to(device)};
    }
    
    return {grad_tensor};
}

// ReLU
std::shared_ptr<Tensor> ReLUFunc::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    //std::cout<<"Relu"<<"\n";
    if (inputs[0]->device() == Device::CUDA) {
        return relu_forward_cuda(inputs[0]);
    }
    const auto& x = inputs[0]->data_cpu();
    std::vector<float> result_data(x.size());
    #pragma omp parallel for
    for(size_t i=0; i<x.size(); ++i) {
        result_data[i] = std::max(0.0f, x[i]);
    }
    return Tensor::create(result_data, inputs[0]->shape());
}

std::vector<std::shared_ptr<Tensor>> ReLUFunc::_backward(const std::shared_ptr<Tensor>& grad_output) {
    //std::cout<<"ReLUFunc_backward begin"<<std::endl;
    auto x = _saved_inputs[0];
    if (x->device() == Device::CUDA) {
        return {relu_backward_cuda(grad_output, x)};
    }

    const auto& x_data = x->data_cpu();
    std::vector<float> mask_data(x_data.size());
    // ReLU的梯度是0或1
    #pragma omp parallel for
    for(size_t i = 0; i < x_data.size(); ++i) {
        mask_data[i] = x_data[i] > 0 ? 1.0f : 0.0f;
    }
    auto mask = Tensor::create(mask_data, x->shape())->to(x->device());
    return {grad_output->mul(mask)};
}

// Reshape
std::shared_ptr<Tensor> ReshapeFunc::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    //std::cout<<"Reshape"<<"\n";
    auto input = inputs[0];
    if (input->device() == Device::CUDA) {
        return reshape_forward_cuda(input, _new_shape);
    }
    return Tensor::create(input->data_cpu(), _new_shape);
}

std::vector<std::shared_ptr<Tensor>> ReshapeFunc::_backward(const std::shared_ptr<Tensor>& grad_output) {
    //std::cout<<"Reshape_backward begin"<<std::endl;
    auto original_shape = _saved_inputs[0]->shape();
    return { grad_output->reshape(original_shape) };
}

// Conv2DFunc
std::shared_ptr<Tensor> Conv2DFunc::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    //std::cout<<"Conv2D"<<"\n";
    auto input = inputs[0];
    auto weight = inputs[1];

    const auto& input_shape = input->shape();
    const auto& weight_shape = weight->shape();
    size_t N = input_shape[0], C_in = input_shape[1], H_in = input_shape[2], W_in = input_shape[3];
    size_t C_out = weight_shape[0], kH = weight_shape[2], kW = weight_shape[3];

    size_t H_out = (H_in + 2 * _padding - kH) / _stride + 1;
    size_t W_out = (W_in + 2 * _padding - kW) / _stride + 1;

    // GPU implementation using im2col
    if (input->device() == Device::CUDA) {
        // 1. Transform input tensor to column matrix
        auto col = im2col_cuda(input, kH, _stride, _padding); // Shape: (C_in*kH*kW, N*H_out*W_out)

        // 2. Reshape weights for matrix multiplication
        auto reshaped_weight = weight->reshape({C_out, C_in * kH * kW});

        // 3. Perform matrix multiplication
        auto matmul_result = reshaped_weight->matmul(col); // Shape: (C_out, N*H_out*W_out)

        // ----------- START: 高效的重排实现 -----------
        // 4. 创建最终的输出张量
        auto output = Tensor::zeros({N, C_out, H_out, W_out}, false, Device::CUDA);
        size_t n_out = output->size();

        if (n_out > 0) {
            int threads = 256;
            int blocks = (n_out + threads - 1) / threads;
            
            // 5. 调用新的内核，一次性完成数据重排
            rearrange_output_kernel_launcher(
                static_cast<const float*>(matmul_result->data_ptr()),
                static_cast<float*>(output->mutable_data_ptr()),
                N, C_out, H_out, W_out
            );
        }
        
        return output;
    }

    // CPU implementation using im2col
    else {
        std::vector<float> output_data(N * C_out * H_out * W_out, 0.0f);
        const auto& input_data = input->data_cpu();
        const auto& weight_data = weight->data_cpu();

        #pragma omp parallel for
        for (size_t n = 0; n < N; ++n) {
            for (size_t c_out = 0; c_out < C_out; ++c_out) {
                for (size_t h_out = 0; h_out < H_out; ++h_out) {
                    for (size_t w_out = 0; w_out < W_out; ++w_out) {
                        float acc = 0.0f;
                        for (size_t c_in = 0; c_in < C_in; ++c_in) {
                            for (size_t kh = 0; kh < kH; ++kh) {
                                for (size_t kw = 0; kw < kW; ++kw) {
                                    int h_in = h_out * _stride - _padding + kh;
                                    int w_in = w_out * _stride - _padding + kw;
                                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                        float in_val = input_data[n*C_in*H_in*W_in + c_in*H_in*W_in + h_in*W_in + w_in];
                                        float w_val = weight_data[c_out*C_in*kH*kW + c_in*kH*kW + kh*kW + kw];
                                        acc += in_val * w_val;
                                    }
                                }
                            }
                        }
                        output_data[n*C_out*H_out*W_out + c_out*H_out*W_out + h_out*W_out + w_out] = acc;
                    }
                }
            }
        }
        return Tensor::create(output_data, {N, C_out, H_out, W_out});
    }
}

std::vector<std::shared_ptr<Tensor>> Conv2DFunc::_backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& input = _saved_inputs[0];
    const auto& weight = _saved_inputs[1];

    // --- GPU Implementation ---
    if (grad_output->device() == Device::CUDA) {
        return conv2d_backward_cuda(grad_output, input, weight, _stride, _padding);
    }

    // --- CPU Implementation ---
    const auto& in_shape = input->shape();
    const auto& w_shape = weight->shape();
    const auto& grad_out_shape = grad_output->shape();
    size_t N = in_shape[0];
    size_t C_in = in_shape[1];
    size_t H_in = in_shape[2];
    size_t W_in = in_shape[3];
    size_t C_out = w_shape[0];
    size_t H_w = w_shape[2];
    size_t W_w = w_shape[3];
    size_t H_out = grad_out_shape[2];
    size_t W_out = grad_out_shape[3];

    // 初始化梯度张量
    auto grad_input_tensor = Tensor::zeros(in_shape, false);
    auto grad_weight_tensor = Tensor::zeros(w_shape, false);

    // 获取可变数据指针
    auto grad_input_data = static_cast<std::vector<float>*>(grad_input_tensor->mutable_data_ptr());
    auto grad_weight_data = static_cast<std::vector<float>*>(grad_weight_tensor->mutable_data_ptr());

    // 获取常量数据
    const auto& input_data = input->data_cpu();
    const auto& weight_data = weight->data_cpu();
    const auto& grad_output_data = grad_output->data_cpu();

    #pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {
        for (size_t c_out = 0; c_out < C_out; ++c_out) {
            for (size_t h_out = 0; h_out < H_out; ++h_out) {
                for (size_t w_out = 0; w_out < W_out; ++w_out) {
                    
                    size_t grad_out_idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out;
                    float grad_out_val = grad_output_data[grad_out_idx];

                    if (grad_out_val == 0.0f) continue;

                    for (size_t c_in = 0; c_in < C_in; ++c_in) {
                        for (size_t kh = 0; kh < H_w; ++kh) {
                            for (size_t kw = 0; kw < W_w; ++kw) {
                                int h_in_idx_int = static_cast<int>(h_out * _stride) - _padding + kh;
                                int w_in_idx_int = static_cast<int>(w_out * _stride) - _padding + kw;

                                if (h_in_idx_int >= 0 && h_in_idx_int < H_in && w_in_idx_int >= 0 && w_in_idx_int < W_in) {
                                    
                                    size_t input_idx = n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_in_idx_int * W_in + w_in_idx_int;
                                    size_t weight_idx = c_out * (C_in * H_w * W_w) + c_in * (H_w * W_w) + kh * W_w + kw;
                                    
                                    #pragma omp atomic
                                    (*grad_weight_data)[weight_idx] += grad_out_val * input_data[input_idx];
                                    
                                    #pragma omp atomic
                                    (*grad_input_data)[input_idx] += grad_out_val * weight_data[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return {grad_input_tensor, grad_weight_tensor};
}

// MaxPool2DFunc
std::shared_ptr<Tensor> MaxPool2DFunc::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    //std::cout<<"MaxPool"<<"\n";
    auto input = inputs[0];

    // GPU implementation
    if (input->device() == Device::CUDA) {
        // Pass the tensor meant for CUDA, not the std::vector
        return maxpool2d_forward_cuda(input, _kernel_size, _stride, _max_indices_tensor);
    }

    // CPU implementation
    const auto& in_shape = input->shape();
    const size_t N = in_shape[0], C = in_shape[1], H_in = in_shape[2], W_in = in_shape[3];

    const size_t H_out = (H_in - _kernel_size) / _stride + 1;
    const size_t W_out = (W_in - _kernel_size) / _stride + 1;
    const std::vector<size_t> out_shape = {N, C, H_out, W_out};

    // Initialize output tensor and resize the CPU indices vector
    auto output = Tensor::zeros(out_shape, false);
    _max_indices.resize(N * C * H_out * W_out);

    // Use the correct methods to access data on the CPU
    const auto& input_data = input->data_cpu();
    auto output_data_vec = static_cast<std::vector<float>*>(output->mutable_data_ptr());

    // Loop for calculation
    #pragma omp parallel for
    for (int i = 0; i < N * C; ++i) {
        int n = i / C;
        int c = i % C;
        for (size_t h_out = 0; h_out < H_out; ++h_out) {
            for (size_t w_out = 0; w_out < W_out; ++w_out) {

                float max_val = -std::numeric_limits<float>::infinity();
                size_t max_idx = 0;

                for (size_t kh = 0; kh < _kernel_size; ++kh) {
                    for (size_t kw = 0; kw < _kernel_size; ++kw) {
                        size_t h_in = h_out * _stride + kh;
                        size_t w_in = w_out * _stride + kw;
                        size_t in_idx = n * (C * H_in * W_in) + c * (H_in * W_in) + h_in * W_in + w_in;

                        if (input_data[in_idx] > max_val) {
                            max_val = input_data[in_idx];
                            max_idx = in_idx;
                        }
                    }
                }

                size_t out_idx = n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out;

                (*output_data_vec)[out_idx] = max_val;
                _max_indices[out_idx] = max_idx;
            }
        }
    }

    return output;
}

std::vector<std::shared_ptr<Tensor>> MaxPool2DFunc::_backward(const std::shared_ptr<Tensor>& grad_output) {
    auto input = _saved_inputs[0];

    // GPU implementation
    if (grad_output->device() == Device::CUDA) {
        // Pass the tensor meant for CUDA
        return {maxpool2d_backward_cuda(grad_output, _max_indices_tensor, input->shape())};
    }

    // CPU implementation
    const auto& in_shape = input->shape();
    auto grad_input = Tensor::zeros(in_shape, false);

    // Use the correct methods to access data
    const auto& grad_output_data = grad_output->data_cpu();
    auto grad_input_data_vec = static_cast<std::vector<float>*>(grad_input->mutable_data_ptr());

    // Use _max_indices to route gradients
    for (size_t i = 0; i < grad_output_data.size(); ++i) {
        size_t max_idx = _max_indices[i];
        (*grad_input_data_vec)[max_idx] += grad_output_data[i];
    }

    return {grad_input};
}