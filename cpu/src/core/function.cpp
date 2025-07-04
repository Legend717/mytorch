#include "core/function.h"
#include "core/tensor.h"
#include <stdexcept>
#include <numeric>

// --- 基类实现 ---
std::shared_ptr<Tensor> Function::apply(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    _saved_inputs = inputs;
    auto output = _forward(inputs); // 执行前向计算

    bool requires_grad = false;
    for (const auto& input : inputs) {
        if (input->requires_grad()) {
            requires_grad = true;
            break;
        }
    }

    if (requires_grad) {
        // --- 这是关键的修复 ---
        // 1. 让输出张量也需要梯度
        output->_requires_grad = true;
        // 2. 设置它的上下文，以便反向传播
        output->set_ctx(shared_from_this());
    }

    return output;
}


std::vector<std::shared_ptr<Tensor>> Function::backward(const std::shared_ptr<Tensor>& grad_output) {
    return _backward(grad_output);
}

// --- Add (with broadcasting support) ---
std::shared_ptr<Tensor> Add::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    auto a = inputs[0];
    auto b = inputs[1];

    // Simple broadcasting: assume b is broadcast over a
    if (a->shape() != b->shape() && b->shape()[0] == 1 && a->shape()[1] == b->shape()[1]) {
        std::vector<float> result_data(a->data().size());
        size_t batch_size = a->shape()[0];
        size_t features = a->shape()[1];
        for(size_t i = 0; i < batch_size; ++i) {
            for(size_t j = 0; j < features; ++j) {
                result_data[i * features + j] = a->data()[i * features + j] + b->data()[j];
            }
        }
        return Tensor::create(result_data, a->shape());
    } else { // Standard element-wise addition
        const auto& a_data = a->data();
        const auto& b_data = b->data();
        std::vector<float> result_data(a_data.size());
        for (size_t i = 0; i < a_data.size(); ++i) {
            result_data[i] = a_data[i] + b_data[i];
        }
        return Tensor::create(result_data, a->shape());
    }
}

std::vector<std::shared_ptr<Tensor>> Add::_backward(const std::shared_ptr<Tensor>& grad_output) {
    auto a = _saved_inputs[0];
    auto b = _saved_inputs[1];
    auto grad_a = grad_output;
    auto grad_b = grad_output;

    // Handle broadcasting in backward pass
    if (a->shape() != b->shape()) {
        // The gradient for the broadcasted tensor 'b' needs to be summed up.
        // grad_output shape is [batch, features], b shape is [1, features]
        size_t batch_size = grad_output->shape()[0];
        size_t features = grad_output->shape()[1];
        std::vector<float> sum_grad_data(features, 0.0f);
        for(size_t i = 0; i < batch_size; ++i) {
            for(size_t j = 0; j < features; ++j) {
                sum_grad_data[j] += grad_output->data()[i * features + j];
            }
        }
        grad_b = Tensor::create(sum_grad_data, b->shape());
    }
    return {grad_a, grad_b};
}

// --- Sub ---
std::shared_ptr<Tensor> Sub::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    const auto& a = inputs[0]->data();
    const auto& b = inputs[1]->data();
    std::vector<float> result_data(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result_data[i] = a[i] - b[i];
    }
    return Tensor::create(result_data, inputs[0]->shape());
}

std::vector<std::shared_ptr<Tensor>> Sub::_backward(const std::shared_ptr<Tensor>& grad_output) {
    auto neg_grad = grad_output->mul(Tensor::create({-1.0f}, {1}));
    return {grad_output, neg_grad};
}


// --- Mul ---
std::shared_ptr<Tensor> Mul::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    const auto& a = inputs[0]->data();
    const auto& b = inputs[1]->data();
    std::vector<float> result_data(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result_data[i] = a[i] * b[i];
    }
    return Tensor::create(result_data, inputs[0]->shape());
}

std::vector<std::shared_ptr<Tensor>> Mul::_backward(const std::shared_ptr<Tensor>& grad_output) {
    auto in0 = _saved_inputs[0];
    auto in1 = _saved_inputs[1];
    return {grad_output->mul(in1), grad_output->mul(in0)};
}


// --- MatMul ---
std::shared_ptr<Tensor> MatMul::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    if (a->shape().size() != 2 || b->shape().size() != 2 || a->shape()[1] != b->shape()[0]) {
        throw std::runtime_error("MatMul shape mismatch.");
    }
    size_t M = a->shape()[0];
    size_t K = a->shape()[1];
    size_t N = b->shape()[1];
    std::vector<float> result_data(M * N, 0.0f);
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t k = 0; k < K; ++k) {
                result_data[i * N + j] += a->data()[i * K + k] * b->data()[k * N + j];
            }
        }
    }
    return Tensor::create(result_data, {M, N});
}

std::vector<std::shared_ptr<Tensor>> MatMul::_backward(const std::shared_ptr<Tensor>& grad_output) {
    auto a = _saved_inputs[0];
    auto b = _saved_inputs[1];
    // grad_a = grad_output @ b.T
    // grad_b = a.T @ grad_output
    auto grad_a = grad_output->matmul(b->transpose());
    auto grad_b = a->transpose()->matmul(grad_output);
    return {grad_a, grad_b};
}


// --- Sum ---
std::shared_ptr<Tensor> Sum::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    float sum_val = std::accumulate(inputs[0]->data().begin(), inputs[0]->data().end(), 0.0f);
    return Tensor::create({sum_val}, {1});
}

std::vector<std::shared_ptr<Tensor>> Sum::_backward(const std::shared_ptr<Tensor>& grad_output) {
    auto original_shape = _saved_inputs[0]->shape();
    // Create a tensor of the original shape, where each element is the incoming gradient.
    // This correctly "un-sums" the gradient.
    size_t total_size = 1;
    for(auto dim : original_shape) total_size *= dim;
    std::vector<float> grad_data(total_size, grad_output->data()[0]);
    return {Tensor::create(grad_data, original_shape)};
}

// --- ReLUFunc ---
std::shared_ptr<Tensor> ReLUFunc::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    const auto& x = inputs[0]->data();
    std::vector<float> result_data(x.size());
    for(size_t i=0; i<x.size(); ++i) {
        result_data[i] = std::max(0.0f, x[i]);
    }
    return Tensor::create(result_data, inputs[0]->shape());
}

std::vector<std::shared_ptr<Tensor>> ReLUFunc::_backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = _saved_inputs[0]->data();
    std::vector<float> mask_data(x.size());
    for(size_t i=0; i<x.size(); ++i) {
        mask_data[i] = x[i] > 0 ? 1.0f : 0.0f;
    }
    auto mask = Tensor::create(mask_data, _saved_inputs[0]->shape());
    return {grad_output->mul(mask)};
}

// --- Conv2DFunc ---
// 这是整个框架中最复杂的操作之一
std::shared_ptr<Tensor> Conv2DFunc::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    auto& input = inputs[0];   // Shape: [N, C_in, H_in, W_in]
    auto& weight = inputs[1];  // Shape: [C_out, C_in, K_h, K_w]

    // 获取维度信息
    size_t N = input->shape()[0];
    size_t C_in = input->shape()[1];
    size_t H_in = input->shape()[2];
    size_t W_in = input->shape()[3];

    size_t C_out = weight->shape()[0];
    size_t K_h = weight->shape()[2];
    size_t K_w = weight->shape()[3];

    // 计算输出维度 (stride=1, padding=0)
    size_t H_out = H_in - K_h + 1;
    size_t W_out = W_in - K_w + 1;

    // 初始化输出张量
    std::vector<size_t> out_shape = {N, C_out, H_out, W_out};
    std::vector<float> out_data(N * C_out * H_out * W_out, 0.0f);

    // 执行卷积操作 (6重循环，非常适合并行化)
    for (size_t n = 0; n < N; ++n) {                     // 遍历每个样本
        for (size_t c_out = 0; c_out < C_out; ++c_out) {   // 遍历每个输出通道 (决定使用哪个卷积核)
            for (size_t h = 0; h < H_out; ++h) {           // 遍历输出的高度
                for (size_t w = 0; w < W_out; ++w) {       // 遍历输出的宽度
                    float sum = 0.0f;
                    for (size_t c_in = 0; c_in < C_in; ++c_in) { // 遍历每个输入通道
                        for (size_t kh = 0; kh < K_h; ++kh) {  // 遍历卷积核的高度
                            for (size_t kw = 0; kw < K_w; ++kw) {  // 遍历卷积核的宽度
                                // 输入图像中的位置
                                size_t h_in_idx = h + kh;
                                size_t w_in_idx = w + kw;
                                
                                // 从多维索引计算一维索引
                                size_t input_idx = n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_in_idx * W_in + w_in_idx;
                                size_t weight_idx = c_out * (C_in * K_h * K_w) + c_in * (K_h * K_w) + kh * K_w + kw;
                                
                                sum += input->data()[input_idx] * weight->data()[weight_idx];
                            }
                        }
                    }
                    size_t out_idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h * W_out + w;
                    out_data[out_idx] = sum;
                }
            }
        }
    }
    return Tensor::create(out_data, out_shape);
}

std::vector<std::shared_ptr<Tensor>> Conv2DFunc::_backward(const std::shared_ptr<Tensor>& grad_output) {
    auto& input = _saved_inputs[0];   // [N, C_in, H_in, W_in]
    auto& weight = _saved_inputs[1];  // [C_out, C_in, K_h, K_w]

    // 获取维度信息
    size_t N = input->shape()[0], C_in = input->shape()[1], H_in = input->shape()[2], W_in = input->shape()[3];
    size_t C_out = weight->shape()[0], K_h = weight->shape()[2], K_w = weight->shape()[3];
    size_t H_out = grad_output->shape()[2], W_out = grad_output->shape()[3];

    // 初始化梯度张量
    auto grad_input = Tensor::zeros(input->shape());
    auto grad_weight = Tensor::zeros(weight->shape());

    // --- 计算 grad_input ---
    // 这相当于用 grad_output 对一个“旋转180度”的卷积核做“完全”卷积
    for (size_t n = 0; n < N; ++n) {
        for (size_t c_out = 0; c_out < C_out; ++c_out) {
            for (size_t h_out = 0; h_out < H_out; ++h_out) {
                for (size_t w_out = 0; w_out < W_out; ++w_out) {
                    float grad_out_val = grad_output->data()[n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out];
                    for (size_t c_in = 0; c_in < C_in; ++c_in) {
                        for (size_t kh = 0; kh < K_h; ++kh) {
                            for (size_t kw = 0; kw < K_w; ++kw) {
                                size_t h_in_idx = h_out + kh;
                                size_t w_in_idx = w_out + kw;
                                
                                size_t grad_input_idx = n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_in_idx * W_in + w_in_idx;
                                size_t weight_idx = c_out * (C_in * K_h * K_w) + c_in * (K_h * K_w) + kh * K_w + kw;
                                
                                grad_input->get_shared_data()->at(grad_input_idx) += grad_out_val * weight->data()[weight_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    
    // --- 计算 grad_weight ---
    // 这相当于用输入 input 对 grad_output 做卷积
    for (size_t c_out = 0; c_out < C_out; ++c_out) {
        for (size_t c_in = 0; c_in < C_in; ++c_in) {
            for (size_t kh = 0; kh < K_h; ++kh) {
                for (size_t kw = 0; kw < K_w; ++kw) {
                    float sum = 0.0f;
                    for (size_t n = 0; n < N; ++n) {
                        for (size_t h_out = 0; h_out < H_out; ++h_out) {
                            for (size_t w_out = 0; w_out < W_out; ++w_out) {
                                size_t h_in_idx = h_out + kh;
                                size_t w_in_idx = w_out + kw;

                                size_t input_idx = n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_in_idx * W_in + w_in_idx;
                                size_t grad_out_idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out;

                                sum += input->data()[input_idx] * grad_output->data()[grad_out_idx];
                            }
                        }
                    }
                    size_t grad_weight_idx = c_out * (C_in * K_h * K_w) + c_in * (K_h * K_w) + kh * K_w + kw;
                    grad_weight->get_shared_data()->at(grad_weight_idx) = sum;
                }
            }
        }
    }

    // 返回的梯度列表需要与输入的顺序一致 (input, weight)
    return {grad_input, grad_weight};
}

// --- MaxPool2DFunc ---
std::shared_ptr<Tensor> MaxPool2DFunc::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    auto& input = inputs[0]; // Shape: [N, C, H_in, W_in]

    size_t N = input->shape()[0], C = input->shape()[1], H_in = input->shape()[2], W_in = input->shape()[3];
    size_t H_out = (H_in - _kernel_size) / _stride + 1;
    size_t W_out = (W_in - _kernel_size) / _stride + 1;
    
    std::vector<size_t> out_shape = {N, C, H_out, W_out};
    std::vector<float> out_data(N * C * H_out * W_out);
    _max_indices.resize(out_data.size());

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t h = 0; h < H_out; ++h) {
                for (size_t w = 0; w < W_out; ++w) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    size_t max_idx = -1;

                    // 在池化窗口内寻找最大值
                    for (size_t kh = 0; kh < _kernel_size; ++kh) {
                        for (size_t kw = 0; kw < _kernel_size; ++kw) {
                            size_t h_in_idx = h * _stride + kh;
                            size_t w_in_idx = w * _stride + kw;
                            size_t current_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + h_in_idx*W_in + w_in_idx;
                            if (input->data()[current_idx] > max_val) {
                                max_val = input->data()[current_idx];
                                max_idx = current_idx;
                            }
                        }
                    }
                    size_t out_idx = n*(C*H_out*W_out) + c*(H_out*W_out) + h*W_out + w;
                    out_data[out_idx] = max_val;
                    _max_indices[out_idx] = max_idx; // 缓存最大值在原图中的一维索引
                }
            }
        }
    }
    return Tensor::create(out_data, out_shape);
}

std::vector<std::shared_ptr<Tensor>> MaxPool2DFunc::_backward(const std::shared_ptr<Tensor>& grad_output) {
    auto& input = _saved_inputs[0];
    auto grad_input = Tensor::zeros(input->shape());
    
    // 梯度只流向最大值所在的位置
    for (size_t i = 0; i < grad_output->data().size(); ++i) {
        grad_input->get_shared_data()->at(_max_indices[i]) += grad_output->data()[i];
    }

    return {grad_input};
}

// --- ReshapeFunc ---
std::shared_ptr<Tensor> ReshapeFunc::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    auto input = inputs[0];
    // A reshape doesn't change the data, just the metadata (shape).
    // The create function will check if the total number of elements matches.
    return Tensor::create(input->data(), _new_shape);
}

std::vector<std::shared_ptr<Tensor>> ReshapeFunc::_backward(const std::shared_ptr<Tensor>& grad_output) {
    // The gradient for a reshape is just the incoming gradient reshaped back to the original input's shape.
    auto original_shape = _saved_inputs[0]->shape();
    return { grad_output->reshape(original_shape) };
}