#include "core/function.h"
#include "core/tensor.h"
#include <stdexcept>
#include <numeric>
#include <omp.h>
#include <algorithm>

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

// 加法
std::vector<std::shared_ptr<Tensor>> Function::backward(const std::shared_ptr<Tensor>& grad_output) {
    return _backward(grad_output);
}

std::shared_ptr<Tensor> Add::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
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
    // 加法操作的梯度是 1，因此 grad_a 和 grad_b 直接等于 grad_output
    auto a = _saved_inputs[0];
    auto b = _saved_inputs[1];
    auto grad_a = grad_output;
    auto grad_b = grad_output;

    // 处理维度不同的情况，此时 grad_b 需要进行广播
    if (a->shape() != b->shape()) {
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

        // for(size_t i = 0; i < batch_size; ++i) {
        //     for(size_t j = 0; j < features; ++j) {
        //         sum_grad_data[j] += grad_output->data_cpu()[i * features + j];
        //     }
        // }
        // grad_b = Tensor::create(sum_grad_data, b->shape());
    }
    return {grad_a, grad_b};
}

// 减法，暂不考虑广播情况
std::shared_ptr<Tensor> Sub::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
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
    // 减法的b梯度是-1，因此 grad_b 等于 -grad_output
    auto neg_grad = grad_output->mul(Tensor::create({-1.0f}, {1}))->to(grad_output->device());
    return {grad_output, neg_grad};
}

// 乘法（逐位）
std::shared_ptr<Tensor> Mul::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
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
    // z = a * b, dl/da = dl/dz * b, dl/db = dl/dz * a
    auto a = _saved_inputs[0];
    auto b = _saved_inputs[1];
    return {grad_output->mul(b), grad_output->mul(a)};
}

// 矩阵乘法
std::shared_ptr<Tensor> MatMul::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs[0]->device() == Device::CUDA) {
        return matmul_forward_cuda(inputs[0], inputs[1]);
    }
    const auto& a = inputs[0];
    const auto& b = inputs[1];
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
                result_data[i * N + j] += a->data_cpu()[i * K + k] * b->data_cpu()[k * N + j];
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

// 求和
std::shared_ptr<Tensor> Sum::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
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
    auto original_shape = _saved_inputs[0]->shape();
    size_t total_size = 1;
    for(auto dim : original_shape) total_size *= dim;
    // 对于求和，梯度是全 1，因此 grad_output 直接作为输出
    std::vector<float> grad_data(total_size, grad_output->data_cpu()[0]);
    return {Tensor::create(grad_data, original_shape)};
}

// ReLU
std::shared_ptr<Tensor> ReLUFunc::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
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
    const auto& x = _saved_inputs[0]->data_cpu();
    auto x_device = _saved_inputs[0]->device();
    std::vector<float> mask_data(x.size());
    // ReLU的梯度是0或1
    #pragma omp parallel for
    for(size_t i = 0; i < x.size(); ++i) {
        mask_data[i] = x[i] > 0 ? 1.0f : 0.0f;
    }
    auto mask = Tensor::create(mask_data, _saved_inputs[0]->shape())->to(x_device);
    return {grad_output->mul(mask)};
}


// Reshape
std::shared_ptr<Tensor> ReshapeFunc::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    auto input = inputs[0];
    if (input->device() == Device::CUDA) {
        return reshape_forward_cuda(input, _new_shape);
    }
    return Tensor::create(input->data_cpu(), _new_shape);
}

std::vector<std::shared_ptr<Tensor>> ReshapeFunc::_backward(const std::shared_ptr<Tensor>& grad_output) {
    auto original_shape = _saved_inputs[0]->shape();
    return { grad_output->reshape(original_shape) };
}