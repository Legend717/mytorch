#include "core/function.h"
#include "core/tensor.h"
#include <stdexcept>
#include <numeric>
#include <algorithm>

// --- Forward-declare the CUDA implementations from function_cuda.cu ---
std::shared_ptr<Tensor> add_forward_cuda(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> mul_forward_cuda(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> matmul_forward_cuda(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> relu_forward_cuda(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> reshape_forward_cuda(const std::shared_ptr<Tensor>& input, const std::vector<size_t>& new_shape);
std::shared_ptr<Tensor> conv2d_forward_cuda(const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& weight, size_t stride, size_t padding);
std::vector<std::shared_ptr<Tensor>> conv2d_backward_cuda(
    const std::shared_ptr<Tensor>& grad_output,
    const std::shared_ptr<Tensor>& input,
    const std::shared_ptr<Tensor>& weight,
    size_t stride, size_t padding
);
std::shared_ptr<Tensor> maxpool2d_forward_cuda(const std::shared_ptr<Tensor>& input, size_t kernel_size, size_t stride, std::vector<size_t>& max_indices);
std::shared_ptr<Tensor> maxpool2d_backward_cuda(const std::shared_ptr<Tensor>& grad_output, const std::shared_ptr<Tensor>& input, const std::vector<size_t>& max_indices);


// --- Base class apply/backward ---
std::shared_ptr<Tensor> Function::apply(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    _saved_inputs = inputs;
    auto output = _forward(inputs);

    bool requires_grad = false;
    for (const auto& input : inputs) {
        if (input && input->requires_grad()) {
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


// --- Add ---
std::shared_ptr<Tensor> Add::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs[0]->device() == Device::CUDA) {
        return add_forward_cuda(inputs[0], inputs[1]);
    }
    const auto& a_data = inputs[0]->data_cpu();
    const auto& b_data = inputs[1]->data_cpu();
    if (a_data.size() != b_data.size()) throw std::runtime_error("Add shape mismatch for CPU impl.");
    std::vector<float> result_data(a_data.size());
    for (size_t i = 0; i < a_data.size(); ++i) {
        result_data[i] = a_data[i] + b_data[i];
    }
    return Tensor::create(result_data, inputs[0]->shape());
}

std::vector<std::shared_ptr<Tensor>> Add::_backward(const std::shared_ptr<Tensor>& grad_output) {
    return {grad_output, grad_output};
}


// --- Sub ---
std::shared_ptr<Tensor> Sub::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    // For now, we assume Sub is related to Add. A proper CUDA kernel would be needed for GPU.
    if (inputs[0]->device() == Device::CUDA) {
        // Create a tensor of -1s and multiply
        auto neg_b = inputs[1]->mul(Tensor::create({-1.0f}, {1}, false)->to(Device::CUDA));
        return add_forward_cuda(inputs[0], neg_b);
    }
    const auto& a_data = inputs[0]->data_cpu();
    const auto& b_data = inputs[1]->data_cpu();
    if (a_data.size() != b_data.size()) throw std::runtime_error("Sub shape mismatch for CPU impl.");
    std::vector<float> result_data(a_data.size());
    for (size_t i = 0; i < a_data.size(); ++i) {
        result_data[i] = a_data[i] - b_data[i];
    }
    return Tensor::create(result_data, inputs[0]->shape());
}

std::vector<std::shared_ptr<Tensor>> Sub::_backward(const std::shared_ptr<Tensor>& grad_output) {
    auto neg_grad = grad_output->mul(Tensor::create({-1.0f}, {1}, false)->to(grad_output->device()));
    return {grad_output, neg_grad};
}


// --- Mul ---
std::shared_ptr<Tensor> Mul::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs[0]->device() == Device::CUDA) {
        return mul_forward_cuda(inputs[0], inputs[1]);
    }
    const auto& a_data = inputs[0]->data_cpu();
    const auto& b_data = inputs[1]->data_cpu();
    if (a_data.size() != b_data.size()) throw std::runtime_error("Mul shape mismatch for CPU impl.");
    std::vector<float> result_data(a_data.size());
    for (size_t i = 0; i < a_data.size(); ++i) {
        result_data[i] = a_data[i] * b_data[i];
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
    if (inputs[0]->device() == Device::CUDA) {
        return matmul_forward_cuda(inputs[0], inputs[1]);
    }
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    if (a->shape().size() != 2 || b->shape().size() != 2 || a->shape()[1] != b->shape()[0]) {
        throw std::runtime_error("MatMul shape mismatch.");
    }
    size_t M = a->shape()[0];
    size_t K = a->shape()[1];
    size_t N = b->shape()[1];
    std::vector<float> result_data(M * N, 0.0f);
    const auto& a_data = a->data_cpu();
    const auto& b_data = b->data_cpu();

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t k = 0; k < K; ++k) {
                result_data[i * N + j] += a_data[i * K + k] * b_data[k * N + j];
            }
        }
    }
    return Tensor::create(result_data, {M, N});
}

std::vector<std::shared_ptr<Tensor>> MatMul::_backward(const std::shared_ptr<Tensor>& grad_output) {
    auto a = _saved_inputs[0];
    auto b = _saved_inputs[1];
    auto grad_a = grad_output->matmul(b->transpose());
    auto grad_b = a->transpose()->matmul(grad_output);
    return {grad_a, grad_b};
}


// --- Sum ---
std::shared_ptr<Tensor> Sum::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    const auto& x = inputs[0];
    if (x->device() == Device::CUDA) {
        // Placeholder for a proper CUDA reduction kernel
        float sum_val = 0.0f;
        auto x_cpu = x->data_cpu();
        for(const auto& val : x_cpu) sum_val += val;
        return Tensor::create({sum_val}, {1}, x->requires_grad())->to(Device::CUDA);
    }
    float sum_val = 0.0f;
    const auto& data = x->data_cpu();
    for(const auto& val : data) sum_val += val;
    return Tensor::create({sum_val}, {1}, x->requires_grad());
}

std::vector<std::shared_ptr<Tensor>> Sum::_backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = _saved_inputs[0];
    auto ones = Tensor::ones(x->shape(), false, x->device());
    // This assumes broadcasting of the scalar grad_output
    return {ones->mul(grad_output)};
}


// --- ReLUFunc ---
std::shared_ptr<Tensor> ReLUFunc::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs[0]->device() == Device::CUDA) {
        return relu_forward_cuda(inputs[0]);
    }
    const auto& x = inputs[0]->data_cpu();
    std::vector<float> result_data(x.size());
    for(size_t i=0; i<x.size(); ++i) {
        result_data[i] = std::max(0.0f, x[i]);
    }
    return Tensor::create(result_data, inputs[0]->shape());
}

std::vector<std::shared_ptr<Tensor>> ReLUFunc::_backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = _saved_inputs[0];
    auto x_cpu = x->data_cpu();
    std::vector<float> mask_data(x_cpu.size());
    for(size_t i=0; i<x_cpu.size(); ++i) {
        mask_data[i] = x_cpu[i] > 0 ? 1.0f : 0.0f;
    }
    auto mask = Tensor::create(mask_data, x->shape())->to(x->device());
    return {grad_output->mul(mask)};
}


// --- ReshapeFunc ---
std::shared_ptr<Tensor> ReshapeFunc::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    auto input = inputs[0];
    if (input->device() == Device::CUDA) {
        return reshape_forward_cuda(input, _new_shape);
    }
    return Tensor::create(input->data_cpu(), _new_shape, false);
}

std::vector<std::shared_ptr<Tensor>> ReshapeFunc::_backward(const std::shared_ptr<Tensor>& grad_output) {
    auto original_shape = _saved_inputs[0]->shape();
    return { grad_output->reshape(original_shape) };
}


// --- Conv2DFunc ---
std::shared_ptr<Tensor> Conv2DFunc::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    auto& input = inputs[0];
    auto& weight = inputs[1];

    if (input->device() == Device::CUDA) {
        return conv2d_forward_cuda(input, weight, _stride, _padding);
    }
    throw std::runtime_error("Conv2D CPU forward pass is not implemented yet.");
}

// ✨ 2. 将此函数内容替换为以下实现
std::vector<std::shared_ptr<Tensor>> Conv2DFunc::_backward(const std::shared_ptr<Tensor>& grad_output) {
    auto& input = _saved_inputs[0];
    auto& weight = _saved_inputs[1];
    // _saved_inputs[2] 应该是偏置，但它的梯度是独立计算的

    if (input->device() == Device::CUDA) {
        auto grads = conv2d_backward_cuda(grad_output, input, weight, _stride, _padding);
        // conv2d_backward_cuda 返回 {grad_input, grad_weight, grad_bias}
        // 我们需要返回 {grad_input, grad_weight} 因为偏置的梯度会单独处理
        // 或者，我们需要调整以匹配 Function::_backward 的期望
        // 假设 backward 期望 {grad_input, grad_weight, grad_bias}
        // 如果AddFunc在forward中使用了bias, _saved_inputs会有3个元素
        // 但Conv2D::forward中，bias是分开add的，所以ctx只保存了input和weight

        // 这里的逻辑需要小心：Conv2D的forward是 conv(x,w) + b
        // 所以计算图是 Add -> Conv2D
        // Add的backward会产生grad_output_for_conv 和 grad_bias
        // Conv2D的backward接收grad_output_for_conv, 产生grad_input和grad_weight
        // 这里的grad_output实际上是Add的输出梯度

        // 从我们的实现来看，conv2d_backward_cuda已经计算了所有梯度
        // 但在 Function::backward 循环中，梯度是累加的
        // Conv2D::forward 返回 output->add(_bias), 这意味着Add是父节点
        // 所以这里的 grad_output 是来自Add节点的，它等于Add输出的梯度
        // Add的backward会计算对bias的梯度
        // 因此，Conv2DFunc::_backward只需要返回对input和weight的梯度

        auto grad_input = grads[0];
        auto grad_weight = grads[1];
        // 我们不需要返回grad_bias，因为它是由Add操作的上下文处理的
        return {grad_input, grad_weight};
    }
    throw std::runtime_error("Conv2D CPU backward pass is not implemented yet.");
}

// --- MaxPool2DFunc ---
// ✨ MODIFIED: This now dispatches to the CUDA implementation
std::shared_ptr<Tensor> MaxPool2DFunc::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    auto& input = inputs[0];
    if (input->device() == Device::CUDA) {
        return maxpool2d_forward_cuda(input, _kernel_size, _stride, _max_indices);
    }
    throw std::runtime_error("MaxPool2D CPU forward pass is not implemented yet.");
}

std::vector<std::shared_ptr<Tensor>> MaxPool2DFunc::_backward(const std::shared_ptr<Tensor>& grad_output) {
    auto& input = _saved_inputs[0];
    if (input->device() == Device::CUDA) {
        // 调用新实现的CUDA反向传播函数
        auto grad_input = maxpool2d_backward_cuda(grad_output, input, _max_indices);
        return {grad_input};
    }
    // 保留针对CPU的未实现错误
    throw std::runtime_error("MaxPool2D CPU backward pass is not implemented yet.");
}