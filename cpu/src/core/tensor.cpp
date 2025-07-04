#include "core/tensor.h"
#include "core/function.h"
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <set>

Tensor::Tensor(const std::vector<float>& data, std::vector<size_t> shape, bool requires_grad)
    : _data(std::make_shared<std::vector<float>>(data)), _shape(shape), _requires_grad(requires_grad), _grad(nullptr), _ctx(nullptr) {
    size_t total_size = 1;
    for(size_t dim : _shape) total_size *= dim;
    if (total_size != _data->size()) {
        throw std::runtime_error("Shape and data size mismatch.");
    }
    compute_strides();
}

void Tensor::compute_strides() {
    _strides.resize(_shape.size());
    size_t stride = 1;
    for (int i = _shape.size() - 1; i >= 0; --i) {
        _strides[i] = stride;
        stride *= _shape[i];
    }
}

std::shared_ptr<Tensor> Tensor::create(const std::vector<float>& data, std::vector<size_t> shape, bool requires_grad) {
    // This allows make_shared to call the private constructor
    struct MakeSharedEnabler : public Tensor {
        MakeSharedEnabler(const std::vector<float>& data, std::vector<size_t> shape, bool requires_grad)
            : Tensor(data, shape, requires_grad) {}
    };
    return std::make_shared<MakeSharedEnabler>(data, shape, requires_grad);
}

std::shared_ptr<Tensor> Tensor::randn(const std::vector<size_t>& shape, bool requires_grad) {
    size_t total_size = 1;
    for(size_t dim : shape) total_size *= dim;
    std::vector<float> data(total_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);

    for(size_t i = 0; i < total_size; ++i) {
        data[i] = d(gen);
    }
    return create(data, shape, requires_grad);
}

std::shared_ptr<Tensor> Tensor::ones(const std::vector<size_t>& shape, bool requires_grad) {
    size_t total_size = 1;
    for(size_t dim : shape) total_size *= dim;
    return create(std::vector<float>(total_size, 1.0f), shape, requires_grad);
}

std::shared_ptr<Tensor> Tensor::zeros(const std::vector<size_t>& shape, bool requires_grad) {
    size_t total_size = 1;
    for(size_t dim : shape) total_size *= dim;
    return create(std::vector<float>(total_size, 0.0f), shape, requires_grad);
}

float Tensor::item() const {
    if (_data->size() != 1) {
        throw std::runtime_error("item() can only be called on tensors with a single element.");
    }
    return (*_data)[0];
}

void Tensor::backward() {
    if (!_requires_grad) {
        std::cerr << "Warning: called backward() on a tensor that does not require grad." << std::endl;
        return;
    }
    
    // 拓扑排序
    std::vector<std::shared_ptr<Tensor>> topo_order;
    std::set<Tensor*> visited;
    std::function<void(Tensor*)> build_topo = 
        [&](Tensor* t) {
        if (visited.find(t) == visited.end()) {
            visited.insert(t);
            if (t->ctx()) {
                for (auto& input : t->ctx()->_saved_inputs) {
                    build_topo(input.get());
                }
            }
            topo_order.push_back(t->shared_from_this());
        }
    };

    build_topo(this);
    
    // 初始化梯度
    this->_grad = ones(this->shape()); // grad should have same shape as tensor

    // 反向传播梯度
    std::reverse(topo_order.begin(), topo_order.end());

    for (auto& t : topo_order) {
        if (t->ctx()) {
            if (t->grad() == nullptr) continue; // Skip if no gradient flows to this tensor

            auto grads = t->ctx()->backward(t->grad());
            auto& inputs = t->ctx()->_saved_inputs;

            for (size_t i = 0; i < inputs.size(); ++i) {
                if (inputs[i]->requires_grad()) {
                    if (inputs[i]->_grad == nullptr) {
                        inputs[i]->_grad = grads[i];
                    } else {
                        auto old_grad_data = inputs[i]->_grad->get_shared_data();
                        const auto& new_grad_data = grads[i]->data();
                        
                        if (old_grad_data->size() != new_grad_data.size()) {
                             throw std::runtime_error("Gradient shape mismatch during accumulation.");
                        }

                        for (size_t j = 0; j < old_grad_data->size(); ++j) {
                            (*old_grad_data)[j] += new_grad_data[j];
                        }
                    }
                }
            }
        }
    }
}

std::shared_ptr<Tensor> Tensor::transpose() const {
    if (_shape.size() != 2) throw std::runtime_error("Transpose is only supported for 2D tensors.");
    std::vector<size_t> new_shape = {_shape[1], _shape[0]};
    std::vector<float> new_data(new_shape[0] * new_shape[1]);
    for(size_t i=0; i<new_shape[0]; ++i) {
        for(size_t j=0; j<new_shape[1]; ++j) {
            new_data[i * new_shape[1] + j] = (*_data)[j * _shape[1] + i];
        }
    }
    return create(new_data, new_shape, _requires_grad);
}

std::shared_ptr<Tensor> Tensor::reshape(const std::vector<size_t>& new_shape) {
    size_t new_total_size = 1;
    for(size_t dim : new_shape) new_total_size *= dim;
    if (new_total_size != this->data().size()){
        throw std::runtime_error("Reshape size mismatch.");
    }
    
    // Create the function and apply it, which handles autograd tracking.
    auto func = std::make_shared<ReshapeFunc>(new_shape);
    return func->apply({shared_from_this()});
}

// --- 运算符实现 ---
// The helper template was causing issues. This is a simpler and more direct approach.

std::shared_ptr<Tensor> Tensor::add(const std::shared_ptr<Tensor>& other) { 
    auto func = std::make_shared<Add>();
    return func->apply({shared_from_this(), other});
}

std::shared_ptr<Tensor> Tensor::sub(const std::shared_ptr<Tensor>& other) { 
    auto func = std::make_shared<Sub>();
    return func->apply({shared_from_this(), other});
}

std::shared_ptr<Tensor> Tensor::mul(const std::shared_ptr<Tensor>& other) { 
    auto func = std::make_shared<Mul>();
    return func->apply({shared_from_this(), other});
}

std::shared_ptr<Tensor> Tensor::div(const std::shared_ptr<Tensor>& other) { 
    throw std::runtime_error("Div not implemented"); 
}

std::shared_ptr<Tensor> Tensor::matmul(const std::shared_ptr<Tensor>& other) { 
    auto func = std::make_shared<MatMul>();
    return func->apply({shared_from_this(), other});
}

std::shared_ptr<Tensor> Tensor::sum() { 
    auto func = std::make_shared<Sum>();
    return func->apply({shared_from_this()});
}

std::shared_ptr<Tensor> Tensor::relu() { 
    auto func = std::make_shared<ReLUFunc>();
    return func->apply({shared_from_this()});
}

// --- 方便的自由函数运算符 ---
std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return a->add(b); }
std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return a->sub(b); }
std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return a->mul(b); }