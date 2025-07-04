#include "core/tensor.h"
#include "core/function.h"
#include <numeric>
#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>

// --- Device-independent constructor and helpers ---

Tensor::Tensor(std::vector<size_t> shape, bool requires_grad, Device device)
    : _shape(shape), _requires_grad(requires_grad), _device(device), _grad(nullptr), _ctx(nullptr), _data(nullptr) {
    // The actual memory allocation is handled by allocate_data() which is
    // defined in tensor.cu to be device-aware.
    allocate_data();
}


// ✨ FIX: The destructor is now only defined in tensor.cu to avoid multiple definitions.
// Tensor::~Tensor() { ... } // REMOVED


// ✨ FIX: Add the missing definition for the Tensor::create factory method.
std::shared_ptr<Tensor> Tensor::create(const std::vector<float>& data, std::vector<size_t> shape, bool requires_grad) {
    size_t total_size = 1;
    for(const auto& dim : shape) total_size *= dim;
    if (data.size() != total_size) {
        throw std::runtime_error("Data size does not match shape for Tensor::create");
    }

    // Create a tensor on the CPU first
    auto t = std::make_shared<Tensor>(shape, requires_grad, Device::CPU);
    // Copy the data into the new tensor's data structure
    *static_cast<std::vector<float>*>(t->mutable_data_ptr()) = data;
    return t;
}

void Tensor::compute_strides() {
    _strides.resize(_shape.size());
    size_t stride = 1;
    for (int i = _shape.size() - 1; i >= 0; --i) {
        _strides[i] = stride;
        stride *= _shape[i];
    }
}

size_t Tensor::size() const {
    if (_shape.empty()) return 0;
    return std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<size_t>());
}

void Tensor::set_grad(std::shared_ptr<Tensor> grad) {
    if (grad && this->device() != grad->device()) {
        throw std::runtime_error("Tensor and its gradient must be on the same device.");
    }
    _grad = grad;
}

// --- Backward pass (device-independent core logic) ---
void Tensor::backward() {
    if (!_requires_grad) {
        std::cerr << "Warning: called backward() on a tensor that does not require grad." << std::endl;
        return;
    }

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

    this->set_grad(Tensor::ones(this->shape(), false, this->device()));

    std::reverse(topo_order.begin(), topo_order.end());

    for (auto& t : topo_order) {
        if (t->ctx()) {
            if (t->grad() == nullptr) continue;
            auto grads = t->ctx()->backward(t->grad());
            auto& inputs = t->ctx()->_saved_inputs;

            for (size_t i = 0; i < inputs.size(); ++i) {
                if (inputs[i]->requires_grad()) {
                    if (grads[i]) {
                        if (inputs[i]->_grad == nullptr) {
                            inputs[i]->set_grad(grads[i]);
                        } else {
                            auto new_grad = inputs[i]->grad()->add(grads[i]);
                            inputs[i]->set_grad(new_grad);
                        }
                    }
                }
            }
        }
    }
}

std::shared_ptr<Tensor> Tensor::reshape(const std::vector<size_t>& new_shape) {
    size_t old_size = this->size();
    size_t new_size = 1;
    for(const auto& dim : new_shape) new_size *= dim;
    if (old_size != new_size) {
        throw std::runtime_error("Reshape size mismatch.");
    }
    auto func = std::make_shared<ReshapeFunc>(new_shape);
    return func->apply({shared_from_this()});
}

// --- Operators (device-independent wrappers around Functions) ---
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

// --- Free function operators ---
std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return a->add(b); }
std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return a->sub(b); }
std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return a->mul(b); }