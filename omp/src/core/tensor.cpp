#include "core/tensor.h"
#include "core/function.h"
#include "rand/rand.h"
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <set>

#include <vector>
#include <memory>
#include <string>
#include <functional>

// 构造函数
Tensor::Tensor(const std::vector<float>& data, std::vector<size_t>shape, bool requires_grad) : _data(std::make_shared<std::vector<float>>(data)), _shape(shape), _requires_grad(requires_grad), _grad(nullptr), _ctx(nullptr) {
    size_t tot_size = 1;
    for (size_t dim : shape) tot_size *= dim;
    if (tot_size != _data->size()) {
        throw std::runtime_error("Shape and data size do not match.");
    }
    compute_stride();
}

// 计算步幅
void Tensor::compute_stride() {
    _stride.resize(_shape.size());
    size_t stride = 1;
    for (int i = _shape.size() - 1; i >= 0; i--) {
        _stride[i] = stride;
        stride *= _shape[i];
    }
}

// 用来创建Tensor的静态函数
std::shared_ptr<Tensor> Tensor::create(const std::vector<float>& data, std::vector<size_t> shape, bool requires_grad) {
    // 使用嵌套辅助类来实现make_shared
    struct MakeSharedEnabler : public Tensor {
        MakeSharedEnabler(const std::vector<float>& data, std::vector<size_t> shape, bool requires_grad)
            : Tensor(data, shape, requires_grad) {}
    };
    return std::make_shared<MakeSharedEnabler>(data, shape, requires_grad);
}

// 一些工具函数，用于快捷创建常用的Tensor
std::shared_ptr<Tensor> Tensor::randn(const std::vector<size_t>& shape, bool requires_grad) {
    size_t tot_size = 1;
    for(size_t dim : shape) tot_size *= dim;
    std::vector<float> data(tot_size);

    std::normal_distribution<> d(0, 1);
        
    // 随机数函数并不是线程安全的，在此暂时不进行并行
    for(size_t i = 0; i < tot_size; i++) {
        data[i] = d(MyRand::global_rand_generater);
    }
    return create(data, shape, requires_grad);
}

std::shared_ptr<Tensor> Tensor::ones(const std::vector<size_t>& shape, bool requires_grad) {
    size_t tot_size = 1;
    for(size_t dim : shape) tot_size *= dim;
    return create(std::vector<float>(tot_size, 1.0f), shape, requires_grad);
}

std::shared_ptr<Tensor> Tensor::zeros(const std::vector<size_t>& shape, bool requires_grad) {
    size_t total_size = 1;
    for(size_t dim : shape) total_size *= dim;
    return create(std::vector<float>(total_size, 0.0f), shape, requires_grad);
}

float Tensor::item() const {
    if (_data->size() != 1) {
        throw std::runtime_error("item() 只能用于标量Tensor");
    }
    return (*_data)[0];
}

// 核心功能
void Tensor::backward() {
    if (!_requires_grad) {
        std::cerr << "Warning: 在一个无梯度的Tensor上调用backward()函数\n";
        return;
    }

    // 拓扑排序
    std::vector<std::shared_ptr<Tensor>> topo_order;
    std::set<Tensor*> visited;
    // 使用辅助函数，方便递归调用，构建计算图
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

    // 反向传播
    this->_grad = ones(this->shape()); // 初始化梯度为对自身的导数，即为1
    std::reverse(topo_order.begin(), topo_order.end());
    for (auto& t : topo_order) {
        if (t->ctx()) {
            // 跳过没有梯度的参数
            if (t->grad() == nullptr) continue;

            auto grads = t->ctx()->backward(t->grad());
            auto& inputs = t->ctx()->_saved_inputs;

            for (size_t i = 0; i < inputs.size(); i++) {
                if (inputs[i]->requires_grad()) {
                    if (inputs[i]->grad() == nullptr) {
                        inputs[i]->_grad = grads[i];
                    }
                    else {
                        auto old_grad_data = inputs[i]->_grad->get_shared_data();
                        const auto& new_grad_data = grads[i]->data();
                        
                        if (old_grad_data->size() != new_grad_data.size()) {
                            throw std::runtime_error("反向传播时，梯度大小不匹配");
                        }

                        for (size_t j = 0; j < old_grad_data->size(); j++) {
                            (*old_grad_data)[j] += new_grad_data[j];
                        }
                    }
                }
            }
        }
    }

}

// 转置
std::shared_ptr<Tensor> Tensor::transpose() const {
    if (_shape.size() != 2) throw std::runtime_error("转置操作仅支持二维Tensor");
    std::vector<size_t> new_shape = {_shape[1], _shape[0]};
    std::vector<float> new_data(new_shape[0] * new_shape[1]);
    for(size_t i = 0; i < new_shape[0]; i++) {
        for(size_t j = 0; j < new_shape[1]; j++) {
            new_data[i * new_shape[1] + j] = (*_data)[j * _shape[1] + i];
        }
    }
    return create(new_data, new_shape, _requires_grad);
}

std::shared_ptr<Tensor> Tensor::reshape(const std::vector<size_t>& new_shape) {
    size_t new_tot_size = 1;
    for(size_t dim : new_shape) new_tot_size *= dim;
    if (new_tot_size != this->data().size()){
        throw std::runtime_error("Reshape 大小不匹配！");
    }
    
    auto func = std::make_shared<ReshapeFunc>(new_shape);
    return func->apply({shared_from_this()});
}

std::shared_ptr<Tensor> Tensor::slice(size_t start, size_t end) const{
    /*在Tensor中的第一个维度切出从start到end的部分*/
    if(_shape.size() < 1 || start >= end || end > _shape[0]){
        throw std::runtime_error("切片参数无效");
    }
    size_t feature_size = 1;
    for(size_t  i = 1; i < _shape.size(); ++i){
        feature_size *= _shape[i];
    }
    auto start_sl = _data->begin() + start*feature_size;
    auto end_sl = _data->begin() + end*feature_size;
    std::vector<float> sliced_data(start_sl, end_sl);
    std::vector<size_t> new_shape = _shape;
    new_shape[0] = end -start;
    
    return create(sliced_data, new_shape, _requires_grad);
}

// 运算符重载
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
    if(other->data().size() != 1) throw std::runtime_error("除法仅支持标量");
    auto inverse = create({1.0f/ other->data()[0]}, {1});
    return this->mul(inverse);
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
