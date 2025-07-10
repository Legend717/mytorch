#include "core/tensor.h"
#include "core/function.h"
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <set>

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <random>
#include <stdexcept>
// 构造函数
Tensor::Tensor(std::vector<size_t> shape, bool requires_grad, Device device)
    : _shape(shape), _requires_grad(requires_grad), _device(device), _grad(nullptr), _ctx(nullptr), _data(nullptr) {
    allocate_data();
}

std::shared_ptr<Tensor> Tensor::create(const std::vector<float>& data, std::vector<size_t> shape, bool requires_grad) {
    size_t total_size = 1;
    for(const auto& dim : shape) total_size *= dim;
    if (data.size() != total_size) {
        throw std::runtime_error("创建Tensor时，数据大小与形状不匹配");
    }

    auto t = std::make_shared<Tensor>(shape, requires_grad, Device::CPU);
    *static_cast<std::vector<float>*>(t->mutable_data_ptr()) = data;
    return t;
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

// 计算需要分配空间大小
size_t Tensor::size() const {
    if (_shape.empty()) return 0;
    return std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<size_t>());
}

// 设置梯度
void Tensor::set_grad(std::shared_ptr<Tensor> grad) {
    if (grad && this->device() != grad->device()) {
        throw std::runtime_error("Tensor和Grad所在的设备不同");
    }
    _grad = grad;
}

// 用来创建Tensor的静态函数
// std::shared_ptr<Tensor> Tensor::create(const std::vector<float>& data, std::vector<size_t> shape, bool requires_grad) {
//     // 使用嵌套辅助类来实现make_shared
//     struct MakeSharedEnabler : public Tensor {
//         MakeSharedEnabler(const std::vector<float>& data, std::vector<size_t> shape, bool requires_grad)
//             : Tensor(data, shape, requires_grad) {}
//     };
//     return std::make_shared<MakeSharedEnabler>(data, shape, requires_grad);
// }

// 弃用，升级成支持gpu版本
// // 一些工具函数，用于快捷创建常用的Tensor
// std::shared_ptr<Tensor> Tensor::randn(const std::vector<size_t>& shape, bool requires_grad) {
//     size_t tot_size = 1;
//     for(size_t dim : shape) tot_size *= dim;
//     std::vector<float> data(tot_size);

//     // 随机数函数并不是线程安全的，在此暂时不进行并行
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::normal_distribution<> d(0, 1);

//     for(size_t i = 0; i < tot_size; i++) {
//         data[i] = d(gen);
//     }
//     return create(data, shape, requires_grad);
// }

// std::shared_ptr<Tensor> Tensor::ones(const std::vector<size_t>& shape, bool requires_grad) {
//     size_t tot_size = 1;
//     for(size_t dim : shape) tot_size *= dim;
//     return create(std::vector<float>(tot_size, 1.0f), shape, requires_grad);
// }

// std::shared_ptr<Tensor> Tensor::zeros(const std::vector<size_t>& shape, bool requires_grad) {
//     size_t total_size = 1;
//     for(size_t dim : shape) total_size *= dim;
//     return create(std::vector<float>(total_size, 0.0f), shape, requires_grad);
// }

// 弃用，升级成支持gpu版本
// float Tensor::item() const {
//     if (_data->size() != 1) {
//         throw std::runtime_error("item() 只能用于标量Tensor");
//     }
//     return (*_data)[0];
// }

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
    this->set_grad(Tensor::ones(this->shape(), false, this->device())); // 初始化梯度为对自身的导数，即为1
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

// 弃用
// // 转置
// std::shared_ptr<Tensor> Tensor::transpose() const {
//     if (_shape.size() != 2) throw std::runtime_error("转置操作仅支持二维Tensor");
//     std::vector<size_t> new_shape = {_shape[1], _shape[0]};
//     std::vector<float> new_data(new_shape[0] * new_shape[1]);
//     for(size_t i = 0; i < new_shape[0]; i++) {
//         for(size_t j = 0; j < new_shape[1]; j++) {
//             new_data[i * new_shape[1] + j] = (*_data)[j * _shape[1] + i];
//         }
//     }
//     return create(new_data, new_shape, _requires_grad);
// }

std::shared_ptr<Tensor> Tensor::reshape(const std::vector<size_t>& new_shape) {
    size_t new_tot_size = 1;
    for(size_t dim : new_shape) new_tot_size *= dim;
    if (new_tot_size != this->size()){
        throw std::runtime_error("Reshape 大小不匹配！");
    }
    
    auto func = std::make_shared<ReshapeFunc>(new_shape);
    return func->apply({shared_from_this()});
}

// 弃用
// std::shared_ptr<Tensor> Tensor::slice(size_t start, size_t end) const{
//     /*在Tensor中的第一个维度切出从start到end的部分*/
//     if(_shape.size() < 1 || start >= end || end > _shape[0]){
//         throw std::runtime_error("切片参数无效");
//     }
//     size_t feature_size = 1;
//     for(size_t  i = 1; i < _shape.size(); ++i){
//         feature_size *= _shape[i];
//     }
//     auto start_sl = _data->begin() + start*feature_size;
//     auto end_sl = _data->begin() + end*feature_size;
//     std::vector<float> sliced_data(start_sl, end_sl);
//     std::vector<size_t> new_shape = _shape;
//     new_shape[0] = end -start;
    
//     return create(sliced_data, new_shape, _requires_grad);
// }

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

// 方便使用运算符
std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return a->add(b); }
std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return a->sub(b); }
std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return a->mul(b); }

