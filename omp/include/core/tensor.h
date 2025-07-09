#pragma once

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include "rand/rand.h"

class Function; // 先声明，用于声明Tensor的friend

class Tensor: public std::enable_shared_from_this<Tensor> {
public:
    // 禁止直接构造，请使用static函数构造
    Tensor(const std::vector<float>& data, std::vector<size_t>shape, bool requires_grad = false);

    // 用来创建Tensor的静态函数
    static std::shared_ptr<Tensor> create(const std::vector<float>& data, std::vector<size_t> shape, bool requres_grad = false);

    // 一些工具函数，用于快捷创建常用的Tensor
    static std::shared_ptr<Tensor> randn(const std::vector<size_t>& shape, bool requires_grad = false);
    static std::shared_ptr<Tensor> ones(const std::vector<size_t>& shape, bool requires_grad = false);
    static std::shared_ptr<Tensor> zeros(const std::vector<size_t>& shape, bool requires_grad = false);

    // 一些属性
    const std::vector<size_t>& shape() const { return _shape; }
    const std::vector<float>& data() const { return *_data; }
    float item() const;
    std::shared_ptr<Tensor> grad() { return _grad; }
    std::shared_ptr<Function> ctx() { return _ctx; }
    std::shared_ptr<std::vector<float>> get_shared_data() {return _data;}
    void set_grad(std::shared_ptr<Tensor> grad) { _grad = grad; }
    bool requires_grad() const { return _requires_grad; }
    void set_ctx(std::shared_ptr<Function> ctx) { _ctx = ctx; }
    bool is_leaf() const { return _ctx == nullptr; }

    // 核心功能
    void backward();
    std::shared_ptr<Tensor> transpose() const;

    // 运算符重载
    std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& other);
    std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& other);
    std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& other);
    std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor>& other);
    std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& other);

    std::shared_ptr<Tensor> sum();
    // std::shared_ptr<Tensor> means();
    std::shared_ptr<Tensor> relu();

    // 一些其他函数
    std::shared_ptr<Tensor> reshape(const std::vector<size_t>& new_shape);
    std::shared_ptr<Tensor> slice (size_t start, size_t end) const;
private:
    // 存储数据
    std::shared_ptr<std::vector<float>> _data;
    std::vector<size_t> _shape;
    std::vector<size_t> _stride;
    bool _requires_grad;
    
    std::shared_ptr<Tensor> _grad;
    std::shared_ptr<Function> _ctx;

    void compute_stride();

    friend class Function;
};