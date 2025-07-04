#pragma once

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <random>

class Function; // 前向声明

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    // 禁止直接构造，强制使用工厂函数以安全地使用 shared_from_this
    Tensor(const std::vector<float>& data, std::vector<size_t> shape, bool requires_grad = false);

    // --- 工厂方法 ---
    static std::shared_ptr<Tensor> create(const std::vector<float>& data, std::vector<size_t> shape, bool requires_grad = false);
    static std::shared_ptr<Tensor> randn(const std::vector<size_t>& shape, bool requires_grad = false);
    static std::shared_ptr<Tensor> ones(const std::vector<size_t>& shape, bool requires_grad = false);
    static std::shared_ptr<Tensor> zeros(const std::vector<size_t>& shape, bool requires_grad = false);

    // --- 访问器 ---
    const std::vector<size_t>& shape() const { return _shape; }
    const std::vector<float>& data() const { return *_data; }
    std::shared_ptr<std::vector<float>> get_shared_data() { return _data; }
    float item() const; // 获取单元素张量的值

    // --- 自动微分相关 ---
    bool requires_grad() const { return _requires_grad; }
    std::shared_ptr<Tensor> grad() { return _grad; }
    std::shared_ptr<Function> ctx() { return _ctx; }
    void set_grad(std::shared_ptr<Tensor> grad) { _grad = grad; }
    void set_ctx(std::shared_ptr<Function> ctx) { _ctx = ctx; }

    // --- 核心功能 ---
    void backward();
    std::shared_ptr<Tensor> transpose() const;

    // --- 运算符重载 ---
    std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& other);
    std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& other);
    std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& other);
    std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor>& other);
    std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& other);
    std::shared_ptr<Tensor> sum();
    std::shared_ptr<Tensor> relu();

    // --- CNN相关 ---
    std::shared_ptr<Tensor> reshape(const std::vector<size_t>& new_shape);
private:
    void compute_strides();

    std::shared_ptr<std::vector<float>> _data;
    std::vector<size_t> _shape;
    std::vector<size_t> _strides;
    bool _requires_grad;

    std::shared_ptr<Tensor> _grad;
    std::shared_ptr<Function> _ctx;

    friend class Function;
};

// --- 方便的自由函数运算符 ---
std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);