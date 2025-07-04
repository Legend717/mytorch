#pragma once

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <random>
#include <stdexcept>

class Function; // 前向声明

// 设备类型枚举
enum class Device {
    CPU,
    CUDA
};

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    // --- 感知设备的构造/析构函数 ---
    Tensor(std::vector<size_t> shape, bool requires_grad = false, Device device = Device::CPU);
    ~Tensor();

    // 禁止拷贝构造和赋值
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // --- 感知设备的工厂方法 ---
    static std::shared_ptr<Tensor> create(const std::vector<float>& data, std::vector<size_t> shape, bool requires_grad = false);
    static std::shared_ptr<Tensor> randn(const std::vector<size_t>& shape, bool requires_grad = false, Device device = Device::CPU);
    static std::shared_ptr<Tensor> ones(const std::vector<size_t>& shape, bool requires_grad = false, Device device = Device::CPU);
    static std::shared_ptr<Tensor> zeros(const std::vector<size_t>& shape, bool requires_grad = false, Device device = Device::CPU);

    // --- 设备管理 ---
    Device device() const { return _device; }
    std::shared_ptr<Tensor> to(Device device);

    // --- 数据访问 ---
    const std::vector<size_t>& shape() const { return _shape; }
    size_t size() const;
    std::vector<float> data_cpu() const; // 将数据从任何设备拷贝到CPU并返回
    float item() const;
    const void* data_ptr() const { return _data; } // 获取通用数据指针(const)
    void* mutable_data_ptr() { return _data; }   // 获取通用数据指针(可变)

    // --- 自动微分 ---
    bool requires_grad() const { return _requires_grad; }
    void set_requires_grad(bool req) { _requires_grad = req; }
    std::shared_ptr<Tensor> grad() { return _grad; }
    std::shared_ptr<Function> ctx() { return _ctx; }
    void set_grad(std::shared_ptr<Tensor> grad);
    void set_ctx(std::shared_ptr<Function> ctx) { _ctx = ctx; }

    // --- 核心功能与运算符 (接口不变) ---
    void backward();
    std::shared_ptr<Tensor> transpose() const;
    std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& other);
    std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& other);
    std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& other);
    std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor>& other);
    std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& other);
    std::shared_ptr<Tensor> sum();
    std::shared_ptr<Tensor> relu();
    std::shared_ptr<Tensor> reshape(const std::vector<size_t>& new_shape);

private:
    void compute_strides();
    void allocate_data();

    void* _data;
    Device _device;

    std::vector<size_t> _shape;
    std::vector<size_t> _strides;
    bool _requires_grad;
    std::shared_ptr<Tensor> _grad;
    std::shared_ptr<Function> _ctx;

    friend class Function;
};

// 自由函数运算符
std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);