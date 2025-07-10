#pragma once

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <random>

class Function; // 先声明，用于声明Tensor的friend

// 设备类型枚举
enum class Device {
    CPU,
    CUDA
};

class Tensor: public std::enable_shared_from_this<Tensor> {
public:
    // 禁止直接构造，请使用static函数构造
    // Tensor(const std::vector<float>& data, std::vector<size_t>shape, bool requires_grad = false, Device device = Device::CPU);

    Tensor(std::vector<size_t>shape, bool requires_grad = false, Device device = Device::CPU);
    ~Tensor();

    // 禁止拷贝构造和赋值
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // 用来创建Tensor的静态函数
    static std::shared_ptr<Tensor> create(const std::vector<float>& data, std::vector<size_t> shape, bool requres_grad = false);

    // 一些工具函数，用于快捷创建常用的Tensor
    static std::shared_ptr<Tensor> randn(const std::vector<size_t>& shape, bool requires_grad = false, Device device = Device::CPU);
    static std::shared_ptr<Tensor> ones(const std::vector<size_t>& shape, bool requires_grad = false, Device device = Device::CPU);
    static std::shared_ptr<Tensor> zeros(const std::vector<size_t>& shape, bool requires_grad = false, Device device = Device::CPU);

    // --- 设备管理 ---
    Device device() const { return _device; }
    std::shared_ptr<Tensor> to(Device device);

    // 自动微分
    bool requires_grad() const { return _requires_grad; }
    void set_requires_grad(bool req) { _requires_grad = req; }
    std::shared_ptr<Tensor> grad() { return _grad; }
    std::shared_ptr<Function> ctx() { return _ctx; }
    void set_grad(std::shared_ptr<Tensor> grad);
    void set_ctx(std::shared_ptr<Function> ctx) { _ctx = ctx; }

    // 数据访问
    // const std::vector<float>& data() const { return *_data; } // cpu代码，弃用
    // std::shared_ptr<std::vector<float>> get_shared_data() {return _data;} // cpu代码，弃用
    const void* data_ptr() const { return _data; } // 获取通用数据指针(const)
    void* mutable_data_ptr() { return _data; }   // 获取通用数据指针(可变)
    const std::vector<size_t>& shape() const { return _shape; }
    size_t size() const;
    std::vector<float> data_cpu() const; // 将数据从任何设备拷贝到CPU并返回
    float item() const;

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
    // std::shared_ptr<std::vector<float>> _data;
    void* _data;
    std::vector<size_t> _shape;
    std::vector<size_t> _stride;
    bool _requires_grad;
    
    std::shared_ptr<Tensor> _grad;
    std::shared_ptr<Function> _ctx;

    Device _device;

    void compute_stride();
    void allocate_data(); // 在gpu中分配

    friend class Function;
};

std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);