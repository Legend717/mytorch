#pragma once
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  // ✅ 必须加这个头文件b
#include <vector>
#include <memory>
#include "tensor.h"  // Include the header file where Tensor is defined
namespace py = pybind11;

class Tensor;

// 继承自enable_shared_from_this，使得Function对象可以被shared_ptr管理
class Function : public std::enable_shared_from_this<Function> {
public:
    virtual ~Function() = default;

    std::shared_ptr<Tensor> apply(const std::vector<std::shared_ptr<Tensor>>& inputs);
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output);
    
    std::vector<std::shared_ptr<Tensor>> _saved_inputs;
    void release_saved_inputs(); 
    
protected:
    // 纯虚函数，由子类实现
    virtual std::shared_ptr<Tensor> _forward(const std::vector<std::shared_ptr<Tensor>>& inputs) = 0;
    virtual std::vector<std::shared_ptr<Tensor>> _backward(const std::shared_ptr<Tensor>& grad_output) = 0;
};

// 运算
class Add : public Function {
protected:
    std::shared_ptr<Tensor> _forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    std::vector<std::shared_ptr<Tensor>> _backward(const std::shared_ptr<Tensor>& grad_output) override;
};

class Sub : public Function {
protected:
    std::shared_ptr<Tensor> _forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    std::vector<std::shared_ptr<Tensor>> _backward(const std::shared_ptr<Tensor>& grad_output) override;
};

class Mul : public Function {
protected:
    std::shared_ptr<Tensor> _forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    std::vector<std::shared_ptr<Tensor>> _backward(const std::shared_ptr<Tensor>& grad_output) override;
};

class MatMul : public Function {
protected:
    std::shared_ptr<Tensor> _forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    std::vector<std::shared_ptr<Tensor>> _backward(const std::shared_ptr<Tensor>& grad_output) override;
};

class Sum : public Function {
protected:
    std::shared_ptr<Tensor> _forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    std::vector<std::shared_ptr<Tensor>> _backward(const std::shared_ptr<Tensor>& grad_output) override;
};

class ReLUFunc : public Function {
protected:
    std::shared_ptr<Tensor> _forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    std::vector<std::shared_ptr<Tensor>> _backward(const std::shared_ptr<Tensor>& grad_output) override;
};

class Conv2DFunc : public Function {
private:
    size_t _stride;
    size_t _padding;
public:
    Conv2DFunc(size_t stride = 1, size_t padding = 0) : _stride(stride), _padding(padding) {}
protected:
    std::shared_ptr<Tensor> _forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    std::vector<std::shared_ptr<Tensor>> _backward(const std::shared_ptr<Tensor>& grad_output) override;
};

class MaxPool2DFunc : public Function {
private:
    size_t _kernel_size;
    size_t _stride;
    std::vector<size_t> _max_indices; // 缓存最大值的位置，用于反向传播

public:
    MaxPool2DFunc(size_t kernel_size, size_t stride) : _kernel_size(kernel_size), _stride(stride) {}
protected:
    std::shared_ptr<Tensor> _forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    std::vector<std::shared_ptr<Tensor>> _backward(const std::shared_ptr<Tensor>& grad_output) override;
};

class ReshapeFunc : public Function {
private:
    std::vector<size_t> _new_shape;
public:
    ReshapeFunc(const std::vector<size_t>& new_shape) : _new_shape(new_shape) {}
protected:
    std::shared_ptr<Tensor> _forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    std::vector<std::shared_ptr<Tensor>> _backward(const std::shared_ptr<Tensor>& grad_output) override;
};

// --- Flash Attention ---
class FlashAttenFunc : public Function {
    private:
        bool _causal;  //other parameters can be fetched from inputs(Tensor class)
        float _sm_scale;
        int _block_dmodel; // 用于存储 block_dmodel
        int _grid[3]; // 用于存储 grid
        std::shared_ptr<Tensor> saved_o; // 保存 O 的张量
        std::shared_ptr<Tensor> saved_l; // 保存 L 的张量
    
    public:
        FlashAttenFunc(bool causal = false, float sm_scale = 1.0f)
            : _causal(causal), _sm_scale(sm_scale) {}
    protected:
        std::shared_ptr<Tensor> _forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override;
     
        std::vector<std::shared_ptr<Tensor>> _backward(const std::shared_ptr<Tensor>& grad_output) override;
    };