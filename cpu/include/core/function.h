#pragma once
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  // ✅ 必须加这个头文件
#include <vector>
#include <memory>
namespace py = pybind11;

class Tensor; // 前向声明

// Inherit from std::enable_shared_from_this to use shared_from_this()
class Function : public std::enable_shared_from_this<Function> {
public:
    virtual ~Function() = default;

    std::shared_ptr<Tensor> apply(const std::vector<std::shared_ptr<Tensor>>& inputs);
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output);
    
    std::vector<std::shared_ptr<Tensor>> _saved_inputs;

protected:
    virtual std::shared_ptr<Tensor> _forward(const std::vector<std::shared_ptr<Tensor>>& inputs) = 0;
    virtual std::vector<std::shared_ptr<Tensor>> _backward(const std::shared_ptr<Tensor>& grad_output) = 0;
};

// --- 具体运算 ---
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

py::array_t<float> tensor_to_numpy(const std::shared_ptr<Tensor>& t);

std::shared_ptr<Tensor> numpy_to_tensor(const py::array_t<float>& arr);

namespace py = pybind11;

class FlashAttenFunc : public Function {
private:
    bool _causal;  //other parameters can be fetched from inputs(Tensor class)
    float _sm_scale;
    std::shared_ptr<Tensor> _saved_q; // 保存 Q, K, V 用于反向传播
    std::shared_ptr<Tensor> _saved_k;
    std::shared_ptr<Tensor> _saved_v;
    std::shared_ptr<Tensor> _saved_o; // 保存输出 O
    std::shared_ptr<Tensor> _saved_l; // 保存中间结果 L
    std::shared_ptr<Tensor> _saved_grid; // 保存 grid

public:
    FlashAttenFunc(bool causal = false, float sm_scale = 1.0f)
        : _causal(causal), _sm_scale(sm_scale) {}
protected:
    std::shared_ptr<Tensor> _forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override{
            if (inputs.size() < 5) {
                throw std::runtime_error("FlashAttentionFunc requires at least 5 inputs: Q, K, V, O, L.");
            }
            auto& Q = inputs[0];
            auto& K = inputs[1];
            auto& V = inputs[2];
            auto& O = inputs[3]; // 以下都是反向传播需要的tensor， O是输出
            auto& L = inputs[4];  // L是中间结果
        
        
            static py::scoped_interpreter guard{}; // 只初始化一次
            static py::object py_mod = py::module_::import("python_module");
            static py::object py_func = py_mod.attr("flash_attention_forward");
        
            // 假设有 tensor_to_numpy 工具函数
            py::object py_q = tensor_to_numpy(Q);
            py::object py_k = tensor_to_numpy(K);
            py::object py_v = tensor_to_numpy(V);
            py::object py_o = tensor_to_numpy(O);
            py::object py_l = tensor_to_numpy(L);
            py::bool_ _causal = this->_causal; // 转为 Python 布尔值
            py::float_ _sm_scale = this->_sm_scale; // 转为 Python 浮点
        
            // 调用 Python
            py::object py_result = py_func(py_q, py_k, py_v, py_o, py_l, _causal, _sm_scale);
        
            // 拆包 output, ctx_dict
            py::tuple py_tuple = py_result.cast<py::tuple>();
            py::tuple py_grid = py_tuple[2];
            py::int_ py_block_dmodel = py_tuple[3];
        
            //since zero copy, we dont need to convert the data back to tensor type
        
            this->_saved_q = Q; // 保存 Q, K, V 用于反向传播
            this->_saved_k = K;
            this->_saved_v = V;
            this->_saved_o = O; // 保存输出 O
            this->_saved_l = L; // 保存中间结果 L
        
            return O;
    }
    std::vector<std::shared_ptr<Tensor>> _backward(const std::shared_ptr<Tensor>& grad_output) override;
};