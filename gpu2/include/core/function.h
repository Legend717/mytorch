#pragma once

#include <vector>
#include <memory>

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
    std::shared_ptr<Tensor> _max_indices_tensor; 
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

// CUDA相关函数
std::shared_ptr<Tensor> add_forward_cuda(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> mul_forward_cuda(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> matmul_forward_cuda(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> relu_forward_cuda(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> reshape_forward_cuda(const std::shared_ptr<Tensor>& input, const std::vector<size_t>& new_shape);
std::shared_ptr<Tensor> add_backward_cuda(const std::shared_ptr<Tensor>& grad_output, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> relu_backward_cuda(const std::shared_ptr<Tensor>& grad_output, const std::shared_ptr<Tensor>& input);std::shared_ptr<Tensor> im2col_cuda(const std::shared_ptr<Tensor>& input, size_t K, size_t S, size_t P);
std::shared_ptr<Tensor> nhwc_to_nchw_cuda(const std::shared_ptr<Tensor>& input_nhwc);
std::vector<std::shared_ptr<Tensor>> conv2d_backward_cuda( const std::shared_ptr<Tensor>& grad_output, const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& weight, size_t stride, size_t padding
);
std::shared_ptr<Tensor> maxpool2d_forward_cuda(const std::shared_ptr<Tensor>& input, size_t kernel_size, size_t stride, std::shared_ptr<Tensor>& max_indices_tensor);
std::shared_ptr<Tensor> maxpool2d_backward_cuda(const std::shared_ptr<Tensor>& grad_output, const std::shared_ptr<Tensor>& max_indices_tensor, const std::vector<size_t>& input_shape);
void rearrange_output_kernel_launcher(const float* matmul_data, float* output_data,
                                     int N, int C_out, int H_out, int W_out);