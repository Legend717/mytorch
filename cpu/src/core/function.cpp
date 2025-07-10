#include "core/function.h"
#include "core/tensor.h"
#include <stdexcept>
#include <numeric>
#include <omp.h>
#include <iostream>

// 基类
std::shared_ptr<Tensor> Function::apply(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    _saved_inputs = inputs;
    auto output = _forward(inputs);

    bool requires_grad = false;
    for (const auto& input: inputs) {
        if (input->requires_grad()) {
            requires_grad = true;
            break;
        }
    }

    if (requires_grad) {
        output->_requires_grad = true;
        output->set_ctx(shared_from_this());
    }
    return output;
}

void Function::release_saved_inputs() {
    _saved_inputs.clear(); // <-- 清空保存的输入
}

// 加法
std::vector<std::shared_ptr<Tensor>> Function::backward(const std::shared_ptr<Tensor>& grad_output) {
    return _backward(grad_output);
}

std::shared_ptr<Tensor> Add::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    auto a = inputs[0];
    auto b = inputs[1];

    // 如果维度不同，则进行广播加法
    if (a->shape() != b->shape() && b->shape()[0] == 1 && a->shape()[1] == b->shape()[1]) {
        std::vector<float> result_data(a->data().size());
        size_t batch_size = a->shape()[0];
        size_t features = a->shape()[1];
        #pragma omp parallel for
        for(size_t i = 0; i < batch_size; ++i) {
            for(size_t j = 0; j < features; ++j) {
                result_data[i * features + j] = a->data()[i * features + j] + b->data()[j];
            }
        }
        return Tensor::create(result_data, a->shape());
    }
    else { // 维度相同，直接加法
        const auto& a_data = a->data();
        const auto& b_data = b->data();
        std::vector<float> result_data(a_data.size());
        #pragma omp parallel for
        for (size_t i = 0; i < a_data.size(); ++i) {
            result_data[i] = a_data[i] + b_data[i];
        }
        return Tensor::create(result_data, a->shape());
    }
    // TODO: 维度不同，但不能广播的情况（暂时假设不存在此错误）
}

std::vector<std::shared_ptr<Tensor>> Add::_backward(const std::shared_ptr<Tensor>& grad_output) {
    // 加法操作的梯度是 1，因此 grad_a 和 grad_b 直接等于 grad_output
    auto a = _saved_inputs[0];
    auto b = _saved_inputs[1];
    auto grad_a = grad_output;
    auto grad_b = grad_output;

    // 处理维度不同的情况，此时 grad_b 需要进行广播
    if (a->shape() != b->shape()) {
        // 由于 b 被广播到 a 的每一行，反向传播时需要将 grad_output 的所有行梯度累加到 b 的对应位置
        size_t batch_size = grad_output->shape()[0];
        size_t features = grad_output->shape()[1];
        std::vector<float> sum_grad_data(features, 0.0f);

        // 优化循环顺序：外循环features，内循环batch_size
        #pragma omp parallel for
        for(size_t j = 0; j < features; ++j) {
            float sum = 0.0f;
            for(size_t i = 0; i < batch_size; ++i) {
                sum += grad_output->data()[i * features + j];
            }
            sum_grad_data[j] = sum;
        }
        grad_b = Tensor::create(sum_grad_data, b->shape());

        // for(size_t i = 0; i < batch_size; ++i) {
        //     for(size_t j = 0; j < features; ++j) {
        //         sum_grad_data[j] += grad_output->data()[i * features + j];
        //     }
        // }
        // grad_b = Tensor::create(sum_grad_data, b->shape());
    }
    return {grad_a, grad_b};
}

// 减法，暂不考虑广播情况
std::shared_ptr<Tensor> Sub::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    const auto& a = inputs[0]->data();
    const auto& b = inputs[1]->data();
    std::vector<float> result_data(a.size());
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); ++i) {
        result_data[i] = a[i] - b[i];
    }
    return Tensor::create(result_data, inputs[0]->shape());
}

std::vector<std::shared_ptr<Tensor>> Sub::_backward(const std::shared_ptr<Tensor>& grad_output) {
    // 减法的b梯度是-1，因此 grad_b 等于 -grad_output
    auto neg_grad = grad_output->mul(Tensor::create({-1.0f}, {1}));
    return {grad_output, neg_grad};
}

// 乘法（逐位）
std::shared_ptr<Tensor> Mul::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    const auto& a = inputs[0]->data();
    const auto& b = inputs[1]->data();
    std::vector<float> result_data(a.size());   
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); ++i) {
        result_data[i] = a[i] * b[i];
    }
    return Tensor::create(result_data, inputs[0]->shape());
}

std::vector<std::shared_ptr<Tensor>> Mul::_backward(const std::shared_ptr<Tensor>& grad_output) {
    // z = a * b, dl/da = dl/dz * b, dl/db = dl/dz * a
    auto a = _saved_inputs[0];
    auto b = _saved_inputs[1];
    return {grad_output->mul(b), grad_output->mul(a)};
}

// 矩阵乘法
std::shared_ptr<Tensor> MatMul::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    if (a->shape().size() != 2 || b->shape().size() != 2 || a->shape()[1] != b->shape()[0]) {
        throw std::runtime_error("矩阵乘法输入维度不匹配");
    }
    size_t M = a->shape()[0];
    size_t K = a->shape()[1];
    size_t N = b->shape()[1];
    std::vector<float> result_data(M * N, 0.0f);
    #pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t k = 0; k < K; ++k) {
                result_data[i * N + j] += a->data()[i * K + k] * b->data()[k * N + j];
            }
        }
    }
    return Tensor::create(result_data, {M, N});
}

std::vector<std::shared_ptr<Tensor>> MatMul::_backward(const std::shared_ptr<Tensor>& grad_output) {
    auto a = _saved_inputs[0];
    auto b = _saved_inputs[1];
    // grad_a = grad_output @ b.T
    // grad_b = a.T @ grad_output
    auto grad_a = grad_output->matmul(b->transpose());
    auto grad_b = a->transpose()->matmul(grad_output);
    return {grad_a, grad_b};
}

// 求和
std::shared_ptr<Tensor> Sum::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    // float sum_val = std::accumulate(inputs[0]->data().begin(), inputs[0]->data().end(), 0.0f);
    const auto& data = inputs[0]->data();
    float sum_val = 0.0f;
    #pragma omp parallel for reduction(+:sum_val)
    for (size_t i = 0; i < data.size(); ++i) {
        sum_val += data[i];
    }
    return Tensor::create({sum_val}, {1});
}

std::vector<std::shared_ptr<Tensor>> Sum::_backward(const std::shared_ptr<Tensor>& grad_output) {
    auto original_shape = _saved_inputs[0]->shape();
    size_t total_size = 1;
    for(auto dim : original_shape) total_size *= dim;
    // 对于求和，梯度是全 1，因此 grad_output 直接作为输出
    std::vector<float> grad_data(total_size, grad_output->data()[0]);
    return {Tensor::create(grad_data, original_shape)};
}

// ReLU
std::shared_ptr<Tensor> ReLUFunc::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    const auto& x = inputs[0]->data();
    std::vector<float> result_data(x.size());
    #pragma omp parallel for
    for(size_t i=0; i<x.size(); ++i) {
        result_data[i] = std::max(0.0f, x[i]);
    }
    return Tensor::create(result_data, inputs[0]->shape());
}

std::vector<std::shared_ptr<Tensor>> ReLUFunc::_backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = _saved_inputs[0]->data();
    std::vector<float> mask_data(x.size());
    // ReLU的梯度是0或1
    #pragma omp parallel for
    for(size_t i = 0; i < x.size(); ++i) {
        mask_data[i] = x[i] > 0 ? 1.0f : 0.0f;
    }
    auto mask = Tensor::create(mask_data, _saved_inputs[0]->shape());
    return {grad_output->mul(mask)};
}

// 卷积
std::shared_ptr<Tensor> Conv2DFunc::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    /*
    inputs两个输入，第一个是被卷积的Tensor，第二个是卷积核kernel；
    对图片进行卷积，可以是黑白图片，也可以是彩色图片
    input: [N_in, C_in, H_in, W_in]
    weight: [N_w，C_w, H_w, W_w]
    2d卷积要求两个C相同
    output: [N_in, N_w, H, W]，N_w代表不同的卷积核
    */
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    
    size_t N = input->shape()[0];
    size_t C_in = input->shape()[1];
    size_t H_in = input->shape()[2];
    size_t W_in = input->shape()[3];

    size_t N_w = weight->shape()[0];
    size_t C_w = weight->shape()[1];
    size_t H_w = weight->shape()[2];
    size_t W_w = weight->shape()[3];

    if(C_in != C_w){
        std::cout<<C_in <<" "<< C_w<<std::endl;
        throw std::runtime_error("2d卷积中卷积核和输入的维度不匹配");
    }

    size_t H_out = static_cast<size_t>(floor(static_cast<float> (H_in + 2*_padding - H_w)/_stride)) + 1;
    size_t W_out = static_cast<size_t>(floor(static_cast<float> (W_in + 2*_padding - W_w)/_stride)) + 1;

    std::vector<size_t> out_shape = {N, N_w, H_out, W_out};
    std::vector<float> out_data(N * N_w * H_out * W_out, 0.0f);

    # pragma omp parallel for
    for(size_t n = 0; n < N; ++n){  //各个Batch样本
        for(size_t l = 0; l < N_w; ++l){    //各个卷积核
            for(size_t h = 0; h < H_out; ++h){  //输出的H维度
                for(size_t w = 0; w < W_out; ++w){  //输出的W维度
                    // 输出的每个元素，由互相关操作获得，关键是找到输入的小方块和fliter中的小方块
                    float sum = 0.0f;
                    for(size_t c = 0; c<C_in; ++c){
                        for(size_t kh=0; kh < H_w; ++kh){
                            for(size_t kw=0; kw < W_w; ++kw){
                                int h_in_idx = static_cast<int> (h * _stride - _padding + kh);
                                int w_in_idx = static_cast<int> (w * _stride - _padding + kw);
                                
                                if(h_in_idx >=0 && h_in_idx < H_in && w_in_idx >=0 && w_in_idx < W_in){ //input的shape [N, C, H, W], weight的shape[N, C, H, W]
                                    size_t input_idx = n*(C_in * H_in * W_in) + c * (H_in * W_in) + h_in_idx*W_in + w_in_idx;
                                    size_t weight_idx = l * (C_w * H_w * W_w) + c * (H_w * W_w) + kh * W_w + kw;
                                    sum += input->data()[input_idx]* weight->data()[weight_idx];
                                }
                            }
                        }
                    }
                    size_t out_idx = n * (N_w * H_out * W_out) + l * (H_out * W_out) + h * W_out + w;
                    out_data[out_idx] = sum;
                }
            }
        }
    }
    return Tensor::create(out_data, out_shape);
}

std::vector<std::shared_ptr<Tensor>> Conv2DFunc::_backward(const std::shared_ptr<Tensor>& grad_output) {
    /*
    对于权重的梯度，为 Input和grad_output 的卷积
    对于输入的梯度，为 grad_output 和 旋转的卷积核 的全卷积 
    input: [N_in, C_in, H_in, W_in]
    weight: [N_w，C_w, H_w, W_w]
    */
    const auto& input = _saved_inputs[0];
    const auto& weight = _saved_inputs[1];

    const auto& in_shape = input->shape();
    const auto& w_shape = weight->shape();
    const auto& grad_out_shape = grad_output->shape();
    size_t N = in_shape[0];
    size_t C_in = in_shape[1];
    size_t H_in = in_shape[2];
    size_t W_in = in_shape[3];
    size_t C_out = w_shape[0];
    size_t H_w = w_shape[2];
    size_t W_w = w_shape[3];
    size_t H_out = grad_out_shape[2];
    size_t W_out = grad_out_shape[3];

    // 初始化梯度张量
    // 使用 zeros 初始化，确保梯度从0开始累加
    auto grad_input = Tensor::zeros(in_shape, true);
    auto grad_weight = Tensor::zeros(w_shape, true);

    auto grad_input_data = grad_input->get_shared_data();
    auto grad_weight_data = grad_weight->get_shared_data();

    const auto& input_data = input->data();
    const auto& weight_data = weight->data();
    const auto& grad_output_data = grad_output->data();

    #pragma omp parallel for 
    for (size_t n = 0; n < N; ++n) {
        for (size_t c_out = 0; c_out < C_out; ++c_out) {
            for (size_t h_out = 0; h_out < H_out; ++h_out) {
                for (size_t w_out = 0; w_out < W_out; ++w_out) {
                    
                    size_t grad_out_idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out;
                    float grad_out_val = grad_output_data[grad_out_idx];

                    if (grad_out_val == 0.0f) continue;

                    for (size_t c_in = 0; c_in < C_in; ++c_in) {
                        for (size_t kh = 0; kh < H_w; ++kh) {
                            for (size_t kw = 0; kw < W_w; ++kw) {
                                // 定义int类型的索引，用于边界检查
                                int h_in_idx_int = static_cast<int>(h_out * _stride) - _padding + kh;
                                int w_in_idx_int = static_cast<int>(w_out * _stride) - _padding + kw;

                                // 边界检查
                                if (h_in_idx_int >= 0 && h_in_idx_int < H_in && w_in_idx_int >= 0 && w_in_idx_int < W_in) {
                                    
                                    // 直接使用检查过的int值进行计算，它们会被安全地转换为size_t
                                    size_t input_idx = n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_in_idx_int * W_in + w_in_idx_int;
                                    size_t weight_idx = c_out * (C_in * H_w * W_w) + c_in * (H_w * W_w) + kh * W_w + kw;
                                    
                                    // 累加梯度
                                    // 注意：这里需要使用原子操作来保证并行安全
                                    #pragma omp atomic
                                    (*grad_weight_data)[weight_idx] += grad_out_val * input_data[input_idx];
                                    
                                    #pragma omp atomic
                                    (*grad_input_data)[input_idx] += grad_out_val * weight_data[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return {grad_input, grad_weight};
}

// 最大池化
std::shared_ptr<Tensor> MaxPool2DFunc::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    const auto& input = inputs[0];
    const auto& in_shape = input->shape();
    const size_t N = in_shape[0], C = in_shape[1], H_in = in_shape[2], W_in = in_shape[3];
    
    const size_t H_out = (H_in - _kernel_size) / _stride + 1;
    const size_t W_out = (W_in - _kernel_size) / _stride + 1;
    const std::vector<size_t> out_shape = {N, C, H_out, W_out};

    //初始化输出张量和调整索引vector
    auto output = Tensor::zeros(out_shape, false);
    _max_indices.resize(N * C * H_out * W_out);

    const auto& input_data = input->data();
    auto output_data = output->get_shared_data();
    
    // 循环计算
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t h_out = 0; h_out < H_out; ++h_out) {
                for (size_t w_out = 0; w_out < W_out; ++w_out) {
                    
                    float max_val = -std::numeric_limits<float>::infinity();
                    size_t max_idx = 0;

                    for (size_t kh = 0; kh < _kernel_size; ++kh) {
                        for (size_t kw = 0; kw < _kernel_size; ++kw) {
                            size_t h_in = h_out * _stride + kh;
                            size_t w_in = w_out * _stride + kw;
                            size_t in_idx = n * (C * H_in * W_in) + c * (H_in * W_in) + h_in * W_in + w_in;

                            if (input_data[in_idx] > max_val) {
                                max_val = input_data[in_idx];
                                max_idx = in_idx;
                            }
                        }
                    }
                    
                    size_t out_idx = n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out;

                    (*output_data)[out_idx] = max_val;
                    _max_indices[out_idx] = max_idx;
                }
            }
        }
    }
    
    return output;
}

std::vector<std::shared_ptr<Tensor>> MaxPool2DFunc::_backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& input = _saved_inputs[0];
    const auto& in_shape = input->shape();

    auto grad_input = Tensor::zeros(in_shape, true);
    
    const auto& grad_output_data = grad_output->data();
    auto grad_input_data = grad_input->get_shared_data();

    const size_t grad_output_size = grad_output->data().size();

    // 使用_max_indices
    for (size_t i = 0; i < grad_output_size; ++i) {
        size_t max_idx = _max_indices[i];
        (*grad_input_data)[max_idx] += grad_output_data[i];
    }
    
    return {grad_input};
}

// Reshape
std::shared_ptr<Tensor> ReshapeFunc::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    auto input = inputs[0];
    return Tensor::create(input->data(), _new_shape);
}

std::vector<std::shared_ptr<Tensor>> ReshapeFunc::_backward(const std::shared_ptr<Tensor>& grad_output) {
    auto original_shape = _saved_inputs[0]->shape();
    return { grad_output->reshape(original_shape) };
}


// --- FlashAttentionFunc ---

namespace py = pybind11;

//TODO: tensor to numpy 和 numpy to tensor 的转换函数需要实现:
py::array_t<float> tensor_to_numpy(const std::shared_ptr<Tensor>& t) {
    std::vector<ssize_t> shape(t->shape().begin(), t->shape().end());
    return py::array_t<float>(
        shape,
        t->data().data()
    );
}

// 从 numpy 转为 Tensor（copy）
std::shared_ptr<Tensor> numpy_to_tensor(const py::array_t<float>& arr) {
    auto buf = arr.request();
    const float* ptr = static_cast<float*>(buf.ptr);
    std::vector<float> data(ptr, ptr + buf.size);
    std::vector<size_t> shape(buf.shape.begin(), buf.shape.end());
    return std::make_shared<Tensor>(std::move(data), std::move(shape));
}

std::shared_ptr<Tensor> FlashAttenFunc::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.size() < 3) {
        throw std::runtime_error("FlashAttentionFunc requires at least 3 inputs: Q, K, V");
    }
    try {
        py::gil_scoped_acquire gil;  // ⬅️ 保证持有 GIL

        auto& Q = inputs[0];
        auto& K = inputs[1];
        auto& V = inputs[2];
        auto O = Tensor::zeros(Q->shape(), false); 
        auto L = Tensor::zeros({Q->shape()[0] * Q->shape()[1], Q->shape()[2]}, false);

        // 只初始化一次 Python 模块和函数
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("insert")(0, "/home/rogers/Documents/project/mytorch/cpu/src/nn/");
        py::module_ attn = py::module_::import("flash_attn");

        // Tensor 转 numpy
        py::object py_q = tensor_to_numpy(Q);
        py::object py_k = tensor_to_numpy(K);
        py::object py_v = tensor_to_numpy(V);
        py::object py_o = tensor_to_numpy(O);
        py::object py_l = tensor_to_numpy(L);
        py::bool_ _causal = this->_causal;
        py::float_ _sm_scale = this->_sm_scale;


        py::object py_result = attn.attr("attention")(py_q, py_k, py_v, py_o, py_l, _causal, _sm_scale);

        // 解析返回值
        py::tuple py_tuple = py_result.cast<py::tuple>();
        py::tuple py_grid = py_tuple[2];
        py::int_ py_block_dmodel = py_tuple[3];
        py::object py_o_result = py_tuple[0]; // O
        py::object py_l_result = py_tuple[1]; // L

        
        this->_block_dmodel = py_block_dmodel.cast<int>();
        this->_grid[0] = py_grid[0].cast<int>();
        this->_grid[1] = py_grid[1].cast<int>();
        this->_grid[2] = py_grid[2].cast<int>();
        auto c_result_o = numpy_to_tensor(py_o_result.cast<py::array_t<float>>());
        auto c_result_l = numpy_to_tensor(py_l_result.cast<py::array_t<float>>());
        this-> saved_o = c_result_o;
        this-> saved_l = c_result_l;
        // printf("FlashAttentionFunc forward saved inputs size: %zu\n", _saved_inputs.size());
        //输出所有的saved input的size
        // for (size_t i = 0; i < _saved_inputs.size(); ++i) {
        //     printf("saved input[%zu] shape: ", i);
        //     for (auto dim : _saved_inputs[i]->shape()) {
        //         printf("%zu ", dim);
        //     }
        //     printf("\n");
        // }
        // printf("save grad bool %s\n", _saved_inputs[0]->requires_grad() ? "true" : "false");
        return c_result_o;
    } catch (const py::error_already_set& e) {
        throw std::runtime_error("Error in FlashAttentionFunc: " + std::string(e.what()));
    }
}

void show10_tensor(std::shared_ptr<Tensor> t) {
    for (size_t i = 0; i < std::min(10UZ, t->data().size()); ++i) {
        printf("tensor[%zu] = %f\n", i, t->data()[i]);
    }
}


std::vector<std::shared_ptr<Tensor>> FlashAttenFunc::_backward(const std::shared_ptr<Tensor>& grad_output) {
    printf("cpp backwarding \n");
    // for(auto &t : _saved_inputs) {
    //     show10_tensor(t);
    // }
    auto d_q = Tensor::zeros(grad_output->shape(), false);
    auto d_k = Tensor::zeros(grad_output->shape(), false);
    auto d_v = Tensor::zeros(grad_output->shape(), false);

    py::gil_scoped_acquire gil;  // ⬅️ 保证持有 GIL
    // 1. 获取输入张量
    py::object py_q = tensor_to_numpy(_saved_inputs[0]);
    py::object py_k = tensor_to_numpy(_saved_inputs[1]);
    py::object py_v = tensor_to_numpy(_saved_inputs[2]);
    py::object py_o = tensor_to_numpy(this->saved_o);
    py::object py_l = tensor_to_numpy(this->saved_l);
    py::object py_grad = tensor_to_numpy(grad_output);
    py::bool_ _causal = this->_causal;
    py::float_ _sm_scale = this->_sm_scale;
    py::int_ py_block_dmodel = this->_block_dmodel;
    py::int_ py_grid0 = this->_grid[0];
    py::int_ py_grid1 = this->_grid[1];
    py::int_ py_grid2 = this->_grid[2];
    py::object py_dq = tensor_to_numpy(d_q);
    py::object py_dk = tensor_to_numpy(d_k);
    py::object py_dv = tensor_to_numpy(d_v);

    //call python srcipt backward
    py::module_ attn = py::module_::import("flash_attn");
    py::object py_result = attn.attr("backward")(py_dq, py_dk, py_dv, py_grad,
        py_q, py_k, py_v, py_o, py_l, 
        py_grid0, py_grid1, py_grid2, py_block_dmodel, _causal, _sm_scale
    );
    // 2. 解析返回值
    py::tuple py_tuple = py_result.cast<py::tuple>();
    //3 转换为Tensor
    std::shared_ptr<Tensor> grad_q = numpy_to_tensor(py_tuple[0].cast<py::array_t<float>>());
    std::shared_ptr<Tensor> grad_k = numpy_to_tensor(py_tuple[1].cast<py::array_t<float>>());
    std::shared_ptr<Tensor> grad_v = numpy_to_tensor(py_tuple[2].cast<py::array_t<float>>());

    //check 10 ele from grad_q
    // for (size_t i = 0; i < std::min(10UL, grad_q->data().size()); ++i) {
    //     printf("grad_q[%zu] = %f\n", i, grad_q->data()[i]);
    // }
    // printf("transform is done!\n");
    // 4. 返回梯度列表
    return {grad_q, grad_k, grad_v};
}