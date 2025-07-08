#include <iostream>
#include <memory>
#include <vector>

#include "core/tensor.h"
#include "core/function.h"
#include "nn/module.h"
#include "nn/linear.h"
#include "nn/activations.h"
#include "nn/sequential.h"
#include "nn/conv.h"
#include "nn/pool.h"
#include "nn/flatten.h"
#include "nn/flash-attn.h"
#include "optim/sgd.h"

int main() {
    static py::scoped_interpreter guard{}; // 只初始化一次
    try {
        auto q_train = Tensor::randn({64, 8, 128, 64}, true); // 假设输入是64个样本，每个样本8个通道，128x128的图像
        auto k_train = Tensor::randn({64, 8, 128, 64}, true); 
        auto v_train = Tensor::randn({64, 8, 128, 64}, true); 
        auto o_train = Tensor::zeros({64, 8, 128, 64}, false); // 输出张量
        auto L_train = Tensor::zeros({64 * 8, 128}, false);
        // 假设有 tensor_to_numpy 工具函数
        py::module_ sys = py::module_::import("sys");
        py::module_ np = py::module_::import("numpy");
        sys.attr("path").attr("insert")(0, py::str("/home/rogers/Documents/project/mytorch/cpu/src/nn"));
        static py::object py_mod = py::module_::import("flash-attn");
        
        static py::object py_func = py_mod.attr("attention");

        // 假设有 tensor_to_numpy 工具函数
        py::object py_q = tensor_to_numpy(q_train);
        py::object py_k = tensor_to_numpy(k_train);
        py::object py_v = tensor_to_numpy(v_train);
        py::object py_o = tensor_to_numpy(o_train);
        py::object py_l = tensor_to_numpy(L_train);

        py::bool_ _causal = false; // 转为 Python 布尔值
        py::float_ _sm_scale = 1.0f; // 转为 Python

        // 测试 Flash Attention
        py::object py_result = py_func(py_q, py_k, py_v, py_o, py_l, _causal, _sm_scale);
        // 处理返回值

        py::tuple py_result_tuple = py::cast<py::tuple>(py_result);
        if (py_result_tuple.size() != 4) {
            throw std::runtime_error("Expected 4 outputs from attention function");
        }

        
        // auto attn_layer = std::make_shared<nn::FlashAttn>(false, 1.0f); 
        // auto o = attn_layer->forward({q_train, k_train, v_train});

        //check if the output is a numpy array
        
    
    }catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

   

    return 0;
}
