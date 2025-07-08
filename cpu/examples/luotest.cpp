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
#include "optim/sgd.h"

int main() {
    py::scoped_interpreter guard{};  // 启动 Python 解释器

    try {
        // 添加 site-packages 到 sys.path
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("insert")(0, "/home/rogers/miniconda3/envs/pytorch/lib/python3.12/site-packages");
        sys.attr("path").attr("insert")(0, "../src/nn");
        py::module_ attn = py::module_::import("flash-attn");


        // 验证 numpy 可用
        py::module_ np = py::module_::import("numpy");
        std::cout << "numpy version: " << np.attr("__version__").cast<std::string>() << std::endl;

        auto q_train = Tensor::randn({16, 8, 128, 16}, true); // 假设输入是64个样本，每个样本8个通道，128x128的图像
        auto k_train = Tensor::randn({16, 8, 128, 16}, true); 
        auto v_train = Tensor::randn({16, 8, 128, 16}, true); 
        auto o = Tensor::randn({16, 8, 128, 16}, true); 
        auto l = Tensor::randn({128,16}, false); 
    
        // --- 转为 numpy 并打印 ---
        auto np_q = tensor_to_numpy(q_train);
        auto np_k = tensor_to_numpy(k_train);
        auto np_v = tensor_to_numpy(v_train);
        auto np_o = tensor_to_numpy(o);
        auto np_l = tensor_to_numpy(l);

        attn.attr("attention")(
            np_q, np_k, np_v, np_o, np_l, true, 1.0f
        );
    
    } catch (const py::error_already_set& e) {
        std::cerr << "Python error: " << e.what() << std::endl;
    }

   

    return 0;
}
