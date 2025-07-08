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
    py::scoped_interpreter guard{};  // 只初始化一次解释器
    std::cout << "Python interpreter initialized" << std::endl;

    // 创建输入
    auto Q = Tensor::randn({64, 8, 128, 64});
    auto K = Tensor::randn({64, 8, 128, 64});
    auto V = Tensor::randn({64, 8, 128, 64});

    // 调用 flash attention 前向
    auto attn_layer = std::make_shared<nn::FlashAttn>(false, 1.0f); 
    auto o = attn_layer->forward({Q, K, V});

    return 0;  // 解释器由 guard 自动清理
}



// int main() {
//     py::scoped_interpreter guard{}; // 只初始化一次
//     try {
//         // 添加 site-packages 到 sys.path
//         py::module_ sys = py::module_::import("sys");
//         sys.attr("path").attr("insert")(0, "/home/rogers/miniconda3/envs/pytorch/lib/python3.12/site-packages");
//         sys.attr("path").attr("insert")(0, "../src/nn");
//         py::module_ attn = py::module_::import("flash-attn");

//                 // 验证 numpy 可用
//         py::module_ np = py::module_::import("numpy");
//         std::cout << "numpy version: " << np.attr("__version__").cast<std::string>() << std::endl;

//         auto q_train = Tensor::randn({16, 8, 128, 16}, true); // 假设输入是64个样本，每个样本8个通道，128x128的图像
//         auto k_train = Tensor::randn({16, 8, 128, 16}, true); 
//         auto v_train = Tensor::randn({16, 8, 128, 16}, true); 


//         // auto np_q = tensor_to_numpy(q_train);
//         // auto np_k = tensor_to_numpy(k_train);
//         // auto np_v = tensor_to_numpy(v_train);
//         // auto np_o = tensor_to_numpy(o_train);
//         // auto np_l = tensor_to_numpy(L_train);


//         // 测试 Flash Attention

//         // attn.attr("attention")(
//         //     np_q, np_k, np_v, np_o, np_l, true, 1.0f
//         // );

//         auto attn_layer = std::make_shared<nn::FlashAttn>(false, 1.0f); 
//         auto o = attn_layer->forward({q_train, k_train, v_train});

//         //check if the output is a numpy array
        
//     }catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }

//     return 0;
// }
