#include "nn/flash-attn.h"
#include "core/function.h"
#include <cmath>

namespace nn {

FlashAttn::FlashAttn(bool causal, float sm_scale): Module(), _causal(causal), _sm_scale(sm_scale){}

std::shared_ptr<Tensor> FlashAttn::forward(std::vector<std::shared_ptr<Tensor>> input) {
    // 1. 调用底层的注意力运算
    auto flash_func = std::make_shared<FlashAttenFunc>(_causal, _sm_scale); 
    printf("calling FlashAttn forward with %zu inputs\n", input.size());
    
    auto output = flash_func->apply(input); // 假设输入是一个包含 Q, K, V 的张量
    return output;
}

std::shared_ptr<Tensor> FlashAttn::forward(std::shared_ptr<Tensor> input) {
    // 占位实现：返回与输入形状一致的零梯度张量
    throw std::runtime_error("FlashAttenFunc does not support single tensor forward.");
}

std::vector<std::shared_ptr<Tensor>> FlashAttn::parameters() {
    return {};
}

} // namespace nn