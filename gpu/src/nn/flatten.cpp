#include "nn/flatten.h"
#include "core/tensor.h"
#include "core/function.h" // 确保包含了 Function 的定义

namespace nn {

std::shared_ptr<Tensor> Flatten::forward(std::shared_ptr<Tensor> input) {
    _original_shape = input->shape();

    size_t batch_size = _original_shape[0];
    size_t flattened_size = 1;
    for (size_t i = 1; i < _original_shape.size(); ++i) {
        flattened_size *= _original_shape[i];
    }
    
    // --- 修改 ---
    // 调用 Tensor::reshape，这个方法内部会创建一个 ReshapeFunc 并链接计算图
    return input->reshape({batch_size, flattened_size});
}

} // namespace nn