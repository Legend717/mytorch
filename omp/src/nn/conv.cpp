#include "nn/conv.h"
#include "core/function.h"
#include <cmath>

namespace nn {

Conv2D::Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding)
    : _stride(stride), _padding(padding) {
    
    // Kaiming He 初始化
    float stdv = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));
    std::vector<size_t> weight_shape = {out_channels, in_channels, kernel_size, kernel_size};
    _weight = Tensor::randn(weight_shape, true);
    
    // 用stdv缩放
    auto weight_data = _weight->get_shared_data();
    for(auto& val : *weight_data) {
        val *= stdv;
    }

    // 偏置初始化为0
    std::vector<size_t> bias_shape = {1, out_channels, 1, 1};
    _bias = Tensor::zeros(bias_shape, true);
}
std::shared_ptr<Tensor> Conv2D::forward(std::shared_ptr<Tensor> input) {
    // 1. 调用底层的卷积运算
    auto conv_func = std::make_shared<Conv2DFunc>(_stride, _padding);
    auto output = conv_func->apply({input, _weight});

    // 2. 添加偏置 (需要广播)
    // 我们的 Add function 尚不支持4D广播，这里简化处理
    auto output_data = output->get_shared_data();
    const auto& bias_data = _bias->data();

    size_t N = output->shape()[0];
    size_t C_out = output->shape()[1];
    size_t H_out = output->shape()[2];
    size_t W_out = output->shape()[3];
    
    for(size_t n = 0; n < N; ++n) {
        for(size_t c = 0; c < C_out; ++c) {
            for(size_t h = 0; h < H_out; ++h) {
                for(size_t w = 0; w < W_out; ++w) {
                    (*output_data)[n * (C_out*H_out*W_out) + c * (H_out*W_out) + h * W_out + w] += bias_data[c];
                }
            }
        }
    }
    // 注意：上面手动的偏置加法没有构建计算图！
    // 一个完整的实现会创建一个支持广播的Add Function。为简化，我们暂时忽略偏置的梯度。

    return output;
}

std::vector<std::shared_ptr<Tensor>> Conv2D::parameters() {
    return {_weight, _bias};
}

} // namespace nn