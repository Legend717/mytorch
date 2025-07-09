#include "nn/conv.h"
#include "core/function.h"
#include <cmath>

namespace nn {

Conv2D::Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding)
    : _stride(stride), _padding(padding) {
    // Kaiming He 初始化
    float stdv = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));
    size_t weight_size = out_channels * in_channels * kernel_size * kernel_size;

    std::vector<float> weight_data_cpu(weight_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    for(auto& val : weight_data_cpu) {
        val = d(gen) * stdv;
    }
    
    std::vector<size_t> weight_shape = {out_channels, in_channels, kernel_size, kernel_size};
    _weight = Tensor::create(weight_data_cpu, weight_shape, true);

    // 偏置初始化为0
    std::vector<size_t> bias_shape = {1, out_channels, 1, 1};
    _bias = Tensor::zeros(bias_shape, true);
}

std::shared_ptr<Tensor> Conv2D::forward(std::shared_ptr<Tensor> input) {
    auto conv_func = std::make_shared<Conv2DFunc>(_stride, _padding);
    auto output = conv_func->apply({input, _weight});

    // 现在这个add操作将依赖于支持广播的CUDA核
    return output->add(_bias);
}

// 弃用
// std::shared_ptr<Tensor> Conv2D::forward(std::shared_ptr<Tensor> input) {
//     // 1. 调用底层的卷积运算
//     auto conv_func = std::make_shared<Conv2DFunc>(_stride, _padding);
//     auto output = conv_func->apply({input, _weight});

//     // 2. 添加偏置 (需要广播)
//     // 我们的 Add function 尚不支持4D广播，这里简化处理
//     auto output_data = output->mutable_data_ptr();
//     const auto& bias_data = _bias->data();

//     size_t N = output->shape()[0];
//     size_t C_out = output->shape()[1];
//     size_t H_out = output->shape()[2];
//     size_t W_out = output->shape()[3];
    
//     for(size_t n = 0; n < N; ++n) {
//         for(size_t c = 0; c < C_out; ++c) {
//             for(size_t h = 0; h < H_out; ++h) {
//                 for(size_t w = 0; w < W_out; ++w) {
//                     (*output_data)[n * (C_out*H_out*W_out) + c * (H_out*W_out) + h * W_out + w] += bias_data[c];
//                 }
//             }
//         }
//     }
//     // 注意：上面手动的偏置加法没有构建计算图！
//     // 一个完整的实现会创建一个支持广播的Add Function。为简化，我们暂时忽略偏置的梯度。

//     return output;
// }

std::vector<std::shared_ptr<Tensor>> Conv2D::parameters() {
    return {_weight, _bias};
}

} // namespace nn