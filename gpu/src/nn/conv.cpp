#include "nn/conv.h"
#include "core/function.h"
#include <cmath>
#include <random> // 需要包含random头文件

namespace nn {

Conv2D::Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding)
    : _stride(stride), _padding(padding) {

    // Kaiming He 初始化
    float stdv = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));
    size_t weight_size = out_channels * in_channels * kernel_size * kernel_size;

    // --- 修改: 更高效的初始化 ---
    // 1. 在CPU上创建随机数据
    std::vector<float> weight_data_cpu(weight_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    for(auto& val : weight_data_cpu) {
        val = d(gen) * stdv; // 直接应用缩放
    }
    
    // 2. 从CPU数据创建Tensor
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

// parameters 和 to 方法保持不变
std::vector<std::shared_ptr<Tensor>> Conv2D::parameters() {
    return {_weight, _bias};
}

void Conv2D::to(Device device) {
    if (_weight) _weight = _weight->to(device);
    if (_bias) _bias = _bias->to(device);
}

} // namespace nn