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

}

std::shared_ptr<Tensor> Conv2D::forward(std::shared_ptr<Tensor> input) {
    auto conv_func = std::make_shared<Conv2DFunc>(_stride, _padding);
    return conv_func->apply({input, _weight});
}


std::vector<std::shared_ptr<Tensor>> Conv2D::parameters() {
    return {_weight};
}

} // namespace nn