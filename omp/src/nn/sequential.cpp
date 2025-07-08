#include "nn/sequential.h"

namespace nn {

Sequential::Sequential(std::vector<std::shared_ptr<Module>> layers) : _layers(layers) {}

std::shared_ptr<Tensor> Sequential::forward(std::shared_ptr<Tensor> input) {
    auto current_input = input;
    for (auto& layer : _layers) {
        current_input = layer->forward(current_input);
    }
    return current_input;
}

std::vector<std::shared_ptr<Tensor>> Sequential::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto& layer : _layers) {
        auto layer_params = layer->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

} // namespace nn