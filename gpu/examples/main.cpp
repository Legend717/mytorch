#include <iostream>
#include <memory>
#include <vector>
#include <cuda_runtime.h>
#include <random>
#include <algorithm>

#include "core/tensor.h"
#include "nn/module.h"
#include "nn/linear.h"
#include "nn/activations.h"
#include "nn/sequential.h"
#include "nn/conv.h"
#include "nn/pool.h"
#include "nn/flatten.h"
#include "optim/sgd.h"

// 辅助函数：均方误差损失 (保持不变)
std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target) {
    auto diff = pred->sub(target);
    auto sq_diff = diff->mul(diff);
    return sq_diff->sum();
}

// ✨ 新增：辅助函数，用于创建有特征模式的图像数据
void generate_pattern_data(std::vector<float>& data, int class_id, size_t channels, size_t height, size_t width) {
    size_t image_size = channels * height * width;
    data.resize(image_size);
    std::fill(data.begin(), data.end(), 0.0f); // 背景设为0

    // 根据类别ID创建不同的模式
    // 例如：类别0在左上角有高亮，类别1在右下角
    size_t pattern_size = height / 4;
    size_t start_h = (class_id == 0) ? height / 4 : height / 2;
    size_t start_w = (class_id == 0) ? width / 4 : width / 2;

    for(size_t c = 0; c < channels; ++c) {
        for(size_t h = start_h; h < start_h + pattern_size; ++h) {
            for(size_t w = start_w; w < start_w + pattern_size; ++w) {
                data[c * (height * width) + h * width + w] = 1.0f; // 特征区域设为1
            }
        }
    }
}


int main() {
    // --- 1. 选择设备 ---
    Device device = Device::CUDA;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        device = Device::CUDA;
        std::cout << "--- MiniTorchGPU: Using CUDA device ---" << std::endl;
    } else {
        std::cout << "--- MiniTorchGPU: Using CPU device ---" <<std::endl;
    }

    // --- 2. 定义超参数 ---
    const size_t BATCH_SIZE = 4; // 建议为偶数，方便下面生成数据
    const size_t IN_CHANNELS = 1;
    const size_t IMG_HEIGHT = 28;
    const size_t IMG_WIDTH = 28;
    const size_t NUM_CLASSES = 10; // 即使只用2类，也保持10以测试模型输出维度
    const float LEARNING_RATE = 0.001f;
    const int EPOCHS = 100; // 增加训练轮数以观察效果

    // --- 3. 定义CNN模型 (与之前相同) ---
    auto model = std::make_shared<nn::Sequential>(
        std::vector<std::shared_ptr<nn::Module>>{
            std::make_shared<nn::Conv2D>(IN_CHANNELS, 6, 5),
            std::make_shared<nn::ReLU>(),
            std::make_shared<nn::MaxPool2D>(2, 2),
            std::make_shared<nn::Conv2D>(6, 16, 5),
            std::make_shared<nn::ReLU>(),
            std::make_shared<nn::MaxPool2D>(2, 2),
            std::make_shared<nn::Flatten>(),
            std::make_shared<nn::Linear>(16 * 4 * 4, 120),
            std::make_shared<nn::ReLU>(),
            std::make_shared<nn::Linear>(120, 84),
            std::make_shared<nn::ReLU>(),
            std::make_shared<nn::Linear>(84, NUM_CLASSES)
        }
    );

    // --- 4. 将模型移动到目标设备 ---
    model->to(device);
    std::cout << "Model moved to target device." << std::endl;

    // --- 5. 创建优化器 ---
    optim::SGD optimizer(model->parameters(), LEARNING_RATE);

    // --- 6. ✨ 创建有意义的虚拟数据 ---
    std::vector<float> x_batch_data;
    std::vector<float> y_batch_data;

    for(size_t i = 0; i < BATCH_SIZE; ++i) {
        // 生成一半的样本为类别0，一半为类别1
        int class_id = (i < BATCH_SIZE / 2) ? 0 : 1;

        // 生成带特征的图像数据
        std::vector<float> image_data;
        generate_pattern_data(image_data, class_id, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH);
        x_batch_data.insert(x_batch_data.end(), image_data.begin(), image_data.end());

        // 生成对应的One-Hot标签
        std::vector<float> label_data(NUM_CLASSES, 0.0f);
        label_data[class_id] = 1.0f;
        y_batch_data.insert(y_batch_data.end(), label_data.begin(), label_data.end());
    }

    auto X_train_cpu = Tensor::create(x_batch_data, {BATCH_SIZE, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH});
    auto y_train_cpu = Tensor::create(y_batch_data, {BATCH_SIZE, NUM_CLASSES});

    auto X_train = X_train_cpu->to(device);
    auto y_train = y_train_cpu->to(device);
    std::cout << "Data moved to target device." << std::endl;

    std::cout << "Training for " << EPOCHS << " epochs..." << std::endl;

    // --- 7. 训练循环 ---
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        optimizer.zero_grad();

        auto y_pred = model->forward(X_train);
        auto loss = mse_loss(y_pred, y_train);
        loss->backward();
        optimizer.step();

        if ((epoch + 1) % 10 == 0) {
            std::cout << "Epoch [" << epoch + 1 << "/" << EPOCHS << "], Loss: " << loss->item() << std::endl;
        }
    }

    std::cout << "--- Training Finished ---" << std::endl;

    // --- 8. (可选) 检查最终预测结果 ---
    std::cout << "--- Final Predictions ---" << std::endl;
    auto final_pred = model->forward(X_train);
    auto final_pred_cpu = final_pred->data_cpu();
    auto y_train_cpu_data = y_train->data_cpu();

    for(size_t i=0; i<BATCH_SIZE; ++i){
        std::cout << "Sample " << i << ": ";
        float max_pred = -1e9;
        int pred_idx = -1;
        int true_idx = -1;
        for(size_t j=0; j<NUM_CLASSES; ++j){
            if(final_pred_cpu[i*NUM_CLASSES + j] > max_pred){
                max_pred = final_pred_cpu[i*NUM_CLASSES + j];
                pred_idx = j;
            }
            if(y_train_cpu_data[i*NUM_CLASSES + j] == 1.0f){
                true_idx = j;
            }
        }
        std::cout << "Predicted class: " << pred_idx << ", True class: " << true_idx << std::endl;
    }


    return 0;
}