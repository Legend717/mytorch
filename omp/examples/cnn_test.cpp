#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <numeric>

#include "core/tensor.h"
#include "nn/module.h"
#include "nn/linear.h"
#include "nn/activations.h"
#include "nn/sequential.h"
#include "nn/conv.h"
#include "nn/pool.h"
#include "nn/flatten.h"
#include "optim/sgd.h"
#include "loader/mnist_loader.h"

// 辅助函数：均方误差损失
std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target) {
    auto diff = pred->sub(target);
    auto sq_diff = diff->mul(diff);
    return sq_diff->sum()->div(Tensor::create({(float)pred->shape()[0]},{1})); // 按批次大小取平均
}

float calculate_accuracy(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target);


int main() {
    std::cout << "--- MiniTorchCPU CNN Training Example ---" << std::endl;

    // 1. 定义超参数
    const size_t BATCH_SIZE = 512;
    const size_t IN_CHANNELS = 1; // 灰度图
    const size_t IMG_HEIGHT = 28;
    const size_t IMG_WIDTH = 28;
    const size_t NUM_CLASSES = 10;
    const float LEARNING_RATE = 0.001f;
    const int EPOCHS = 10;
    const std::string MNIST_DATA_PATH = "../data"; 

    // 2. 定义CNN模型 (类LeNet结构)
    auto model = std::make_shared<nn::Sequential>(
        std::vector<std::shared_ptr<nn::Module>>{
            // 输入: [B, 1, 28, 28]
            std::make_shared<nn::Conv2D>(IN_CHANNELS, 6, 5), // -> [B, 6, 24, 24]
            std::make_shared<nn::ReLU>(),
            std::make_shared<nn::MaxPool2D>(2, 2),          // -> [B, 6, 12, 12]
            
            std::make_shared<nn::Conv2D>(6, 16, 5),         // -> [B, 16, 8, 8]
            std::make_shared<nn::ReLU>(),
            std::make_shared<nn::MaxPool2D>(2, 2),          // -> [B, 16, 4, 4]

            std::make_shared<nn::Flatten>(),                // -> [B, 16*4*4] = [B, 256]

            std::make_shared<nn::Linear>(16 * 4 * 4, 120),  // -> [B, 120]
            std::make_shared<nn::ReLU>(),
            std::make_shared<nn::Linear>(120, 84),          // -> [B, 84]
            std::make_shared<nn::ReLU>(),
            std::make_shared<nn::Linear>(84, NUM_CLASSES)   // -> [B, 10]
        }
    );

    // 创建优化器
    optim::SGD optimizer(model->parameters(), LEARNING_RATE);

    // 加载MNIST数据
    std::shared_ptr<Tensor> X_train, y_train, X_test, y_test;
    try {
        MnistLoader::load(MNIST_DATA_PATH, X_train, y_train, X_test, y_test);
    } catch (const std::runtime_error& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        std::cerr << "MNIST数据文件解压并放置在" << MNIST_DATA_PATH << " 目录中。" << std::endl;
        return 1;
    }

    size_t num_samples = X_train->shape()[0];
    size_t num_batches = num_samples / BATCH_SIZE;

    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "开始训练" << EPOCHS << "个周期..." << std::endl;

    // 训练循环
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // 每个周期开始时打乱数据
        std::vector<size_t> indices(num_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), MyRand::global_rand_generater);

        float epoch_loss = 0;
        float epoch_accuracy = 0;

        for (size_t i = 0; i < num_batches; ++i) {
            optimizer.zero_grad();

            // 创建一个小批量
            auto x_batch = X_train->slice(i * BATCH_SIZE, (i + 1) * BATCH_SIZE);
            auto y_batch = y_train->slice(i * BATCH_SIZE, (i + 1) * BATCH_SIZE);
            
            x_batch = x_batch->reshape({BATCH_SIZE, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH});
            
            // 前向传播
            auto y_pred = model->forward(x_batch);

            // 计算损失
            auto loss = mse_loss(y_pred, y_batch);
            epoch_loss += loss->item();

            epoch_accuracy += calculate_accuracy(y_pred, y_batch);

            // 反向传播
            loss->backward();

            // 更新权重
            optimizer.step();
        }

        std::cout << "周期 [" << epoch + 1 << "/" << EPOCHS << "], 平均损失: " << epoch_loss / num_batches <<std::endl;
        std::cout << "训练集准确率: " << (epoch_accuracy / num_batches) * 100.0f << "%" << std::endl;
    }

    std::cout << "--- 训练完成 ---" << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::seconds>(end -start).count();
    std::cout<<"用时："<<time<<"s"<<std::endl;

    return 0;
}

float calculate_accuracy(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target) {
    const auto& pred_data = pred->data();
    const auto& target_data = target->data();
    const auto& shape = pred->shape();

    size_t batch_size = shape[0];
    size_t num_classes = shape[1];
    int correct_count = 0;

    for (size_t i = 0; i < batch_size; ++i) {
        //找到预测概率最高的类别索引
        auto pred_start = pred_data.begin() + i * num_classes;
        auto pred_end = pred_start + num_classes;
        size_t pred_label = std::distance(pred_start, std::max_element(pred_start, pred_end));

        //找到真实标签的类别索引
        auto target_start = target_data.begin() + i * num_classes;
        auto target_end = target_start + num_classes;
        size_t true_label = std::distance(target_start, std::max_element(target_start, target_end));

        if (pred_label == true_label) {
            correct_count++;
        }
    }
    return static_cast<float>(correct_count) / batch_size;
}