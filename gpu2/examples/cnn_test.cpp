#include <iostream>
#include <memory>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <stdexcept>

#include <cuda_runtime.h>

#include "core/tensor.h"
#include "nn/module.h"
#include "nn/linear.h"
#include "nn/activations.h"
#include "nn/sequential.h"
#include "nn/flatten.h" // 引入 Flatten 层
#include "optim/sgd.h"
#include "loader/mnist_loader.h"
#include "nn/conv.h"
#include "nn/pool.h"

// 辅助函数：计算准确率
// 将GPU上的Tensor传回CPU计算
float calculate_accuracy(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target) {
    // 安全地将数据从GPU拷贝到CPU
    auto pred_data = pred->data_cpu();
    auto target_data = target->data_cpu();

    const auto& shape = pred->shape();
    size_t batch_size = shape[0];
    size_t num_classes = shape[1];
    int correct_count = 0;

    for (size_t i = 0; i < batch_size; ++i) {
        // 找到预测概率最高的类别索引
        auto pred_start = pred_data.begin() + i * num_classes;
        auto pred_end = pred_start + num_classes;
        size_t pred_label = std::distance(pred_start, std::max_element(pred_start, pred_end));

        // 找到真实标签的类别索引 (one-hot)
        auto target_start = target_data.begin() + i * num_classes;
        auto target_end = target_start + num_classes;
        size_t true_label = std::distance(target_start, std::max_element(target_start, target_end));

        if (pred_label == true_label) {
            correct_count++;
        }
    }
    return static_cast<float>(correct_count) / batch_size;
}

// 辅助函数：均方误差损失
// 确保所有操作都在同一设备上
std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target) {
    auto diff = pred->sub(target);
    auto sq_diff = diff->mul(diff);
    auto sum_loss = sq_diff->sum();

    // 创建一个标量张量用于缩放，并移动到与pred相同的设备
    float divisor = 1.0f / pred->shape()[0]; // 按批次大小取平均
    auto scalar = Tensor::create({divisor}, {1}, false)->to(pred->device());

    return sum_loss->mul(scalar); // 使用乘法代替未实现的除法
}

int main() {
    //设置设备
    Device device = Device::CPU;
    int device_count = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    if (cuda_err == cudaSuccess && device_count > 0) {
        device = Device::CUDA;
        std::cout << "--- MiniTorchGPU MNIST 训练 (使用 CUDA) ---" << std::endl;
    } else {
        std::cout << "--- MiniTorchGPU MNIST 训练 (使用 CPU) ---" << std::endl;
    }

    //超参数
    const size_t BATCH_SIZE = 64;
    const size_t IN_CHANNELS = 1; // 灰度图
    const size_t IMG_HEIGHT = 28;
    const size_t IMG_WIDTH = 28;
    const size_t NUM_CLASSES = 10;
    const float LEARNING_RATE = 0.001f;
    const int EPOCHS = 10;
    const std::string MNIST_DATA_PATH = "../data";

    //定义模型
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

    //将模型所有参数移动到目标设备
    model->to(device);
    std::cout << "模型成功加载到设备" << std::endl;

    //创建优化器
    optim::SGD optimizer(model->parameters(), LEARNING_RATE);

    //加载MNIST数据(初始加载到CPU)
    std::shared_ptr<Tensor> X_train_cpu, y_train_cpu, X_test_cpu, y_test_cpu;
    try {
        MnistLoader::load(MNIST_DATA_PATH, X_train_cpu, y_train_cpu, X_test_cpu, y_test_cpu);
    } catch (const std::runtime_error& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        std::cerr << "请确保MNIST数据文件已解压并放置在 " << MNIST_DATA_PATH << " 目录中。" << std::endl;
        return 1;
    }

    // ----------- 新增的关键步骤 -----------
    // 将数据重塑为CNN所需的4D张量: (N, C, H, W)
    X_train_cpu = X_train_cpu->reshape({X_train_cpu->shape()[0], IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH});
    X_test_cpu = X_test_cpu->reshape({X_test_cpu->shape()[0], IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH});
    std::cout << "数据已被重塑为4D张量，以适配CNN模型。" << std::endl;
    // ------------------------------------

    // 仅将较小的测试集移动到GPU
    auto X_test = X_test_cpu->to(device);
    auto y_test = y_test_cpu->to(device);
    std::cout << "数据已加载到CPU, 测试数据已移动到目标设备" << std::endl;

    size_t num_samples = X_train_cpu->shape()[0];
    size_t num_batches = num_samples / BATCH_SIZE;

    //用于打乱数据的随机数生成器
    std::random_device rd;
    std::mt19937 g(rd());

    std::cout << "开始训练 " << EPOCHS << " 个周期..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    //训练
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::vector<size_t> indices(num_samples);
        std::iota(indices.begin(), indices.end(), 0);
        // std::shuffle(indices.begin(), indices.end(), g); // 如果需要随机化，请取消注释

        float epoch_loss = 0;

        for (size_t i = 0; i < num_batches; ++i) {
            optimizer.zero_grad();

            // 1. 从CPU上的4D完整数据中切片出小批量
            auto x_batch_cpu = X_train_cpu->slice(i * BATCH_SIZE, (i + 1) * BATCH_SIZE);
            auto y_batch_cpu = y_train_cpu->slice(i * BATCH_SIZE, (i + 1) * BATCH_SIZE);

            // 2. **仅将当前批次的数据移动到目标设备**
            auto x_batch = x_batch_cpu->to(device);
            auto y_batch = y_batch_cpu->to(device);

            std::cout<<1<<std::endl;
            // 前向传播 (现在输入形状正确了)
            auto y_pred = model->forward(x_batch);
            std::cout<<2<<std::endl;
            // 计算损失
            auto loss = mse_loss(y_pred, y_batch);
            epoch_loss += loss->item();

            // 反向传播
            loss->backward();
            std::cout<<3<<std::endl;
            // 更新权重
            optimizer.step();
            std::cout<<4<<std::endl;
        }

        // 在每个周期结束后，在测试集上评估模型
        auto y_pred_test = model->forward(X_test);
        float test_accuracy = calculate_accuracy(y_pred_test, y_test);

        std::cout << "周期 [" << epoch + 1 << "/" << EPOCHS
                  << "], 平均损失: " << epoch_loss / num_batches
                  << ", 测试集准确率: " << test_accuracy * 100.0f << "%" << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "--- 训练完成 ---" << std::endl;
    std::cout << "总用时: " << duration << " ms" << std::endl;

    return 0;
}