#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <string>
#include "core/tensor.h"

// MNIST 数据加载器类
class MnistLoader {
private:
    // 将大端字节序的32位整数转换为主机字节序
    static uint32_t swap_endian(uint32_t val) {
        val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0x00FF00FF);
        return (val << 16) | (val >> 16);
    }

    // 读取图像数据
    static std::shared_ptr<Tensor> load_images(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("无法打开文件: " + path);
        }

        uint32_t magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;

        file.read(reinterpret_cast<char*>(&magic_number), 4);
        magic_number = swap_endian(magic_number);

        file.read(reinterpret_cast<char*>(&num_images), 4);
        num_images = swap_endian(num_images);

        file.read(reinterpret_cast<char*>(&num_rows), 4);
        num_rows = swap_endian(num_rows);

        file.read(reinterpret_cast<char*>(&num_cols), 4);
        num_cols = swap_endian(num_cols);
        
        std::cout << "图像: " << num_images 
                  << ", 尺寸: " << num_rows << "x" << num_cols << std::endl;

        size_t image_size = num_rows * num_cols;
        std::vector<float> image_data(num_images * image_size);

        for (size_t i = 0; i < num_images; ++i) {
            std::vector<unsigned char> buffer(image_size);
            file.read(reinterpret_cast<char*>(buffer.data()), image_size);
            for (size_t j = 0; j < image_size; ++j) {
                // 将像素值归一化到 [0, 1] 范围
                image_data[i * image_size + j] = buffer[j] / 255.0f;
            }
        }
        
        // --- 修改点 ---
        // 使用静态工厂方法创建Tensor，这是推荐的方式
        return Tensor::create(image_data, {num_images, image_size});
    }

    // 读取标签数据
    static std::shared_ptr<Tensor> load_labels(const std::string& path, size_t num_classes = 10) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("无法打开文件: " + path);
        }

        uint32_t magic_number = 0, num_labels = 0;

        file.read(reinterpret_cast<char*>(&magic_number), 4);
        magic_number = swap_endian(magic_number);

        file.read(reinterpret_cast<char*>(&num_labels), 4);
        num_labels = swap_endian(num_labels);
        
        std::cout << "标签数量: " << num_labels << std::endl;

        std::vector<float> label_data(num_labels * num_classes, 0.0f);
        std::vector<unsigned char> buffer(num_labels);
        file.read(reinterpret_cast<char*>(buffer.data()), num_labels);

        // 将标签转换为独热编码 (one-hot)
        for (size_t i = 0; i < num_labels; ++i) {
            label_data[i * num_classes + buffer[i]] = 1.0f;
        }
        
        // --- 修改点 ---
        // 使用静态工厂方法创建Tensor
        return Tensor::create(label_data, {num_labels, num_classes});
    }

public:
    // 公共接口，用于加载整个数据集
    static void load(const std::string& data_path,
                     std::shared_ptr<Tensor>& X_train,
                     std::shared_ptr<Tensor>& y_train,
                     std::shared_ptr<Tensor>& X_test,
                     std::shared_ptr<Tensor>& y_test) {
        
        std::cout << "正在从路径加载 MNIST 数据集: " << data_path << std::endl;
        
        X_train = load_images(data_path + "/train-images.idx3-ubyte");
        y_train = load_labels(data_path + "/train-labels.idx1-ubyte");
        X_test = load_images(data_path + "/t10k-images.idx3-ubyte");
        y_test = load_labels(data_path + "/t10k-labels.idx1-ubyte");

        std::cout << "MNIST 数据加载完成。" << std::endl;
    }
};

#endif // MNIST_LOADER_H