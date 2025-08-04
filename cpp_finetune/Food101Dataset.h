#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <fstream>
#include <iostream>

class Food101Dataset : public torch::data::Dataset<Food101Dataset> {
private:
    std::vector<std::tuple<std::string, int64_t>> image_info_;
    std::string root_path_;
    torch::Tensor mean_, std_;

public:
    explicit Food101Dataset(const std::string& root_path, const std::string& split = "train")
        : root_path_(root_path) {

        // 1. Create a map from class name to label index
        std::map<std::string, int64_t> class_to_idx;
        std::ifstream classes_file(root_path_ + "/meta/classes.txt");
        std::string class_name;
        int64_t idx = 0;
        while (std::getline(classes_file, class_name)) {
            class_to_idx[class_name] = idx++;
        }

        // 2. Read the train.txt or test.txt file
        std::ifstream split_file(root_path_ + "/meta/" + split + ".txt");
        std::string line;
        while (std::getline(split_file, line)) {
            // Line format is "class_name/image_id"
            size_t slash_pos = line.find('/');
            std::string current_class = line.substr(0, slash_pos);
            std::string image_id = line.substr(slash_pos + 1);

            // Construct full path and get label
            std::string image_path = root_path_ + "/images/" + current_class + "/" + image_id + ".jpg";
            int64_t label = class_to_idx[current_class];
            image_info_.emplace_back(image_path, label);
        }

        // 3. Define normalization constants
        mean_ = torch::tensor({0.485, 0.456, 0.406}).view({3, 1, 1});
        std_ = torch::tensor({0.229, 0.224, 0.225}).view({3, 1, 1});
    }

    torch::data::Example<> get(size_t index) override {
        std::string image_path = std::get<0>(image_info_[index]);
        int64_t label = std::get<1>(image_info_[index]);

        cv::Mat image = cv::imread(image_path);
        // Note: C++ version uses simple resize for simplicity.
        // A full implementation of RandomResizedCrop is more complex.
        cv::resize(image, image, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        auto image_tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kByte);
        image_tensor = image_tensor.permute({2, 0, 1}); // HWC to CHW
        image_tensor = image_tensor.to(torch::kFloat32).div(255.0);
        image_tensor = image_tensor.sub(mean_).div(std_);

        auto label_tensor = torch::tensor(label, torch::kInt64);

        return {image_tensor, label_tensor};
    }

    torch::optional<size_t> size() const override {
        return image_info_.size();
    }
};
