#include "Food101Dataset.h"
#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

int main() {
    // --- 1. Configuration ---
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using device: CUDA" << std::endl;
    }

    const std::string DATA_ROOT = "../food-101"; // Relative path to data
    const std::string MODEL_PATH = "../resnet50_for_food101_finetuning.pt";
    const int NUM_EPOCHS = 3;
    const int BATCH_SIZE = 64;
    const double LEARNING_RATE = 0.001;

    // --- 2. Data Loading ---
    auto train_dataset = Food101Dataset(DATA_ROOT, "train")
        .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(8));

    size_t dataset_size = train_dataset.size().value();
    std::cout << "Training data loaded: " << dataset_size << " images." << std::endl;

    // --- 3. Model Loading and Setup ---
    torch::jit::script::Module model = torch::jit::load(MODEL_PATH);

    std::vector<torch::Tensor> params_to_update;
    for (auto& param : model.named_parameters()) {
        if (param.name.find("fc") != std::string::npos) {
            param.value.set_requires_grad(true);
            params_to_update.push_back(param.value);
        } else {
            param.value.set_requires_grad(false);
        }
    }
    std::cout << "Number of parameter groups to fine-tune: " << params_to_update.size() << std::endl;
    model.to(device);

    // --- 4. Training Loop ---
    torch::optim::Adam optimizer(params_to_update, torch::optim::AdamOptions(LEARNING_RATE));

    std::cout << "\nStarting C++ fine-tuning on Food-101..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    model.train();

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        double running_loss = 0.0;
        int64_t batch_idx = 0;

        for (auto& batch : *train_loader) {
            auto inputs = batch.data.to(device, true); // non_blocking
            auto labels = batch.target.to(device, true);

            optimizer.zero_grad();
            auto outputs = model.forward({inputs}).toTensor();
            auto loss = torch::cross_entropy_loss(outputs, labels);

            loss.backward();
            optimizer.step();

            running_loss += loss.item<double>() * inputs.size(0);

            if (++batch_idx % 100 == 0) {
                std::cout << "  Epoch [" << epoch + 1 << "/" << NUM_EPOCHS
                          << "], Batch [" << batch_idx << "/" << (dataset_size/BATCH_SIZE)
                          << "], Loss: " << loss.item<double>() << std::endl;
            }
        }
        double epoch_loss = running_loss / dataset_size;
        std::cout << "Epoch " << epoch + 1 << " Summary -> Loss: " << epoch_loss << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "\nC++ fine-tuning finished in " << elapsed.count() << " seconds." << std::endl;
    model.save("food101_finetuned_from_cpp.pt");

    return 0;
}
