#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <xtorch/xtorch.h>


int main() {

    const std::string DATA_ROOT = "../food-101"; // Relative path to data
    const std::string MODEL_PATH = "./resnet50_for_food101_finetuning.pt";
    const int NUM_EPOCHS = 3;
    const int BATCH_SIZE = 64;
    const double LEARNING_RATE = 0.001;

    std::cout.precision(10);
    int epochs = 1;

    std::vector<std::shared_ptr<xt::Module>> transform_list;
    transform_list.push_back(std::make_shared<xt::transforms::image::Resize>(std::vector<int64_t>{224, 224}));
    transform_list.push_back(
        std::make_shared<xt::transforms::general::Normalize>(std::vector<float>{0.5, 0.5, 0.5},
                                                             std::vector<float>{0.5, 0.5, 0.5}));
    auto compose = std::make_unique<xt::transforms::Compose>(transform_list);
    auto dataset = xt::datasets::Food101("/home/kami/Documents/datasets/", xt::datasets::DataMode::TRAIN, false,
                                         std::move(compose));
    xt::dataloaders::ExtendedDataLoader data_loader(dataset, BATCH_SIZE, true, 16, 2);
    torch::Device device(torch::kCUDA);

    // --- 3. Model Loading and Setup ---
    torch::jit::script::Module model = torch::jit::load(MODEL_PATH);

    std::vector<torch::Tensor> params_to_update;
    for (auto param : model.named_parameters()) {
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

        for (auto& batch : data_loader) {
            auto inputs = batch.first.to(device, true); // non_blocking
            auto labels = batch.second.to(device, true);

            optimizer.zero_grad();
            auto outputs = model.forward({inputs}).toTensor();
            auto loss = torch::cross_entropy_loss(outputs, labels);

            loss.backward();
            optimizer.step();

            running_loss += loss.item<double>() * inputs.size(0);

            if (++batch_idx % 100 == 0) {
                cout << "Batch: " << batch_idx << " Loss:" << loss.item() << endl;
            }
        }
        //double epoch_loss = running_loss / dataset_size;
        //std::cout << "Epoch " << epoch + 1 << " Summary -> Loss: " << epoch_loss << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "\nC++ fine-tuning finished in " << elapsed.count() << " seconds." << std::endl;
    model.save("food101_finetuned_from_cpp.pt");

    return 0;
}
