// model_manager.cpp
#include "manager.h"
#include "esasrec.h"
#include "transformer.h"
#include <torch/torch.h>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <random>

ModelManager* ModelManager::instance = nullptr;

ModelManager* ModelManager::getInstance() {
    if (instance == nullptr) {
        instance = new ModelManager();
    }
    return instance;
}

ModelManager::ModelManager() {
    if (torch::cuda::is_available()) {
        device = torch::Device("cuda");
    }
    std::filesystem::create_directories("saved_models");
}

ModelManager::~ModelManager() {
    save_all();
}

std::string ModelManager::get_model_path(int model_id) const {
    std::filesystem::path models_dir = "saved_models";
    if (!std::filesystem::exists(models_dir)) {
        std::filesystem::create_directories(models_dir);
    }
    return (models_dir / (std::to_string(model_id) + "_model.pt")).string();
}

bool ModelManager::create_model(    
    int model_id,
    int num_items,
    int max_seq_len,
    int embed_dim,
    int num_layers,
    int num_heads,
    int ff_dim,
    float dropout_rate
) {
    if (has_model(model_id)) {
        return false;
    }
    auto model = eSASRec();
    model->set_up(
        num_items, max_seq_len, embed_dim, num_layers, 
        num_heads, ff_dim, dropout_rate
    );
    model->to(device);
    models_[model_id] = model;
    return true;
}

void ModelManager::train_model(
    int model_id,
    torch::Tensor train_data,
    int epochs,
    int batch_size,
    float learning_rate
) {

    auto model = models_[model_id];
    model->train();
    
    torch::optim::Adam optimizer(
        model->parameters(), 
        torch::optim::AdamOptions(learning_rate)
    );
    
    train_data = train_data.to(device);
    int num_samples = train_data.size(0);
    int seq_len = train_data.size(1) - 1;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        torch::Tensor indices = torch::randperm(num_samples, device);
        for (int start = 0; start < num_samples; start += batch_size) {
            auto batch_indices = indices.slice(0, start, std::min(start + batch_size, num_samples));
            auto batch_data = train_data.index_select(0, batch_indices);

            auto input_seq = batch_data.slice(1, 0, seq_len);
            auto targets = batch_data.select(1, seq_len);
            optimizer.zero_grad();
            auto logits = model->forward(input_seq);
            auto last_logits = logits.select(1, -1);
            auto loss = torch::nn::functional::cross_entropy(
                last_logits, 
                targets, 
                torch::nn::functional::CrossEntropyFuncOptions().ignore_index(0)
            );
            loss.backward();
            optimizer.step();
        }
    }
}

bool ModelManager::load_model(int model_id) {
    std::string path = get_model_path(model_id);
    
    if (!std::filesystem::exists(path)) {
        return false;
    }

    eSASRec model;
    torch::load(model, path);
    models_[model_id] = model;
        
    return true;
}

bool ModelManager::save_model(int model_id) {
    
    if (!has_model(model_id)) {
        return false;
    }
    
    try {
        auto model = models_[model_id];
        torch::save(model, get_model_path(model_id));

        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

void ModelManager::save_all() {
    
    for (const auto& [model_id, entry] : models_) {
        save_model(model_id);
    }
}

eSASRec ModelManager::get_model(int model_id) {
    
    return models_.at(model_id);
}

bool ModelManager::has_model(int model_id) {
    
    return models_.find(model_id) != models_.end();
}

bool ModelManager::unload_model(int model_id) {
    
    if (!has_model(model_id)) {
        return false;
    }
    models_.erase(model_id);
    return true;
}

bool ModelManager::delete_model(int model_id) {
    std::string path = get_model_path(model_id);
    if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
        return true;
    }
    return false;
}

TopKResult ModelManager::predict(
    int model_id,
    torch::Tensor sequence,
    int top_k
) {
    
    auto& model = models_.at(model_id);
    
    model->eval();
    torch::NoGradGuard no_grad;
    
    sequence = sequence.to(device);
    return model->predict_next(sequence, top_k);
}

torch::Tensor ModelManager::get_embeddings(int model_id) {
    
    auto& model = models_.at(model_id);
    return model->get_item_embeddings().to(torch::kCPU);
}

extern "C" {
    ModelManager* get_model_manager_instance() {
        return ModelManager::getInstance();
    }
}