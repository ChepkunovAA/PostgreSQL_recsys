// model_manager.h
#ifndef MODEL_MANAGER_H
#define MODEL_MANAGER_H

#include <string>
#include <unordered_map>
#include <memory>
#include <torch/torch.h>
#include <mutex>
#include "esasrec.h"

class ModelManager {
  public:
    static ModelManager* getInstance();

    // Создание
    bool create_model(    
        int model_id,
        int num_items,
        int max_seq_len,
        int embed_dim,
        int num_layers,
        int num_heads,
        int ff_dim,
        float dropout_rate
    );

    // Обучение новой модели
    void train_model(
        int model_id,
        torch::Tensor train_data,
        int epochs,
        int batch_size,
        float learning_rate
    );

    // Загрузка модели с диска
    bool load_model(int model_id);

    // Сохранение модели на диск
    bool save_model(int model_id);

    // Сохранить все загруженные модели
    void save_all();

    // Получить указатель на модель
    eSASRec get_model(int model_id);

    // Проверить, загружена ли модель
    bool has_model(int model_id);

    // Выгрузить модель из памяти
    bool unload_model(int model_id);

    // Удалить модель с диска
    bool delete_model(int model_id);

    // Предсказание следующего элемента
    TopKResult predict(
        int model_id,
        torch::Tensor sequence,
        int top_k = 10
    );

    torch::Tensor get_embeddings(int model_id);

  private:
    ModelManager();
    ~ModelManager();

    static ModelManager* instance;

    std::unordered_map<int, eSASRec> models_;

    torch::Device device = torch::Device("cpu");

    // Путь для сохранения моделей
    std::string get_model_path(int model_id) const;
};

extern "C" {
    ModelManager* get_model_manager_instance();
}

#endif // MODEL_MANAGER_H