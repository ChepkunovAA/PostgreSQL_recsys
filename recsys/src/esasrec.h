#ifndef ESASREC_H
#define ESASREC_H

#include <iostream>
#include <cmath>
#include <torch/torch.h>

struct TopKResult {
    torch::Tensor indices;
    torch::Tensor values;
};

struct eSASRecImpl : torch::nn::Module {
    int num_items;
    int max_seq_len;
    int embed_dim;
    int num_layers;
    int num_heads;
    int ff_dim;
    float dropout_rate;
    
    torch::nn::Embedding item_embedding{nullptr};
    torch::nn::Embedding pos_embedding{nullptr};
    torch::nn::Dropout embed_dropout{nullptr};

    torch::nn::ModuleList ligr_blocks;
    torch::nn::LayerNorm final_norm{nullptr};

    void set_up(
        int num_items,
        int max_seq_len,
        int embed_dim = 128,
        int num_layers = 2,
        int num_heads = 4,
        int ff_dim = 512,
        float dropout_rate = 0.2
    );

    torch::Tensor get_item_embeddings() const;
    
    torch::Tensor create_attention_mask(torch::Tensor& seq);

    torch::Tensor forward(torch::Tensor& seq);
    
    TopKResult predict_next(torch::Tensor& seq, int k);
};
TORCH_MODULE(eSASRec);

#endif // ESASREC_H