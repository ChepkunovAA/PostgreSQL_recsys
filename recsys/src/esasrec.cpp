#include <iostream>
#include <cmath>
#include <torch/torch.h>
#include "esasrec.h"
#include "transformer.h"
    
    void eSASRecImpl::set_up(
        int num_items,
        int max_seq_len,
        int embed_dim,
        int num_layers,
        int num_heads,
        int ff_dim,
        float dropout_rate
    ) {
        this->num_items = num_items;
        this->max_seq_len = max_seq_len;
        this->embed_dim = embed_dim;
        this->num_layers = num_layers;
        this->num_heads = num_heads;
        this->ff_dim = ff_dim;
        this->dropout_rate = dropout_rate;

        item_embedding = register_module("item_embedding", 
            torch::nn::Embedding(
                torch::nn::EmbeddingOptions(num_items + 1, embed_dim).padding_idx(0)
            )
        );
        pos_embedding = register_module("pos_embedding", 
            torch::nn::Embedding(max_seq_len + 1, embed_dim));
        embed_dropout = register_module("embed_dropout", 
            torch::nn::Dropout(dropout_rate));
        for (int i = 0; i < num_layers; i++) {
            ligr_blocks->push_back(LiGR(embed_dim, num_heads, ff_dim, dropout_rate));
        }
        register_module("ligr_blocks", ligr_blocks);
        
        final_norm = register_module("final_norm", 
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));

        for (auto& p : parameters()) {
            if (p.dim() > 1) {
                torch::nn::init::xavier_uniform_(p);
            }
        }
    }

    torch::Tensor eSASRecImpl::get_item_embeddings() const {
        return item_embedding->weight.slice(0, 1, num_items + 1).detach();
    }
    
    torch::Tensor eSASRecImpl::create_attention_mask(torch::Tensor& seq) {
        auto device = seq.device();
        auto T = seq.size(1);
        
        auto padding_mask = (seq == 0).unsqueeze(1).unsqueeze(2);
        auto causal_mask = torch::triu(torch::ones({T, T}), 1).to(torch::kBool).to(device).unsqueeze(0).unsqueeze(0);
        auto mask = padding_mask | causal_mask; 
        return mask;
    }

    torch::Tensor eSASRecImpl::forward(torch::Tensor& seq) {
        auto device = seq.device();
        auto B = seq.size(0);
        auto T = seq.size(1);
        auto positions = torch::arange(T, device)
            .unsqueeze(0)
            .repeat({B, 1});

        auto item_emb = item_embedding->forward(seq) 
            / std::sqrt(static_cast<double>(embed_dim));

        auto pos_emb = pos_embedding->forward(positions);

        auto x = item_emb + pos_emb;
        x = embed_dropout->forward(x);
        auto mask = create_attention_mask(seq);

        for (int i = 0; i < num_layers; i++) {
            x = ligr_blocks[i]->as<LiGR>()->forward(x, mask);
        }
        
        x = final_norm->forward(x);
        auto item_embeds = item_embedding->weight;
        auto logits = torch::matmul(x, item_embeds.transpose(0, 1));
        
        return logits;
    }
    
    TopKResult eSASRecImpl::predict_next(torch::Tensor& seq, int k) {
        if (seq.dim() == 1) {
            seq = seq.unsqueeze(0);
        }
        
        auto logits = forward(seq);
        auto last_logits = logits.select(1, -1);
        
        auto probs = torch::softmax(last_logits, -1);
        auto topk_result = torch::topk(probs, k, -1);

        auto values = std::get<0>(topk_result).squeeze(0);
        auto indices = std::get<1>(topk_result).squeeze(0);
        
        return {indices, values};
    }