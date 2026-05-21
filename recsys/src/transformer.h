#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <cmath>
#include <torch/torch.h>


struct MultiHeadAttentionImpl : torch::nn::Module{
    int embed_dim;
    int num_head;
    int head_dim;
    torch::nn::Linear w_q{nullptr},w_v{nullptr},w_k{nullptr},w_o{nullptr};

    MultiHeadAttentionImpl(int embed_dim,int num_head);

    torch::Tensor forward(torch::Tensor x, torch::Tensor attn_mask = {});
};
TORCH_MODULE(MultiHeadAttention);


struct FeedForwardImpl : torch::nn::Module {
    torch::nn::Linear w1{nullptr}, w2{nullptr}, w3{nullptr};

    FeedForwardImpl(int embed_dim, int ff_dim);

    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(FeedForward);

struct LiGRImpl : torch::nn::Module {
    MultiHeadAttention mha{nullptr};
    FeedForward ffn{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};

    torch::nn::Linear gate_attn{nullptr};
    torch::nn::Linear gate_ffn{nullptr};

    float dropout_rate;

    LiGRImpl(int embed_dim, int heads, int ff_dim, float dropout_rate);

    torch::Tensor forward(torch::Tensor x, torch::Tensor attn_mask = {});
};
TORCH_MODULE(LiGR);

#endif // TRANSFORMER_H