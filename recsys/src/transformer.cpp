#include <cmath>
#include <torch/torch.h>
#include "transformer.h"



    MultiHeadAttentionImpl::MultiHeadAttentionImpl(int embed_dim,int num_head):embed_dim(embed_dim),num_head(num_head){
        head_dim = embed_dim / num_head;
        w_q = register_module("w_q", torch::nn::Linear(embed_dim, embed_dim));
        w_k = register_module("w_k", torch::nn::Linear(embed_dim, embed_dim));
        w_v = register_module("w_v", torch::nn::Linear(embed_dim, embed_dim));
        w_o = register_module("w_o", torch::nn::Linear(embed_dim, embed_dim));
    }

    torch::Tensor MultiHeadAttentionImpl::forward(torch::Tensor x, torch::Tensor attn_mask){
        auto B = x.size(0);
        auto T = x.size(1);
        auto Q = w_q(x);
        auto K = w_k(x);
        auto V = w_v(x);
        Q = Q.view({B,T,num_head,head_dim}).transpose(1,2);
        K = K.view({B,T,num_head,head_dim}).transpose(1,2);
        V = V.view({B,T,num_head,head_dim}).transpose(1,2);

        auto scores = torch::matmul(Q,K.transpose(-2,-1)) / std::sqrt((float)head_dim);

        if (attn_mask.defined()) {
            scores = scores.masked_fill(attn_mask, -1e9);
        }

        auto attn = torch::softmax(scores, -1);
        auto context = torch::matmul(attn, V);

        context = context.transpose(1,2).contiguous().view({B, T, embed_dim});
        return w_o(context);
    }

    FeedForwardImpl::FeedForwardImpl(int embed_dim, int ff_dim) {
        w1 = register_module("w1", torch::nn::Linear(embed_dim, ff_dim));
        w2 = register_module("w2", torch::nn::Linear(embed_dim, ff_dim));
        w3 = register_module("w3", torch::nn::Linear(ff_dim, embed_dim));
    }

    torch::Tensor FeedForwardImpl::forward(torch::Tensor x) {
        auto gate = torch::silu(w1->forward(x));
        auto up = w2->forward(x);
        return w3->forward(gate * up);
    }

    LiGRImpl::LiGRImpl(int embed_dim, int heads, int ff_dim, float dropout_rate) : dropout_rate(dropout_rate) {
        mha = register_module("mha", MultiHeadAttention(embed_dim, heads));
        ffn = register_module("ffn", FeedForward(embed_dim, ff_dim));
        norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));
        norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));

        gate_attn = register_module("gate_attn", torch::nn::Linear(embed_dim, 1));
        gate_ffn = register_module("gate_ffn", torch::nn::Linear(embed_dim, 1));
    }

    torch::Tensor LiGRImpl::forward(torch::Tensor x, torch::Tensor attn_mask) {
        auto h_attn_input = x;

        x = norm1(x);
        x = mha->forward(x, attn_mask);
        x = torch::dropout(x, dropout_rate, is_training());

        auto gate_attn_values = torch::sigmoid(gate_attn(h_attn_input));
        x = h_attn_input + gate_attn_values * x;

        auto h_ffn_input = x;

        x = norm2(x);
        x = ffn->forward(x);
        x = torch::dropout(x, dropout_rate, is_training());

        auto gate_ffn_values = torch::sigmoid(gate_ffn(h_ffn_input));
        x = h_ffn_input + gate_ffn_values * x;

        return x;
    }