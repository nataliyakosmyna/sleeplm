/// CoCa-style Attentional Pooler for SleepLM.
///
/// Python: `class AttnPooler(nn.Module)` in biosignals_coca_model.py
///
/// Uses learned query tokens to compress a variable-length sequence
/// into a fixed number of output tokens via multi-head cross-attention.
///
/// - contrastive_pooler (n_query=1)  → global CLS token for contrastive loss
/// - decoder_pooler (n_query=32)     → compressed tokens for text decoder

use burn::prelude::*;
use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;

/// Fused multi-head attention for pooling (query ≠ key/value).
#[derive(Module, Debug)]
pub struct AttnPooler<B: Backend> {
    /// Learned query tokens: [1, n_query, dim].
    pub query_tokens: Param<Tensor<B, 3>>,
    /// Fused QKV projection for cross-attention.
    /// We use separate Q, K, V projections for clarity.
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub n_query: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> AttnPooler<B> {
    pub fn new(dim: usize, num_heads: usize, n_query: usize, device: &B::Device) -> Self {
        let head_dim = dim / num_heads;
        Self {
            query_tokens: Param::initialized(
                burn::module::ParamId::new(),
                Tensor::zeros([1, n_query, dim], device),
            ),
            // nn.MultiheadAttention uses a single in_proj_weight [3*D, D]
            // We model it as separate projections for weight loading compatibility
            q_proj: LinearConfig::new(dim, dim).with_bias(true).init(device),
            k_proj: LinearConfig::new(dim, dim).with_bias(true).init(device),
            v_proj: LinearConfig::new(dim, dim).with_bias(true).init(device),
            out_proj: LinearConfig::new(dim, dim).with_bias(true).init(device),
            n_query,
            num_heads,
            head_dim,
        }
    }

    /// x_seq: [B, L, D] — encoder features
    /// Returns: [B, n_query, D]
    pub fn forward(&self, x_seq: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, l, _d] = x_seq.dims();
        let (h, dh) = (self.num_heads, self.head_dim);
        let nq = self.n_query;

        // Expand query tokens: [1, nq, D] → [B, nq, D]
        let q_input = self.query_tokens.val().expand([b, nq, h * dh]);

        let q = self.q_proj.forward(q_input).reshape([b, nq, h, dh]).swap_dims(1, 2);
        let k = self.k_proj.forward(x_seq.clone()).reshape([b, l, h, dh]).swap_dims(1, 2);
        let v = self.v_proj.forward(x_seq).reshape([b, l, h, dh]).swap_dims(1, 2);

        let scale = (dh as f64).powf(-0.5) as f32;
        let attn = softmax(q.matmul(k.transpose()).mul_scalar(scale), 3);
        let out = attn.matmul(v);

        let out = out.swap_dims(1, 2).reshape([b, nq, h * dh]);
        self.out_proj.forward(out)
    }
}
