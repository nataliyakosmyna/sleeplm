/// Dual RoPE Attention for SleepLM's factorized attention.
///
/// Python: `class DualRoPEAttention(nn.Module)` in biosignals_coca_model.py
///
/// Two variants:
/// - **Temporal**: standard fixed RoPE (theta=10000)
/// - **Channel**: learnable RoPE (shared across blocks)

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;

use super::rope::apply_interleaved_rope;

/// SwiGLU activation: SiLU(W1(x)) * W2(x)
///
/// Python: `class SwiGLU(nn.Module)` in biosignals_coca_model.py
#[derive(Module, Debug)]
pub struct SwiGLU<B: Backend> {
    pub w1: Linear<B>,
    pub w2: Linear<B>,
}

impl<B: Backend> SwiGLU<B> {
    pub fn new(dim_in: usize, dim_out: usize, bias: bool, device: &B::Device) -> Self {
        Self {
            w1: LinearConfig::new(dim_in, dim_out).with_bias(bias).init(device),
            w2: LinearConfig::new(dim_in, dim_out).with_bias(bias).init(device),
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // SiLU(W1(x)) * W2(x)
        let gate = burn::tensor::activation::silu(self.w1.forward(x.clone()));
        gate * self.w2.forward(x)
    }
}

/// MLP block with SwiGLU or GELU activation.
///
/// Python: `class MLP(nn.Module)` in biosignals_coca_model.py
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    pub gate_proj: Option<SwiGLU<B>>,
    pub up_proj: Option<Linear<B>>,
    pub down_proj: Linear<B>,
    pub use_swiglu: bool,
}

impl<B: Backend> Mlp<B> {
    pub fn new_swiglu(dim: usize, hidden_dim: usize, bias: bool, device: &B::Device) -> Self {
        Self {
            gate_proj: Some(SwiGLU::new(dim, hidden_dim, bias, device)),
            up_proj: None,
            down_proj: LinearConfig::new(hidden_dim, dim).with_bias(bias).init(device),
            use_swiglu: true,
        }
    }

    pub fn new_gelu(dim: usize, hidden_dim: usize, bias: bool, device: &B::Device) -> Self {
        Self {
            gate_proj: None,
            up_proj: Some(LinearConfig::new(dim, hidden_dim).with_bias(bias).init(device)),
            down_proj: LinearConfig::new(hidden_dim, dim).with_bias(bias).init(device),
            use_swiglu: false,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        if self.use_swiglu {
            let h = self.gate_proj.as_ref().unwrap().forward(x);
            self.down_proj.forward(h)
        } else {
            let h = burn::tensor::activation::gelu(
                self.up_proj.as_ref().unwrap().forward(x)
            );
            self.down_proj.forward(h)
        }
    }
}

/// Multi-head self-attention with RoPE applied to Q and K.
///
/// Python: `class DualRoPEAttention(nn.Module)` (temporal variant)
#[derive(Module, Debug)]
pub struct TemporalAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> TemporalAttention<B> {
    pub fn new(embed_dim: usize, num_heads: usize, device: &B::Device) -> Self {
        let head_dim = embed_dim / num_heads;
        Self {
            q_proj: LinearConfig::new(embed_dim, embed_dim).with_bias(false).init(device),
            k_proj: LinearConfig::new(embed_dim, embed_dim).with_bias(false).init(device),
            v_proj: LinearConfig::new(embed_dim, embed_dim).with_bias(false).init(device),
            out_proj: LinearConfig::new(embed_dim, embed_dim).with_bias(true).init(device),
            num_heads,
            head_dim,
        }
    }

    /// x: [B, S, D], freqs: [half], position_ids: [S]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        freqs: Tensor<B, 1>,
        position_ids: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let [b, s, _] = x.dims();
        let (h, dh) = (self.num_heads, self.head_dim);

        let q = self.q_proj.forward(x.clone()).reshape([b, s, h, dh]).swap_dims(1, 2);
        let k = self.k_proj.forward(x.clone()).reshape([b, s, h, dh]).swap_dims(1, 2);
        let v = self.v_proj.forward(x).reshape([b, s, h, dh]).swap_dims(1, 2);

        // Apply RoPE
        let q = apply_interleaved_rope(q, freqs.clone(), position_ids.clone());
        let k = apply_interleaved_rope(k, freqs, position_ids);

        // Scaled dot-product attention
        let scale = (dh as f64).powf(-0.5) as f32;
        let attn = softmax(q.matmul(k.transpose()).mul_scalar(scale), 3);
        let out = attn.matmul(v);

        let out = out.swap_dims(1, 2).reshape([b, s, h * dh]);
        self.out_proj.forward(out)
    }
}

/// Multi-head self-attention with learnable RoPE for channel dimension.
///
/// Python: `class DualRoPEAttention(nn.Module)` (channel variant)
#[derive(Module, Debug)]
pub struct ChannelAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> ChannelAttention<B> {
    pub fn new(embed_dim: usize, num_heads: usize, device: &B::Device) -> Self {
        let head_dim = embed_dim / num_heads;
        Self {
            q_proj: LinearConfig::new(embed_dim, embed_dim).with_bias(false).init(device),
            k_proj: LinearConfig::new(embed_dim, embed_dim).with_bias(false).init(device),
            v_proj: LinearConfig::new(embed_dim, embed_dim).with_bias(false).init(device),
            out_proj: LinearConfig::new(embed_dim, embed_dim).with_bias(true).init(device),
            num_heads,
            head_dim,
        }
    }

    /// x: [B, C, D], learnable_freqs: [half]
    /// Applies learnable RoPE using channel indices as position_ids.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        learnable_freqs: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let [b, c, _] = x.dims();
        let (h, dh) = (self.num_heads, self.head_dim);
        let device = x.device();

        let q = self.q_proj.forward(x.clone()).reshape([b, c, h, dh]).swap_dims(1, 2);
        let k = self.k_proj.forward(x.clone()).reshape([b, c, h, dh]).swap_dims(1, 2);
        let v = self.v_proj.forward(x).reshape([b, c, h, dh]).swap_dims(1, 2);

        // Channel position ids: 0, 1, ..., C-1
        let pos_data: Vec<f32> = (0..c).map(|i| i as f32).collect();
        let position_ids = Tensor::<B, 1>::from_data(
            TensorData::new(pos_data, vec![c]),
            &device,
        );

        let q = apply_interleaved_rope(q, learnable_freqs.clone(), position_ids.clone());
        let k = apply_interleaved_rope(k, learnable_freqs, position_ids);

        let scale = (dh as f64).powf(-0.5) as f32;
        let attn = softmax(q.matmul(k.transpose()).mul_scalar(scale), 3);
        let out = attn.matmul(v);

        let out = out.swap_dims(1, 2).reshape([b, c, h * dh]);
        self.out_proj.forward(out)
    }
}
