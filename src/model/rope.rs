/// Rotary Position Embeddings for SleepLM's dual-axis transformer.
///
/// SleepLM uses two types of RoPE:
/// - **Fixed RoPE** for temporal attention (standard sinusoidal frequencies)
/// - **Learnable RoPE** for channel attention (learned frequency parameters)
///
/// Python: `class RotaryEmbedding(nn.Module)` in biosignals_coca_model.py

use burn::prelude::*;
use burn::module::{Param, ParamId};

/// Apply interleaved RoPE rotation to queries or keys.
///
/// x: [B, num_heads, seq_len, head_dim]
/// freqs: [half] — precomputed or learned frequency parameters
/// position_ids: [seq_len]
///
/// Returns: rotated tensor of same shape.
///
/// Matches the Python implementation:
///   angles = einsum('s,d->sd', position_ids, freqs)
///   cos = cos(angles).repeat_interleave(2, dim=-1)
///   sin = sin(angles).repeat_interleave(2, dim=-1)
///   x_rotated[..., 0::2] = x1 * cos[0::2] - x2 * sin[0::2]
///   x_rotated[..., 1::2] = x1 * sin[1::2] + x2 * cos[1::2]
///
/// Since cos/sin are repeat_interleaved, this is equivalent to
/// rotating each adjacent pair (x[2i], x[2i+1]) by angle[i].
pub fn apply_interleaved_rope<B: Backend>(
    x: Tensor<B, 4>,
    freqs: Tensor<B, 1>,
    position_ids: Tensor<B, 1>,
) -> Tensor<B, 4> {
    let [b, h, s, d] = x.dims();
    let half = d / 2;
    let _device = x.device();

    // angles[i, j] = position_ids[i] * freqs[j]  →  [s, half]
    let pos = position_ids.reshape([s, 1]);
    let freq = freqs.reshape([1, half]);
    let angles = pos.matmul(freq); // [s, half]
    let cos = angles.clone().cos(); // [s, half]
    let sin = angles.sin();         // [s, half]

    // Reshape x to [b, h, s, half, 2] for pair-wise rotation
    let x_pairs = x.reshape([b, h, s, half, 2]);
    let x_even = x_pairs.clone().narrow(4, 0, 1).squeeze::<4>(); // [b, h, s, half]
    let x_odd  = x_pairs.narrow(4, 1, 1).squeeze::<4>();         // [b, h, s, half]

    let cos = cos.reshape([1, 1, s, half]);
    let sin = sin.reshape([1, 1, s, half]);

    // Rotation per pair: out_even = x_even * cos - x_odd * sin
    //                     out_odd  = x_even * sin + x_odd * cos
    let out_even = x_even.clone() * cos.clone() - x_odd.clone() * sin.clone();
    let out_odd  = x_even * sin + x_odd * cos;

    // Interleave back: stack at dim 4 then flatten → [b, h, s, d]
    let out_even = out_even.unsqueeze_dim::<5>(4); // [b, h, s, half, 1]
    let out_odd  = out_odd.unsqueeze_dim::<5>(4);
    Tensor::cat(vec![out_even, out_odd], 4).reshape([b, h, s, d])
}

/// Learnable RoPE for channel attention.
///
/// Python: `RotaryEmbedding(dim, theta=10000, learned_freq=True)`
/// Frequencies are nn.Parameter (learnable), shared across blocks.
#[derive(Module, Debug)]
pub struct LearnableRoPE<B: Backend> {
    /// Learned frequencies: [dim/2]
    pub freqs: Param<Tensor<B, 1>>,
    pub dim: usize,
}

impl<B: Backend> LearnableRoPE<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        let half = dim / 2;
        let init_data: Vec<f32> = vec![0.0; half];
        let freqs = Param::initialized(
            ParamId::new(),
            Tensor::from_data(TensorData::new(init_data, vec![half]), device),
        );
        Self { freqs, dim }
    }

    /// Apply learnable RoPE.
    pub fn rotate(&self, x: Tensor<B, 4>, position_ids: Tensor<B, 1>) -> Tensor<B, 4> {
        apply_interleaved_rope(x, self.freqs.val(), position_ids)
    }
}
