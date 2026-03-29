/// Dual-axis Transformer Block for SleepLM's biosignals encoder.
///
/// Python: `class DualTransformerBlock(nn.Module)` in biosignals_coca_model.py
///
/// Each block applies:
/// 1. Channel attention (across channels per time patch) with learnable RoPE
/// 2. Channel MLP
/// 3. Multiple temporal attention layers (across time patches per channel) with fixed RoPE
/// 4. Temporal MLPs

use burn::prelude::*;

use super::norm::RmsNorm;
use super::dual_attention::{ChannelAttention, TemporalAttention, Mlp};

#[derive(Module, Debug)]
pub struct DualTransformerBlock<B: Backend> {
    // ── Channel attention ────────────────────────────────────────────────
    pub channel_attention: ChannelAttention<B>,
    pub channel_norm: RmsNorm<B>,
    pub channel_mlp: Mlp<B>,
    pub channel_mlp_norm: RmsNorm<B>,

    // ── Temporal attention (may have >1 layer per block) ─────────────────
    pub temporal_attentions: Vec<TemporalAttention<B>>,
    pub temporal_norms: Vec<RmsNorm<B>>,
    pub temporal_mlps: Vec<Mlp<B>>,
    pub temporal_mlp_norms: Vec<RmsNorm<B>>,

    pub embed_dim: usize,
    pub num_temporal_layers: usize,
}

impl<B: Backend> DualTransformerBlock<B> {
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        num_temporal_layers: usize,
        mlp_ratio: f64,
        num_channels: usize,
        mlp_bias: bool,
        eps: f64,
        device: &B::Device,
    ) -> Self {
        let _ = num_channels;
        let mlp_hidden = (embed_dim as f64 * mlp_ratio) as usize;

        let channel_attention = ChannelAttention::new(embed_dim, num_heads, device);
        let channel_norm = RmsNorm::new(embed_dim, eps, device);
        let channel_mlp = Mlp::new_swiglu(embed_dim, mlp_hidden, mlp_bias, device);
        let channel_mlp_norm = RmsNorm::new(embed_dim, eps, device);

        let temporal_attentions = (0..num_temporal_layers)
            .map(|_| TemporalAttention::new(embed_dim, num_heads, device))
            .collect();
        let temporal_norms = (0..num_temporal_layers)
            .map(|_| RmsNorm::new(embed_dim, eps, device))
            .collect();
        let temporal_mlps = (0..num_temporal_layers)
            .map(|_| Mlp::new_swiglu(embed_dim, mlp_hidden, mlp_bias, device))
            .collect();
        let temporal_mlp_norms = (0..num_temporal_layers)
            .map(|_| RmsNorm::new(embed_dim, eps, device))
            .collect();

        Self {
            channel_attention,
            channel_norm,
            channel_mlp,
            channel_mlp_norm,
            temporal_attentions,
            temporal_norms,
            temporal_mlps,
            temporal_mlp_norms,
            embed_dim,
            num_temporal_layers,
        }
    }

    /// x: [B, C, T, D]
    /// temporal_freqs: [half] — fixed RoPE frequencies for temporal axis
    /// temporal_pos_ids: [T] — position indices for temporal axis
    /// channel_freqs: [half] — learnable RoPE frequencies for channel axis
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        temporal_freqs: Tensor<B, 1>,
        temporal_pos_ids: Tensor<B, 1>,
        channel_freqs: Tensor<B, 1>,
    ) -> Tensor<B, 4> {
        let [b, c, t, d] = x.dims();

        // ── 1. Channel-wise attention (across C for each time patch) ────
        // Reshape: [B, C, T, D] → [B*T, C, D]
        let x_ch = x.swap_dims(1, 2) // [B, T, C, D]
            .reshape([b * t, c, d]);

        let ch_out = self.channel_attention.forward(x_ch.clone(), channel_freqs.clone());
        let x_ch = self.channel_norm.forward(x_ch + ch_out);
        let ch_mlp_out = self.channel_mlp.forward(x_ch.clone());
        let x_ch = self.channel_mlp_norm.forward(x_ch + ch_mlp_out);

        // Reshape back: [B*T, C, D] → [B, T, C, D] → [B, C, T, D]
        let x = x_ch.reshape([b, t, c, d]).swap_dims(1, 2);

        // ── 2. Temporal attention (across T for each channel) ───────────
        // Reshape: [B, C, T, D] → [B*C, T, D]
        let mut x_t = x.reshape([b * c, t, d]);

        for i in 0..self.num_temporal_layers {
            let t_out = self.temporal_attentions[i].forward(
                x_t.clone(),
                temporal_freqs.clone(),
                temporal_pos_ids.clone(),
            );
            x_t = self.temporal_norms[i].forward(x_t + t_out);
            let mlp_out = self.temporal_mlps[i].forward(x_t.clone());
            x_t = self.temporal_mlp_norms[i].forward(x_t + mlp_out);
        }

        // Reshape back: [B*C, T, D] → [B, C, T, D]
        x_t.reshape([b, c, t, d])
    }
}
