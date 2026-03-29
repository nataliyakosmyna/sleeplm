/// Pure Transformer Biosignals Encoder for SleepLM.
///
/// Python: `class PureTransformerBiosignalsEncoder(BaseBiosignalsEncoder)` in biosignals_coca_model.py
///
/// Architecture:
/// 1. ChannelPatching — Conv1d per channel (patch_size=16 → 120 patches)
/// 2. Linear projection to transformer_width + channel ID embedding
/// 3. DualTransformerBlocks × 3 (channel + temporal factorized attention)
/// 4. Final RMSNorm
/// 5. CoCa-style attentional pooling:
///    - contrastive_pooler (1 query) → global CLS token
///    - decoder_pooler (32 queries) → decoder memory tokens
/// 6. attn_pool: CLS query attends to dec_tokens (from BaseBiosignalsEncoder)
/// 7. Linear projection to embed_dim (512)

use burn::prelude::*;
use burn::module::ParamId;
type Param<T> = burn::module::Param<T>;
use burn::nn::{Linear, LinearConfig, Embedding, EmbeddingConfig};

use crate::config::BiosignalsCfg;
use super::channel_patching::ChannelPatching;
use super::dual_transformer_block::DualTransformerBlock;
use super::attn_pooler::AttnPooler;
use super::norm::RmsNorm;
// Fixed RoPE frequencies are computed inline in forward()

/// Fused MHA for the BaseBiosignalsEncoder.attn_pool (pool_type='attn').
///
/// Python: `nn.MultiheadAttention(transformer_width, transformer_heads, batch_first=True)`
/// Used as: `attn_pool(query=CLS_token, key=dec_tokens, value=dec_tokens)`
#[derive(Module, Debug)]
pub struct AttnPool<B: Backend> {
    pub in_proj: Linear<B>,   // [D, 3*D]
    pub out_proj: Linear<B>,  // [D, D]
    pub num_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> AttnPool<B> {
    pub fn new(dim: usize, num_heads: usize, device: &B::Device) -> Self {
        Self {
            in_proj: LinearConfig::new(dim, dim * 3).with_bias(true).init(device),
            out_proj: LinearConfig::new(dim, dim).with_bias(true).init(device),
            num_heads,
            head_dim: dim / num_heads,
        }
    }

    /// query: [B, 1, D] (CLS token), kv: [B, N, D] (decoder tokens)
    /// Returns: [B, D]
    pub fn forward(&self, query: Tensor<B, 3>, kv: Tensor<B, 3>) -> Tensor<B, 2> {
        let [b, _sq, _] = query.dims();
        let s_kv = kv.dims()[1];
        let (h, dh) = (self.num_heads, self.head_dim);
        let dim = h * dh;

        // Fused projection for Q (from query)
        let qkv_q = self.in_proj.forward(query);
        let q = qkv_q.narrow(2, 0, dim).reshape([b, 1, h, dh]).swap_dims(1, 2);

        // Fused projection for K, V (from kv)
        let qkv_kv = self.in_proj.forward(kv);
        let k = qkv_kv.clone().narrow(2, dim, dim).reshape([b, s_kv, h, dh]).swap_dims(1, 2);
        let v = qkv_kv.narrow(2, dim * 2, dim).reshape([b, s_kv, h, dh]).swap_dims(1, 2);

        let scale = (dh as f64).powf(-0.5) as f32;
        let attn = burn::tensor::activation::softmax(
            q.matmul(k.transpose()).mul_scalar(scale), 3,
        );
        let out = attn.matmul(v); // [B, H, 1, dh]

        let out = out.swap_dims(1, 2).reshape([b, 1, dim]);
        let out = self.out_proj.forward(out);
        // [B, 1, D] → [B, D]
        let [b2, _, d2] = out.dims();
        out.reshape([b2, d2])
    }
}

#[derive(Module, Debug)]
pub struct BiosignalsEncoder<B: Backend> {
    // ── Patching & projection ───────────────────────────────────────────
    pub patching: ChannelPatching<B>,
    pub embed_projection: Linear<B>,
    pub channel_id_embed: Embedding<B>,

    // ── Shared learnable RoPE for channel attention ─────────────────────
    pub shared_channel_freqs: Param<Tensor<B, 1>>,

    // ── Dual-axis transformer blocks ────────────────────────────────────
    pub blocks: Vec<DualTransformerBlock<B>>,

    // ── Final norm ──────────────────────────────────────────────────────
    pub ln_final: RmsNorm<B>,

    // ── CoCa-style attentional poolers ──────────────────────────────────
    pub contrastive_pooler: AttnPooler<B>,
    pub decoder_pooler: AttnPooler<B>,

    // ── BaseBiosignalsEncoder attn_pool (pool_type='attn') ──────────────
    pub attn_pool: AttnPool<B>,

    // ── Output projection ───────────────────────────────────────────────
    pub proj_to_embed: Linear<B>,

    // ── Config ──────────────────────────────────────────────────────────
    pub num_patches: usize,
    pub transformer_width: usize,
    pub input_channels: usize,
}

impl<B: Backend> BiosignalsEncoder<B> {
    pub fn new(cfg: &BiosignalsCfg, embed_dim: usize, device: &B::Device) -> Self {
        let num_patches = cfg.num_patches();
        let head_dim = cfg.head_dim();
        let half = head_dim / 2;

        let patching = ChannelPatching::new(
            cfg.patch_size, cfg.conv_embed_dim, cfg.input_channels, device,
        );
        let embed_projection = LinearConfig::new(cfg.conv_embed_dim, cfg.transformer_width)
            .with_bias(true)
            .init(device);
        let channel_id_embed = EmbeddingConfig::new(cfg.input_channels, cfg.transformer_width)
            .init(device);

        // Shared learnable frequencies for channel RoPE
        let init_freqs: Vec<f32> = vec![0.0; half];
        let shared_channel_freqs = Param::initialized(
            ParamId::new(),
            Tensor::from_data(TensorData::new(init_freqs, vec![half]), device),
        );

        let blocks = (0..cfg.transformer_layers)
            .map(|_| DualTransformerBlock::new(
                cfg.transformer_width,
                cfg.transformer_heads,
                cfg.num_temporal_layers,
                cfg.mlp_ratio,
                cfg.input_channels,
                cfg.mlp_bias,
                1e-6, // RMSNorm eps
                device,
            ))
            .collect();

        let ln_final = RmsNorm::new(cfg.transformer_width, 1e-6, device);

        let contrastive_pooler = AttnPooler::new(
            cfg.transformer_width, cfg.transformer_heads, 1, device,
        );
        let decoder_pooler = AttnPooler::new(
            cfg.transformer_width, cfg.transformer_heads, cfg.decoder_tokens, device,
        );

        // BaseBiosignalsEncoder.attn_pool
        let attn_pool = AttnPool::new(cfg.transformer_width, cfg.transformer_heads, device);

        let proj_to_embed = LinearConfig::new(cfg.transformer_width, embed_dim)
            .with_bias(true)
            .init(device);

        Self {
            patching,
            embed_projection,
            channel_id_embed,
            shared_channel_freqs,
            blocks,
            ln_final,
            contrastive_pooler,
            decoder_pooler,
            attn_pool,
            proj_to_embed,
            num_patches,
            transformer_width: cfg.transformer_width,
            input_channels: cfg.input_channels,
        }
    }

    /// Encode biosignals to (global_embedding, decoder_tokens).
    ///
    /// biosignals: [B, C, signal_length]
    /// Returns:
    ///   embedding: [B, embed_dim] — global contrastive embedding
    ///   tokens: [B, decoder_tokens, transformer_width] — tokens for text decoder
    pub fn forward(&self, biosignals: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 3>) {
        let [b, c, _t] = biosignals.dims();
        let device = biosignals.device();
        let d = self.transformer_width;
        let num_patches = self.num_patches;

        // 1. Patch per channel → [B, C, T, conv_dim]
        let x = self.patching.forward(biosignals);

        // 2. Project to model dim → [B, C, T, D]
        let x = self.embed_projection.forward(x);

        // 2a. Add channel ID embedding
        let channel_ids: Vec<i64> = (0..c as i64).collect();
        let ch_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(channel_ids, vec![1, c]),
            &device,
        );
        let ch_emb = self.channel_id_embed.forward(ch_ids); // [1, C, D]
        let ch_emb = ch_emb.expand([b, c, d]).reshape([b, c, 1, d]).expand([b, c, num_patches, d]);
        let x = x + ch_emb;

        // 3. Prepare fixed temporal RoPE frequencies
        let head_dim = d / self.contrastive_pooler.num_heads;
        let half = head_dim / 2;
        let temporal_freq_data: Vec<f32> = (0..half)
            .map(|i| 1.0 / 10000.0_f64.powf(2.0 * i as f64 / head_dim as f64) as f32)
            .collect();
        let temporal_freqs = Tensor::<B, 1>::from_data(
            TensorData::new(temporal_freq_data, vec![half]),
            &device,
        );

        // Temporal position IDs
        let pos_data: Vec<f32> = (0..num_patches).map(|i| i as f32).collect();
        let temporal_pos_ids = Tensor::<B, 1>::from_data(
            TensorData::new(pos_data, vec![num_patches]),
            &device,
        );

        // Channel learnable freqs
        let channel_freqs = self.shared_channel_freqs.val();

        // 4. Dual-axis transformer blocks
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(
                x,
                temporal_freqs.clone(),
                temporal_pos_ids.clone(),
                channel_freqs.clone(),
            );
        }

        // 5. Final norm: [B, C, T, D]
        let x = self.ln_final.forward(x);

        // 6. Flatten to sequence: [B, C*T, D]
        let x_seq = x.reshape([b, c * num_patches, d]);

        // 7. CoCa-style attentional pooling
        let global_token = self.contrastive_pooler.forward(x_seq.clone()); // [B, 1, D]
        let dec_tokens = self.decoder_pooler.forward(x_seq);               // [B, Nd, D]

        // 8. BaseBiosignalsEncoder._pool_features with pool_type='attn':
        //    query = global_token (CLS), key/value = dec_tokens
        //    pooled = attn_pool(query=CLS, key=dec_tokens, value=dec_tokens)
        let pooled = self.attn_pool.forward(global_token, dec_tokens.clone()); // [B, D]

        // 9. Project to embed_dim
        let embedding = self.proj_to_embed.forward(pooled); // [B, embed_dim]

        (embedding, dec_tokens)
    }
}
