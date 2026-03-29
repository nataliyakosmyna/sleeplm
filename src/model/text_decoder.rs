/// Cross-Attention Text Decoder for SleepLM (CoCa style).
///
/// Python: `class MultimodalTransformer(Transformer)` in transformer.py
///
/// Each layer has:
/// 1. Self-attention on text (causal) — ResidualAttentionBlock
/// 2. Cross-attention from text to biosignal tokens — ResidualAttentionBlock (is_cross_attention=True)
///
/// Also supports prefix-causal masking for modality condition tokens.

use burn::prelude::*;
use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;

use super::norm::SleepLayerNorm;
use super::text_encoder::FusedSelfAttention;

// ── Cross-Attention Block ───────────────────────────────────────────────────

/// Multi-head cross-attention (Q from text, K/V from biosignals).
#[derive(Module, Debug)]
pub struct FusedCrossAttention<B: Backend> {
    pub in_proj: Linear<B>,   // [D, 3*D]
    pub out_proj: Linear<B>,  // [D, D]
    pub num_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> FusedCrossAttention<B> {
    pub fn new(dim: usize, num_heads: usize, device: &B::Device) -> Self {
        Self {
            in_proj: LinearConfig::new(dim, dim * 3).with_bias(true).init(device),
            out_proj: LinearConfig::new(dim, dim).with_bias(true).init(device),
            num_heads,
            head_dim: dim / num_heads,
        }
    }

    /// q_input: [B, S_q, D] (text)
    /// kv_input: [B, S_kv, D] (biosignal tokens)
    pub fn forward(
        &self,
        q_input: Tensor<B, 3>,
        kv_input: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [b, s_q, _] = q_input.dims();
        let s_kv = kv_input.dims()[1];
        let (h, dh) = (self.num_heads, self.head_dim);
        let dim = h * dh;

        // Q from text
        let qkv_q = self.in_proj.forward(q_input);
        let q = qkv_q.narrow(2, 0, dim).reshape([b, s_q, h, dh]).swap_dims(1, 2);

        // K, V from biosignal tokens
        let qkv_kv = self.in_proj.forward(kv_input);
        let k = qkv_kv.clone().narrow(2, dim, dim).reshape([b, s_kv, h, dh]).swap_dims(1, 2);
        let v = qkv_kv.narrow(2, dim * 2, dim).reshape([b, s_kv, h, dh]).swap_dims(1, 2);

        let scale = (dh as f64).powf(-0.5) as f32;
        let attn = softmax(q.matmul(k.transpose()).mul_scalar(scale), 3);
        let out = attn.matmul(v);

        let out = out.swap_dims(1, 2).reshape([b, s_q, dim]);
        self.out_proj.forward(out)
    }
}

/// Cross-attention residual block.
///
/// Python: ResidualAttentionBlock with is_cross_attention=True
///   x = x + attn(ln_1(q_x), k_x=ln_1_kv(k_x), v_x=ln_1_kv(v_x))
///   x = x + mlp(ln_2(x))
#[derive(Module, Debug)]
pub struct CrossAttentionBlock<B: Backend> {
    pub ln_1: SleepLayerNorm<B>,
    pub ln_1_kv: SleepLayerNorm<B>,
    pub attn: FusedCrossAttention<B>,
    pub ln_2: SleepLayerNorm<B>,
    pub mlp_c_fc: Linear<B>,
    pub mlp_c_proj: Linear<B>,
}

impl<B: Backend> CrossAttentionBlock<B> {
    pub fn new(width: usize, heads: usize, mlp_ratio: f64, device: &B::Device) -> Self {
        let mlp_width = (width as f64 * mlp_ratio) as usize;
        Self {
            ln_1: SleepLayerNorm::new(width, 1e-5, device),
            ln_1_kv: SleepLayerNorm::new(width, 1e-5, device),
            attn: FusedCrossAttention::new(width, heads, device),
            ln_2: SleepLayerNorm::new(width, 1e-5, device),
            mlp_c_fc: LinearConfig::new(width, mlp_width).with_bias(true).init(device),
            mlp_c_proj: LinearConfig::new(mlp_width, width).with_bias(true).init(device),
        }
    }

    /// q_x: text tokens [B, S_text, D]
    /// kv_x: biosignal tokens [B, S_bio, D]
    pub fn forward(&self, q_x: Tensor<B, 3>, kv_x: Tensor<B, 3>) -> Tensor<B, 3> {
        let normed_kv = self.ln_1_kv.forward(kv_x);
        let x = q_x.clone() + self.attn.forward(
            self.ln_1.forward(q_x),
            normed_kv,
        );
        let mlp_out = self.mlp_c_proj.forward(
            burn::tensor::activation::gelu(self.mlp_c_fc.forward(self.ln_2.forward(x.clone())))
        );
        x + mlp_out
    }
}

// ── Self-attention residual block (for decoder) ─────────────────────────────

#[derive(Module, Debug)]
pub struct SelfAttentionBlock<B: Backend> {
    pub ln_1: SleepLayerNorm<B>,
    pub attn: FusedSelfAttention<B>,
    pub ln_2: SleepLayerNorm<B>,
    pub mlp_c_fc: Linear<B>,
    pub mlp_c_proj: Linear<B>,
}

impl<B: Backend> SelfAttentionBlock<B> {
    pub fn new(width: usize, heads: usize, mlp_ratio: f64, device: &B::Device) -> Self {
        let mlp_width = (width as f64 * mlp_ratio) as usize;
        Self {
            ln_1: SleepLayerNorm::new(width, 1e-5, device),
            attn: FusedSelfAttention::new(width, heads, device),
            ln_2: SleepLayerNorm::new(width, 1e-5, device),
            mlp_c_fc: LinearConfig::new(width, mlp_width).with_bias(true).init(device),
            mlp_c_proj: LinearConfig::new(mlp_width, width).with_bias(true).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, attn_mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        let h = x.clone() + self.attn.forward(self.ln_1.forward(x.clone()), attn_mask);
        let mlp_out = self.mlp_c_proj.forward(
            burn::tensor::activation::gelu(self.mlp_c_fc.forward(self.ln_2.forward(h.clone())))
        );
        h + mlp_out
    }
}

// ── Multimodal Text Decoder ─────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct TextDecoder<B: Backend> {
    /// Self-attention blocks (causal on text).
    pub self_attn_blocks: Vec<SelfAttentionBlock<B>>,
    /// Cross-attention blocks (text → biosignals).
    pub cross_attn_blocks: Vec<CrossAttentionBlock<B>>,
    /// Final layer norm.
    pub ln_final: SleepLayerNorm<B>,
    /// Output projection: [width, output_dim=vocab_size].
    pub text_projection: Param<Tensor<B, 2>>,

    pub width: usize,
    pub context_length: usize,
    pub prefix_len: usize,
}

impl<B: Backend> TextDecoder<B> {
    /// output_dim here is **vocab_size** (49408), not embed_dim.
    /// The decoder text_projection maps from width → vocab logits.
    pub fn new(
        width: usize,
        heads: usize,
        layers: usize,
        mlp_ratio: f64,
        context_length: usize,
        output_dim: usize,
        prefix_len: usize,
        device: &B::Device,
    ) -> Self {
        let self_attn_blocks = (0..layers)
            .map(|_| SelfAttentionBlock::new(width, heads, mlp_ratio, device))
            .collect();
        let cross_attn_blocks = (0..layers)
            .map(|_| CrossAttentionBlock::new(width, heads, mlp_ratio, device))
            .collect();

        let ln_final = SleepLayerNorm::new(width, 1e-5, device);
        let text_projection = Param::initialized(
            burn::module::ParamId::new(),
            Tensor::zeros([width, output_dim], device),
        );

        Self {
            self_attn_blocks,
            cross_attn_blocks,
            ln_final,
            text_projection,
            width,
            context_length,
            prefix_len,
        }
    }

    /// Build prefix-causal mask.
    ///
    /// prefix tokens: full bidirectional attention among themselves
    /// text tokens: causal attention + full attention to prefix
    fn build_prefix_causal_mask(
        &self,
        prefix_len: usize,
        text_len: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let total = prefix_len + text_len;
        let neg = -3.4028235e+38_f32;
        let mut mask_data = vec![0.0f32; total * total];

        // Prefix → Text: block
        for i in 0..prefix_len {
            for j in prefix_len..total {
                mask_data[i * total + j] = neg;
            }
        }

        // Text → Text: causal (block future)
        for i in prefix_len..total {
            for j in (i + 1)..total {
                mask_data[i * total + j] = neg;
            }
        }

        Tensor::<B, 2>::from_data(
            TensorData::new(mask_data, vec![total, total]),
            device,
        )
    }

    /// Forward pass.
    ///
    /// biosignal_tokens: [B, N_bio, width] — from biosignals encoder decoder_pooler
    /// text_embs: [B, S_text, width] — token embeddings
    /// condition_embs: Optional [B, prefix_len, width] — modality condition embeddings
    ///
    /// Returns: logits [B, S_text, output_dim]
    pub fn forward(
        &self,
        biosignal_tokens: Tensor<B, 3>,
        text_embs: Tensor<B, 3>,
        condition_embs: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let text_len = text_embs.dims()[1];
        let device = text_embs.device();

        // Prepend condition tokens if provided
        let (x, condition_len) = if let Some(cond) = condition_embs {
            let cond_len = cond.dims()[1];
            (Tensor::cat(vec![cond, text_embs], 1), cond_len)
        } else {
            (text_embs, 0)
        };

        // Build attention mask
        let attn_mask = if condition_len > 0 {
            self.build_prefix_causal_mask(condition_len, text_len, &device)
        } else {
            // Simple causal mask
            self.build_prefix_causal_mask(0, text_len, &device)
        };

        // Interleaved self-attention + cross-attention
        let mut x = x;
        for (sa_block, xa_block) in self.self_attn_blocks.iter().zip(self.cross_attn_blocks.iter()) {
            x = sa_block.forward(x, Some(attn_mask.clone()));
            x = xa_block.forward(x, biosignal_tokens.clone());
        }

        // Final norm
        let out = self.ln_final.forward(x);

        // Project to output dim: [B, S, width] @ [width, output_dim] → [B, S, output_dim]
        let proj = self.text_projection.val().unsqueeze::<3>(); // [1, width, output_dim]
        let out = out.matmul(proj);

        // Strip condition tokens
        if condition_len > 0 {
            out.narrow(1, condition_len, text_len)
        } else {
            out
        }
    }
}
