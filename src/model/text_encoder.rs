/// CLIP-style Text Encoder for SleepLM.
///
/// Python: `class TextTransformer(nn.Module)` in transformer.py
///
/// Architecture:
/// - Token embedding + positional embedding + optional CLS token
/// - Transformer with causal attention mask
/// - Final LayerNorm + text projection
///
/// The text encoder produces:
/// - Global text embedding (for contrastive alignment)
/// - Token-level embeddings (for the decoder)

use burn::prelude::*;
use burn::module::Param;
use burn::nn::{Linear, LinearConfig, Embedding, EmbeddingConfig};
use burn::tensor::activation::softmax;

use super::norm::SleepLayerNorm;

// ── Residual Attention Block ────────────────────────────────────────────────

/// Fused multi-head self-attention (matching nn.MultiheadAttention).
#[derive(Module, Debug)]
pub struct FusedSelfAttention<B: Backend> {
    pub in_proj: Linear<B>,   // [D, 3*D]
    pub out_proj: Linear<B>,  // [D, D]
    pub num_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> FusedSelfAttention<B> {
    pub fn new(dim: usize, num_heads: usize, device: &B::Device) -> Self {
        Self {
            in_proj: LinearConfig::new(dim, dim * 3).with_bias(true).init(device),
            out_proj: LinearConfig::new(dim, dim).with_bias(true).init(device),
            num_heads,
            head_dim: dim / num_heads,
        }
    }

    /// x: [B, S, D], attn_mask: [S, S] additive mask
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        attn_mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let [b, s, _] = x.dims();
        let (h, dh) = (self.num_heads, self.head_dim);
        let dim = h * dh;

        let qkv = self.in_proj.forward(x);
        let q = qkv.clone().narrow(2, 0, dim).reshape([b, s, h, dh]).swap_dims(1, 2);
        let k = qkv.clone().narrow(2, dim, dim).reshape([b, s, h, dh]).swap_dims(1, 2);
        let v = qkv.narrow(2, dim * 2, dim).reshape([b, s, h, dh]).swap_dims(1, 2);

        let scale = (dh as f64).powf(-0.5) as f32;
        let mut attn_logits = q.matmul(k.transpose()).mul_scalar(scale);

        if let Some(mask) = attn_mask {
            // mask: [S, S] → [1, 1, S, S]
            let mask = mask.reshape([1, 1, s, s]);
            attn_logits = attn_logits + mask;
        }

        let attn = softmax(attn_logits, 3);
        let out = attn.matmul(v);

        let out = out.swap_dims(1, 2).reshape([b, s, dim]);
        self.out_proj.forward(out)
    }
}

/// ResidualAttentionBlock — self-attention variant.
///
/// Python: `class ResidualAttentionBlock(nn.Module)` in transformer.py
///   x = x + attn(ln_1(x))
///   x = x + mlp(ln_2(x))
#[derive(Module, Debug)]
pub struct ResidualAttentionBlock<B: Backend> {
    pub ln_1: SleepLayerNorm<B>,
    pub attn: FusedSelfAttention<B>,
    pub ln_2: SleepLayerNorm<B>,
    pub mlp_c_fc: Linear<B>,
    pub mlp_c_proj: Linear<B>,
}

impl<B: Backend> ResidualAttentionBlock<B> {
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

// ── Text Transformer ────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct TextEncoder<B: Backend> {
    pub token_embedding: Embedding<B>,
    pub positional_embedding: Param<Tensor<B, 2>>,
    /// Optional CLS embedding (appended to tokens before positional emb).
    pub cls_emb: Option<Param<Tensor<B, 1>>>,
    pub blocks: Vec<ResidualAttentionBlock<B>>,
    pub ln_final: SleepLayerNorm<B>,
    pub text_projection: Param<Tensor<B, 2>>,

    pub context_length: usize,
    pub width: usize,
    pub vocab_size: usize,
    pub embed_cls: bool,
}

impl<B: Backend> TextEncoder<B> {
    pub fn new(
        context_length: usize,
        vocab_size: usize,
        width: usize,
        heads: usize,
        layers: usize,
        mlp_ratio: f64,
        embed_dim: usize,
        embed_cls: bool,
        device: &B::Device,
    ) -> Self {
        let num_pos = if embed_cls { context_length + 1 } else { context_length };

        let token_embedding = EmbeddingConfig::new(vocab_size, width).init(device);
        let positional_embedding = Param::initialized(
            burn::module::ParamId::new(),
            Tensor::zeros([num_pos, width], device),
        );
        let cls_emb = if embed_cls {
            Some(Param::initialized(
                burn::module::ParamId::new(),
                Tensor::zeros([width], device),
            ))
        } else {
            None
        };

        let blocks = (0..layers)
            .map(|_| ResidualAttentionBlock::new(width, heads, mlp_ratio, device))
            .collect();

        let ln_final = SleepLayerNorm::new(width, 1e-5, device);
        let text_projection = Param::initialized(
            burn::module::ParamId::new(),
            Tensor::zeros([width, embed_dim], device),
        );

        Self {
            token_embedding,
            positional_embedding,
            cls_emb,
            blocks,
            ln_final,
            text_projection,
            context_length,
            width,
            vocab_size,
            embed_cls,
        }
    }

    /// Build causal attention mask: [S, S] with -inf above diagonal.
    fn build_causal_mask(&self, seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
        let neg_inf = -3.4028235e+38_f32; // large neg finite
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = neg_inf;
            }
        }
        Tensor::<B, 2>::from_data(
            TensorData::new(mask_data, vec![seq_len, seq_len]),
            device,
        )
    }

    /// text: [B, S] token ids
    /// Returns: (text_latent [B, embed_dim], token_embs [B, S, width])
    pub fn forward(&self, text: Tensor<B, 2, Int>) -> (Tensor<B, 2>, Tensor<B, 3>) {
        let [b, s] = text.dims();
        let device = text.device();

        // Token embedding
        let mut x = self.token_embedding.forward(text.clone()); // [B, S, width]

        // Append CLS token if configured
        let seq_len = if self.embed_cls {
            let cls = self.cls_emb.as_ref().unwrap().val()
                .reshape([1, 1, self.width])
                .expand([b, 1, self.width]);
            x = Tensor::cat(vec![x, cls], 1); // [B, S+1, width]
            s + 1
        } else {
            s
        };

        // Add positional embedding
        let pos = self.positional_embedding.val().narrow(0, 0, seq_len);
        x = x + pos.unsqueeze(); // broadcast over batch

        // Build causal mask
        let attn_mask = self.build_causal_mask(seq_len, &device);

        // Transformer blocks
        for block in &self.blocks {
            x = block.forward(x, Some(attn_mask.clone()));
        }

        // Python TextTransformer.forward():
        //   if cls_emb is not None:
        //     pooled = text_global_pool(x, pool_type='last')     # take CLS (last token)
        //     pooled = self.ln_final(pooled)                     # LN on pooled only
        //     tokens = x[:, :-1]                                 # raw tokens (no LN, no CLS)
        //   else:
        //     x = self.ln_final(x)
        //     pooled = text_global_pool(x, text, pool_type=self.pool_type)
        //     tokens = x
        let (text_latent, token_embs) = if self.embed_cls {
            // CoCa path: pool FIRST (last token = CLS), then LN only the pooled
            let pooled = x.clone().narrow(1, seq_len - 1, 1); // [B, 1, width]
            let pooled = self.ln_final.forward(pooled);
            let [pb, _, pw] = pooled.dims();
            let pooled = pooled.reshape([pb, pw]); // [B, width]
            let tokens = x.narrow(1, 0, seq_len - 1); // raw, no LN, no CLS
            (pooled, tokens)
        } else {
            let x = self.ln_final.forward(x);
            let pooled = x.clone().narrow(1, s - 1, 1);
            let [pb, _, pw] = pooled.dims();
            let pooled = pooled.reshape([pb, pw]); // [B, width]
            (pooled, x)
        };

        // Project to embed_dim
        let text_latent = text_latent.matmul(self.text_projection.val()); // [B, embed_dim]

        (text_latent, token_embs)
    }
}
