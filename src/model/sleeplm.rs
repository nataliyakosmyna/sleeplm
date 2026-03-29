/// SleepLM — full BiosignalsCoCa model.
///
/// Python: `class BiosignalsCoCa(nn.Module)` in biosignals_coca_model.py
///
/// Components:
/// 1. Biosignals encoder (PureTransformerBiosignalsEncoder)
/// 2. Text encoder (TextTransformer / CLIP-style)
/// 3. Text decoder (MultimodalTransformer / cross-attention CoCa decoder)
/// 4. Logit scale (learnable temperature for contrastive loss)
/// 5. Channel/modality condition embeddings

use burn::prelude::*;
use burn::module::Param;

use crate::config::ModelConfig;
use super::biosignals_encoder::BiosignalsEncoder;
use super::text_encoder::TextEncoder;
use super::text_decoder::TextDecoder;

#[derive(Module, Debug)]
pub struct SleepLM<B: Backend> {
    // ── Sub-models ──────────────────────────────────────────────────────
    pub biosignals: BiosignalsEncoder<B>,
    pub text: TextEncoder<B>,
    pub text_decoder: TextDecoder<B>,

    // ── Contrastive temperature ─────────────────────────────────────────
    pub logit_scale: Param<Tensor<B, 1>>,

    // ── Condition embeddings ────────────────────────────────────────────
    /// Channel/modality embeddings: [num_caption_channels, decoder_width].
    pub channel_embeddings: Param<Tensor<B, 2>>,
    /// Learnable padding embedding for -1 positions: [decoder_width].
    pub padding_embedding: Param<Tensor<B, 1>>,

    // ── Config ──────────────────────────────────────────────────────────
    pub num_caption_channels: usize,
    pub prefix_len: usize,
    pub decoder_width: usize,
}

impl<B: Backend> SleepLM<B> {
    pub fn new(cfg: &ModelConfig, device: &B::Device) -> Self {
        let biosignals = BiosignalsEncoder::new(
            &cfg.biosignals_cfg, cfg.embed_dim, device,
        );

        let text = TextEncoder::new(
            cfg.text_cfg.context_length,
            cfg.text_cfg.vocab_size,
            cfg.text_cfg.width,
            cfg.text_cfg.heads,
            cfg.text_cfg.layers,
            4.0, // mlp_ratio for text encoder
            cfg.embed_dim,
            cfg.text_cfg.embed_cls,
            device,
        );

        // text_decoder.text_projection is [width, vocab_size] — projects to logits
        let text_decoder = TextDecoder::new(
            cfg.multimodal_cfg.width,
            cfg.multimodal_cfg.heads,
            cfg.multimodal_cfg.layers,
            cfg.multimodal_cfg.mlp_ratio,
            cfg.multimodal_cfg.context_length,
            cfg.text_cfg.vocab_size,  // output_dim = vocab_size for caption logits
            cfg.prefix_len,
            device,
        );

        let init_logit_scale = (1.0_f32 / 0.07).ln();
        let logit_scale = Param::initialized(
            burn::module::ParamId::new(),
            Tensor::from_data(
                TensorData::new(vec![init_logit_scale], vec![1]),
                device,
            ),
        );

        let channel_embeddings = Param::initialized(
            burn::module::ParamId::new(),
            Tensor::zeros([cfg.num_caption_channels, cfg.multimodal_cfg.width], device),
        );

        let padding_embedding = Param::initialized(
            burn::module::ParamId::new(),
            Tensor::zeros([cfg.multimodal_cfg.width], device),
        );

        Self {
            biosignals,
            text,
            text_decoder,
            logit_scale,
            channel_embeddings,
            padding_embedding,
            num_caption_channels: cfg.num_caption_channels,
            prefix_len: cfg.prefix_len,
            decoder_width: cfg.multimodal_cfg.width,
        }
    }

    /// Encode biosignals to (global_embedding, decoder_tokens).
    pub fn encode_biosignals(&self, biosignals: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 3>) {
        self.biosignals.forward(biosignals)
    }

    /// Encode text to global embedding.
    pub fn encode_text(&self, text: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let (latent, _) = self.text.forward(text);
        latent
    }

    /// Build condition embeddings from channel/modality indices.
    ///
    /// channel_indices: [B, prefix_len] — indices into channel_embeddings (-1 = padding)
    /// Returns: [B, prefix_len, decoder_width]
    pub fn get_condition_embs(&self, channel_indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, prefix_len] = channel_indices.dims();
        let device = channel_indices.device();
        let d = self.decoder_width;

        // For simplicity, iterate over the prefix positions
        // In practice this is tiny (prefix_len = 1 or 2)
        let mut emb_data = Vec::with_capacity(b * prefix_len * d);

        let ch_embs = self.channel_embeddings.val()
            .into_data().to_vec::<f32>().unwrap();
        let pad_emb = self.padding_embedding.val()
            .into_data().to_vec::<f32>().unwrap();
        let raw = channel_indices.into_data();
        let idx_data: Vec<i64> = raw.to_vec::<i64>()
            .or_else(|_| raw.to_vec::<i32>().map(|v| v.into_iter().map(|x| x as i64).collect()))
            .unwrap();

        for bi in 0..b {
            for pi in 0..prefix_len {
                let idx = idx_data[bi * prefix_len + pi];
                if idx < 0 {
                    emb_data.extend_from_slice(&pad_emb);
                } else {
                    let start = idx as usize * d;
                    emb_data.extend_from_slice(&ch_embs[start..start + d]);
                }
            }
        }

        Tensor::<B, 3>::from_data(
            TensorData::new(emb_data, vec![b, prefix_len, d]),
            &device,
        )
    }

    /// Full forward pass (contrastive + captioning).
    ///
    /// biosignals: [B, C, signal_length]
    /// text: [B, S] token ids
    /// channel_indices: [B, prefix_len] modality indices
    ///
    /// Returns: SleepLMOutput
    pub fn forward(
        &self,
        biosignals: Tensor<B, 3>,
        text: Tensor<B, 2, Int>,
        channel_indices: Option<Tensor<B, 2, Int>>,
    ) -> SleepLMOutput<B> {
        let (bio_latent, bio_tokens) = self.encode_biosignals(biosignals);
        let text_len = text.dims()[1];
        let (text_latent, token_embs) = self.text.forward(text.clone());

        // Labels for caption loss: text[:, 1:]
        let labels = text.narrow(1, 1, text_len - 1);
        let te_len = token_embs.dims()[1];
        let token_embs = token_embs.narrow(1, 0, te_len - 1);

        // Condition embeddings
        let condition_embs = channel_indices.map(|idx| self.get_condition_embs(idx));

        // Decoder: generate logits
        let logits = self.text_decoder.forward(bio_tokens, token_embs, condition_embs);

        SleepLMOutput {
            biosignal_features: bio_latent,
            text_features: text_latent,
            logits,
            labels,
            logit_scale: self.logit_scale.val().exp(),
        }
    }
}

/// Output of SleepLM forward pass.
pub struct SleepLMOutput<B: Backend> {
    /// Biosignal global embedding: [B, embed_dim].
    pub biosignal_features: Tensor<B, 2>,
    /// Text global embedding: [B, embed_dim].
    pub text_features: Tensor<B, 2>,
    /// Caption logits: [B, S-1, vocab_size].
    pub logits: Tensor<B, 3>,
    /// Target labels: [B, S-1].
    pub labels: Tensor<B, 2, Int>,
    /// Temperature scale (scalar).
    pub logit_scale: Tensor<B, 1>,
}
