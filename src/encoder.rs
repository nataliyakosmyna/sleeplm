//! Standalone SleepLM encoder — produce PSG embeddings and captions.
//!
//! SleepLM is an encoder + decoder model (CoCa architecture).
//! The encoder produces:
//! - Global contrastive embeddings [B, embed_dim]
//! - Decoder tokens [B, 32, width] for caption generation

use std::{path::Path, time::Instant};

use anyhow::Context;
use burn::prelude::*;

use crate::{
    config::ModelConfig,
    data::{InputBatch, EpochEmbedding, channel_wise_normalize},
    model::sleeplm::SleepLM,
    weights::load_model,
};

/// High-level SleepLM encoder for inference.
pub struct SleepLMEncoder<B: Backend> {
    model: SleepLM<B>,
    pub model_cfg: ModelConfig,
    device: B::Device,
}

impl<B: Backend> SleepLMEncoder<B> {
    /// Load model from config.json and weights safetensors.
    pub fn load(
        config_path: &Path,
        weights_path: &Path,
        device: B::Device,
    ) -> anyhow::Result<(Self, f64)> {
        let cfg_str = std::fs::read_to_string(config_path)
            .with_context(|| format!("config: {}", config_path.display()))?;
        let model_cfg: ModelConfig = serde_json::from_str(&cfg_str)
            .context("parsing model config")?;

        let t = Instant::now();
        let model = load_model::<B>(
            &model_cfg,
            weights_path.to_str().context("weights path not valid UTF-8")?,
            &device,
        )?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        Ok((Self { model, model_cfg, device }, ms))
    }

    /// Create encoder from a pre-built model.
    pub fn from_model(model: SleepLM<B>, model_cfg: ModelConfig, device: B::Device) -> Self {
        Self { model, model_cfg, device }
    }

    pub fn describe(&self) -> String {
        let c = &self.model_cfg;
        let b = &c.biosignals_cfg;
        format!(
            "SleepLM  embed_dim={}  channels={}  patches={}  blocks={}  decoder_tokens={}",
            c.embed_dim, b.input_channels, b.num_patches(),
            b.transformer_layers, b.decoder_tokens,
        )
    }

    /// Encode a single epoch to a contrastive embedding.
    pub fn encode(&self, batch: &InputBatch<B>) -> anyhow::Result<EpochEmbedding> {
        let signal = channel_wise_normalize(batch.signal.clone());
        let (embedding, _tokens) = self.model.encode_biosignals(signal);

        // L2 normalize
        let norm = embedding.clone().powf_scalar(2.0).sum_dim(1).sqrt();
        let embedding = embedding / norm;

        let embed_dim = self.model_cfg.embed_dim;
        let output_vec = embedding.squeeze::<1>()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("embedding→vec: {e:?}"))?;

        Ok(EpochEmbedding {
            embedding: output_vec,
            embed_dim,
        })
    }

    /// Encode text to a contrastive embedding.
    pub fn encode_text(&self, text_tokens: Tensor<B, 2, Int>) -> anyhow::Result<Vec<f32>> {
        let latent = self.model.encode_text(text_tokens);

        // L2 normalize
        let norm = latent.clone().powf_scalar(2.0).sum_dim(1).sqrt();
        let latent = latent / norm;

        let output_vec = latent.squeeze::<1>()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("text_embedding→vec: {e:?}"))?;

        Ok(output_vec)
    }

    /// Compute cosine similarity between biosignal and text embeddings.
    pub fn similarity(bio_emb: &[f32], text_emb: &[f32]) -> f32 {
        assert_eq!(bio_emb.len(), text_emb.len());
        let dot: f32 = bio_emb.iter().zip(text_emb.iter()).map(|(a, b)| a * b).sum();
        let na: f32 = bio_emb.iter().map(|a| a * a).sum::<f32>().sqrt();
        let nb: f32 = text_emb.iter().map(|b| b * b).sum::<f32>().sqrt();
        if na < 1e-8 || nb < 1e-8 { 0.0 } else { dot / (na * nb) }
    }

    pub fn device(&self) -> &B::Device { &self.device }
    pub fn model(&self) -> &SleepLM<B> { &self.model }
}
