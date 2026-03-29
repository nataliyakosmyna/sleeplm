//! # sleeplm — SleepLM Sleep-Language Foundation Model inference in Rust
//!
//! Pure-Rust inference for the SleepLM (Sleep Language Model) CoCa architecture,
//! built on [Burn 0.20](https://burn.dev).
//!
//! SleepLM aligns polysomnography (PSG) biosignals with natural language using
//! a contrastive-captioning architecture (CoCa). It encodes 30-second sleep
//! epochs (10 channels × 1920 samples @ 64 Hz) into a shared embedding space
//! with text, enabling:
//!
//! - **Signal–text retrieval** via cosine similarity
//! - **Targeted caption generation** conditioned on modality tokens
//!   (brain, heart, respiratory, position/muscle)
//!
//! ## Architecture
//!
//! - **Biosignals encoder**: Dual-axis transformer (channel + temporal attention)
//!   with RoPE, SwiGLU, RMSNorm, and CoCa-style attentional pooling
//! - **Text encoder**: CLIP-style causal transformer with BPE tokenization
//! - **Text decoder**: Cross-attention transformer (text → biosignal tokens)
//!   with prefix-causal masking for modality conditioning
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use sleeplm::SleepLMEncoder;
//!
//! let (model, _ms) = SleepLMEncoder::<B>::load(
//!     Path::new("config.json"),
//!     Path::new("model.safetensors"),
//!     device,
//! )?;
//! ```

pub mod config;
pub mod data;
pub mod encoder;
pub mod model;
pub mod weights;

// Flat re-exports
pub use encoder::SleepLMEncoder;
pub use data::EpochEmbedding;
pub use config::{ModelConfig, BiosignalsCfg, TextCfg, MultimodalCfg};
pub use config::{CHANNEL_NAMES, MODALITY_NAMES, STAGE_EVENT_IDX, modality_index};
pub use data::{InputBatch, build_batch, build_batch_multi, channel_wise_normalize, encode_position};
pub use weights::WeightMap;
