/// Model and runtime configuration for SleepLM inference.
///
/// `ModelConfig` mirrors the Python SleepLM hyperparameters from the JSON config.
/// Field names match `sleep_coca_base_dualtransformer.json`.

use serde::Deserialize;

// ── BiosignalsCfg ─────────────────────────────────────────────────────────────

/// Configuration for the biosignals (PSG) encoder.
#[derive(Debug, Clone, Deserialize)]
pub struct BiosignalsCfg {
    /// Architecture type: "pure_transformer" or "conv_transformer".
    #[serde(default = "default_architecture")]
    pub architecture: String,

    /// Number of input channels (10 for standard PSG).
    #[serde(default = "default_input_channels")]
    pub input_channels: usize,

    /// Length of input time series (1920 = 30s @ 64 Hz).
    #[serde(default = "default_signal_length")]
    pub signal_length: usize,

    /// Sampling rate in Hz.
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: usize,

    /// Patch size for the pure_transformer architecture.
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,

    /// Conv embedding dimension for patch tokenisation.
    #[serde(default = "default_conv_embed_dim")]
    pub conv_embed_dim: usize,

    /// Number of temporal attention layers per dual-axis block.
    #[serde(default = "default_num_temporal_layers")]
    pub num_temporal_layers: usize,

    /// Activation function: "swiglu", "gelu", "relu".
    #[serde(default = "default_activation")]
    pub activation: String,

    /// Normalization type: "rmsnorm" or "layernorm".
    #[serde(default = "default_norm_type")]
    pub norm_type: String,

    /// Whether to use bias in MLP layers.
    #[serde(default)]
    pub mlp_bias: bool,

    /// Share channel RoPE across transformer blocks.
    #[serde(default = "default_true")]
    pub share_channel_rope: bool,

    /// Number of dual-axis transformer blocks.
    #[serde(default = "default_transformer_layers")]
    pub transformer_layers: usize,

    /// Transformer hidden dimension.
    #[serde(default = "default_transformer_width")]
    pub transformer_width: usize,

    /// Number of attention heads.
    #[serde(default = "default_transformer_heads")]
    pub transformer_heads: usize,

    /// MLP expansion ratio.
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f64,

    /// Pooling type: "attn", "avg", "max", "cls".
    #[serde(default = "default_pool_type")]
    pub pool_type: String,

    /// Dropout rate (ignored at inference).
    #[serde(default = "default_dropout")]
    pub dropout: f64,

    /// Number of decoder tokens from the attentional pooler.
    #[serde(default = "default_decoder_tokens")]
    pub decoder_tokens: usize,
}

fn default_architecture() -> String { "pure_transformer".into() }
fn default_input_channels() -> usize { 10 }
fn default_signal_length() -> usize { 1920 }
fn default_sampling_rate() -> usize { 64 }
fn default_patch_size() -> usize { 16 }
fn default_conv_embed_dim() -> usize { 256 }
fn default_num_temporal_layers() -> usize { 1 }
fn default_activation() -> String { "swiglu".into() }
fn default_norm_type() -> String { "rmsnorm".into() }
fn default_true() -> bool { true }
fn default_transformer_layers() -> usize { 3 }
fn default_transformer_width() -> usize { 768 }
fn default_transformer_heads() -> usize { 12 }
fn default_mlp_ratio() -> f64 { 3.0 }
fn default_pool_type() -> String { "attn".into() }
fn default_dropout() -> f64 { 0.1 }
fn default_decoder_tokens() -> usize { 32 }

impl Default for BiosignalsCfg {
    fn default() -> Self {
        Self {
            architecture: default_architecture(),
            input_channels: default_input_channels(),
            signal_length: default_signal_length(),
            sampling_rate: default_sampling_rate(),
            patch_size: default_patch_size(),
            conv_embed_dim: default_conv_embed_dim(),
            num_temporal_layers: default_num_temporal_layers(),
            activation: default_activation(),
            norm_type: default_norm_type(),
            mlp_bias: false,
            share_channel_rope: true,
            transformer_layers: default_transformer_layers(),
            transformer_width: default_transformer_width(),
            transformer_heads: default_transformer_heads(),
            mlp_ratio: default_mlp_ratio(),
            pool_type: default_pool_type(),
            dropout: default_dropout(),
            decoder_tokens: default_decoder_tokens(),
        }
    }
}

impl BiosignalsCfg {
    /// Number of temporal patches per channel.
    pub fn num_patches(&self) -> usize {
        self.signal_length / self.patch_size
    }

    /// Head dimension for attention.
    pub fn head_dim(&self) -> usize {
        self.transformer_width / self.transformer_heads
    }
}

// ── TextCfg ───────────────────────────────────────────────────────────────────

/// Configuration for the CLIP-style text encoder.
#[derive(Debug, Clone, Deserialize)]
pub struct TextCfg {
    #[serde(default = "default_context_length")]
    pub context_length: usize,

    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    #[serde(default = "default_text_layers")]
    pub layers: usize,

    #[serde(default = "default_text_heads")]
    pub heads: usize,

    #[serde(default = "default_text_width")]
    pub width: usize,

    /// Whether to add a CLS embedding (appended to the token sequence).
    #[serde(default)]
    pub embed_cls: bool,

    /// Whether to output per-token embeddings (needed for CoCa decoder).
    #[serde(default)]
    pub output_tokens: bool,
}

fn default_context_length() -> usize { 256 }
fn default_vocab_size() -> usize { 49408 }
fn default_text_layers() -> usize { 12 }
fn default_text_heads() -> usize { 12 }
fn default_text_width() -> usize { 768 }

impl Default for TextCfg {
    fn default() -> Self {
        Self {
            context_length: default_context_length(),
            vocab_size: default_vocab_size(),
            layers: default_text_layers(),
            heads: default_text_heads(),
            width: default_text_width(),
            embed_cls: true,
            output_tokens: true,
        }
    }
}

// ── MultimodalCfg ─────────────────────────────────────────────────────────────

/// Configuration for the cross-attention text decoder.
#[derive(Debug, Clone, Deserialize)]
pub struct MultimodalCfg {
    #[serde(default = "default_mm_width")]
    pub width: usize,

    #[serde(default = "default_mm_context_length")]
    pub context_length: usize,

    #[serde(default = "default_mm_mlp_ratio")]
    pub mlp_ratio: f64,

    #[serde(default = "default_mm_layers")]
    pub layers: usize,

    #[serde(default = "default_mm_heads")]
    pub heads: usize,
}

fn default_mm_width() -> usize { 768 }
fn default_mm_context_length() -> usize { 256 }
fn default_mm_mlp_ratio() -> f64 { 4.0 }
fn default_mm_layers() -> usize { 12 }
fn default_mm_heads() -> usize { 12 }

impl Default for MultimodalCfg {
    fn default() -> Self {
        Self {
            width: default_mm_width(),
            context_length: default_mm_context_length(),
            mlp_ratio: default_mm_mlp_ratio(),
            layers: default_mm_layers(),
            heads: default_mm_heads(),
        }
    }
}

// ── Top-level ModelConfig ─────────────────────────────────────────────────────

/// Full model configuration, mirroring `sleep_coca_base_dualtransformer.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// Shared embedding dimension for contrastive alignment.
    #[serde(default = "default_embed_dim")]
    pub embed_dim: usize,

    /// Biosignals encoder config.
    pub biosignals_cfg: BiosignalsCfg,

    /// Text encoder config.
    pub text_cfg: TextCfg,

    /// Multimodal decoder config.
    pub multimodal_cfg: MultimodalCfg,

    /// Decoder type: "cross_attention" or "concat".
    #[serde(default = "default_decoder_type")]
    pub decoder_type: String,

    /// Number of condition/modality embeddings (5 = 4 modalities + stage_event).
    #[serde(default = "default_num_caption_channels")]
    pub num_caption_channels: usize,

    /// Length of the condition prefix in the decoder.
    #[serde(default = "default_prefix_len")]
    pub prefix_len: usize,
}

fn default_embed_dim() -> usize { 512 }
fn default_decoder_type() -> String { "cross_attention".into() }
fn default_num_caption_channels() -> usize { 5 }
fn default_prefix_len() -> usize { 2 }

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            embed_dim: default_embed_dim(),
            biosignals_cfg: BiosignalsCfg::default(),
            text_cfg: TextCfg::default(),
            multimodal_cfg: MultimodalCfg::default(),
            decoder_type: default_decoder_type(),
            num_caption_channels: default_num_caption_channels(),
            prefix_len: default_prefix_len(),
        }
    }
}

// ── Channel / Modality Constants ──────────────────────────────────────────────

/// Standard PSG channel order expected by SleepLM.
pub const CHANNEL_NAMES: &[&str] = &[
    "ECG", "ABD", "THX", "AF",
    "EOG_E1", "EOG_E2",
    "EEG_C3", "EEG_C4",
    "EMG_Chin", "POS",
];

/// Modality conditioning tokens.
pub const MODALITY_NAMES: &[&str] = &[
    "brain",
    "heart",
    "respiratory",
    "position_muscle",
];

/// Index of the "stage_event" token (= len(MODALITY_NAMES)).
pub const STAGE_EVENT_IDX: usize = 4;

/// Body-position encoding for the POS channel.
pub const POSITION_ENCODING: &[(i32, &str)] = &[
    ( 0, "Right"),
    ( 1, "Left"),
    ( 2, "Supine"),
    ( 3, "Prone"),
    ( 4, "Upright"),
    (-1, "Other/Unknown"),
];

/// Look up modality index by name.
pub fn modality_index(name: &str) -> Option<usize> {
    match name {
        "brain" => Some(0),
        "heart" => Some(1),
        "respiratory" => Some(2),
        "position_muscle" => Some(3),
        "stage_event" => Some(STAGE_EVENT_IDX),
        _ => None,
    }
}
