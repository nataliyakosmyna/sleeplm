/// Data preparation for SleepLM inference.
///
/// SleepLM input: [B, 10, 1920] — 10 PSG channels, 30s @ 64 Hz.
///
/// Channel order:
///   0: ECG, 1: ABD, 2: THX, 3: AF,
///   4: EOG_Left, 5: EOG_Right,
///   6: EEG_C3_A2, 7: EEG_C4_A1,
///   8: EMG_Chin, 9: POS

use burn::prelude::*;

/// A prepared input batch for SleepLM.
pub struct InputBatch<B: Backend> {
    /// PSG signal: [1, 10, 1920].
    pub signal: Tensor<B, 3>,
    /// Number of epochs in this batch.
    pub n_epochs: usize,
}

/// Per-epoch embedding produced by SleepLM.
#[derive(Debug, Clone)]
pub struct EpochEmbedding {
    /// Global contrastive embedding: [embed_dim].
    pub embedding: Vec<f32>,
    /// Embedding dimension.
    pub embed_dim: usize,
}

/// Channel-wise z-score normalization.
///
/// For each channel independently: (x - mean) / (std + eps)
pub fn channel_wise_normalize<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let mean = x.clone().mean_dim(2); // [B, C, 1]
    let diff = x.clone() - mean.clone();
    let var = (diff.clone() * diff).mean_dim(2);
    let std = (var + 1e-8).sqrt();
    (x - mean) / std
}

/// Build a single-epoch InputBatch from a flat f32 signal.
///
/// signal: [10, 1920] row-major (already z-scored and at 64 Hz)
pub fn build_batch<B: Backend>(
    signal: Vec<f32>,
    device: &B::Device,
) -> InputBatch<B> {
    let signal = Tensor::<B, 2>::from_data(
        TensorData::new(signal, vec![10, 1920]),
        device,
    ).unsqueeze_dim::<3>(0); // [1, 10, 1920]

    InputBatch { signal, n_epochs: 1 }
}

/// Build a multi-epoch InputBatch.
///
/// signal: [N, 10, 1920] row-major
pub fn build_batch_multi<B: Backend>(
    signal: Vec<f32>,
    n_epochs: usize,
    device: &B::Device,
) -> InputBatch<B> {
    let signal = Tensor::<B, 3>::from_data(
        TensorData::new(signal, vec![n_epochs, 10, 1920]),
        device,
    );

    InputBatch { signal, n_epochs }
}

/// Body position encoding for the POS channel.
pub fn encode_position(position: &str) -> f32 {
    match position {
        "Right" => 0.0,
        "Left" => 1.0,
        "Supine" => 2.0,
        "Prone" => 3.0,
        "Upright" => 4.0,
        _ => -1.0, // Other/Unknown
    }
}
