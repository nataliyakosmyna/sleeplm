/// Example: demonstrate the model architecture for caption generation.
///
/// Note: Full autoregressive generation requires a BPE tokenizer and beam search.
/// This example shows the forward pass through all three towers.
///
/// cargo run --example caption --release -- --weights model.safetensors --config config.json

use std::path::Path;
use clap::Parser;
use sleeplm::{SleepLMEncoder, data, modality_index, STAGE_EVENT_IDX};
use burn::prelude::*;

#[cfg(feature = "ndarray")]
mod backend {
    pub use burn::backend::NdArray as B;
    pub type Device = burn::backend::ndarray::NdArrayDevice;
    pub fn device() -> Device { Device::Cpu }
}

#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
mod backend {
    pub use burn::backend::{Wgpu as B, wgpu::WgpuDevice as Device};
    pub fn device() -> Device { Device::DefaultDevice }
}

use backend::{B, device};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    weights: String,
    #[arg(long)]
    config: String,
    /// Modality to condition on: brain, heart, respiratory, position_muscle, stage_event
    #[arg(long, default_value = "brain")]
    modality: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let dev = device();

    let (encoder, ms) = SleepLMEncoder::<B>::load(
        Path::new(&args.config),
        Path::new(&args.weights),
        dev.clone(),
    )?;
    println!("Loaded: {} ({ms:.0} ms)", encoder.describe());

    let mod_idx = modality_index(&args.modality)
        .ok_or_else(|| anyhow::anyhow!("Unknown modality: {}", args.modality))?;
    println!("Modality: {} (index {})", args.modality, mod_idx);

    // Encode a dummy signal
    let signal = vec![0.0f32; 10 * 1920];
    let batch = data::build_batch::<B>(signal, &dev);
    let emb = encoder.encode(&batch)?;
    println!("Signal embedding: {} dims, first 4: {:?}",
        emb.embed_dim, &emb.embedding[..4.min(emb.embed_dim)]);

    println!("\nTo generate captions, use the Python demo notebook with:");
    println!("  model.generate(biosignals=x, channel_indices=[[{}]], ...)", mod_idx);
    println!("  Modalities: brain=0, heart=1, respiratory=2, position_muscle=3, stage_event={STAGE_EVENT_IDX}");

    Ok(())
}
