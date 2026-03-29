/// Example: encode a PSG epoch to a contrastive embedding.
///
/// cargo run --example embed --release -- --weights model.safetensors --config config.json

use std::path::Path;
use clap::Parser;
use sleeplm::{SleepLMEncoder, data};

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
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let dev = device();

    let (model, ms) = SleepLMEncoder::<B>::load(
        Path::new(&args.config),
        Path::new(&args.weights),
        dev.clone(),
    )?;
    println!("Loaded: {} ({ms:.0} ms)", model.describe());

    // Dummy epoch: zeros
    let signal = vec![0.0f32; 10 * 1920];
    let batch = data::build_batch::<B>(signal, &dev);

    let emb = model.encode(&batch)?;
    println!("Embedding dim: {}", emb.embed_dim);
    println!("First 8 values: {:?}", &emb.embedding[..8.min(emb.embed_dim)]);

    // Compute self-similarity (should be 1.0)
    let sim = SleepLMEncoder::<B>::similarity(&emb.embedding, &emb.embedding);
    println!("Self-similarity: {sim:.6}");

    Ok(())
}
