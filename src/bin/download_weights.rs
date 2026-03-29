/// download_weights — fetch SleepLM model weights from HuggingFace.
///
/// Usage:
///   cargo run --bin download_weights --release --features hf-download

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "Download SleepLM weights from HuggingFace (yang-ai-lab/SleepLM-Base)")]
struct Args {
    /// HuggingFace repo ID.
    #[arg(long, default_value = "yang-ai-lab/SleepLM-Base")]
    repo: String,

    /// Filename to download.
    #[arg(long, default_value = "model_checkpoint.pt")]
    filename: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    print!("Downloading {} from {} … ", args.filename, args.repo);
    let path = download(&args.repo, &args.filename)?;
    println!("{}", path.display());

    println!("\nNote: The downloaded checkpoint is in PyTorch format.");
    println!("Convert to safetensors with:");
    println!("  cargo run --bin convert_weights -- --input {} --output model.safetensors", path.display());

    Ok(())
}

#[cfg(feature = "hf-download")]
fn download(repo: &str, filename: &str) -> Result<std::path::PathBuf> {
    use hf_hub::api::sync::ApiBuilder;
    let api = ApiBuilder::new().with_progress(true).build()?;
    Ok(api.model(repo.to_string()).get(filename)?)
}

#[cfg(not(feature = "hf-download"))]
fn download(_repo: &str, _filename: &str) -> Result<std::path::PathBuf> {
    anyhow::bail!("Compile with --features hf-download to enable weight downloading")
}
