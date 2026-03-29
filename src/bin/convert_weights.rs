/// Convert SleepLM PyTorch checkpoint (.pt) to safetensors format.
///
/// This is a placeholder — actual conversion requires Python since
/// the .pt format needs pickle deserialization.
///
/// Usage:
///   python3 scripts/convert_to_safetensors.py --input model_checkpoint.pt --output model.safetensors

use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "Convert SleepLM PyTorch checkpoint to safetensors")]
struct Args {
    /// Input PyTorch checkpoint (.pt).
    #[arg(long)]
    input: String,

    /// Output safetensors file.
    #[arg(long)]
    output: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    eprintln!("PyTorch .pt → safetensors conversion requires Python.");
    eprintln!("Run:");
    eprintln!("  python3 scripts/convert_to_safetensors.py --input {} --output {}",
        args.input, args.output);
    eprintln!();
    eprintln!("Install requirements first:");
    eprintln!("  pip install torch safetensors");

    Ok(())
}
