/// Generate synthetic PSG data for testing sleeplm without real data.
///
/// Usage:
///   cargo run --bin gen_sample_psg --release -- --output sample.bin
///   cargo run --bin gen_sample_psg --release -- --output sample.bin --epochs 10

use clap::Parser;
use std::io::Write;

#[derive(Parser, Debug)]
#[command(about = "Generate synthetic PSG data for testing")]
struct Args {
    /// Output binary file path (f32 little-endian).
    #[arg(long, short, default_value = "data/sample_psg.bin")]
    output: String,

    /// Number of 30-second epochs to generate.
    #[arg(long, default_value_t = 4)]
    epochs: usize,
}

const CHANNEL_NAMES: &[&str] = &[
    "ECG", "ABD", "THX", "AF",
    "EOG_E1", "EOG_E2",
    "EEG_C3", "EEG_C4",
    "EMG_Chin", "POS",
];

const N_CHANNELS: usize = 10;
const SAMPLES: usize = 1920; // 30s @ 64 Hz
const SFREQ: f32 = 64.0;

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let dt = 1.0 / SFREQ;
    let mut all_data = Vec::with_capacity(args.epochs * N_CHANNELS * SAMPLES);
    let mut noise_state: u32 = 0xDEAD_BEEF;

    for epoch in 0..args.epochs {
        let t_offset = epoch as f32 * 30.0;

        for ch in 0..N_CHANNELS {
            for t_idx in 0..SAMPLES {
                let time = t_offset + t_idx as f32 * dt;
                let val = match ch {
                    0 => { // ECG — simulated QRS-like
                        let phase = (time * 1.2 * std::f32::consts::TAU).fract();
                        if phase < 0.05 { 1.0 } else if phase < 0.1 { -0.3 } else { 0.0 }
                    }
                    1 | 2 | 3 => { // Respiratory — slow sinusoidal
                        let freq = 0.2 + ch as f32 * 0.05;
                        (freq * time * std::f32::consts::TAU).sin() * 0.5
                    }
                    4 | 5 => { // EOG — slow eye movements
                        (0.3 * time * std::f32::consts::TAU).sin() * 0.3
                    }
                    6 | 7 => { // EEG — alpha + delta
                        let alpha = (10.0 * time * std::f32::consts::TAU).sin() * 0.2;
                        let delta = (2.0 * time * std::f32::consts::TAU).sin() * 0.5;
                        alpha + delta
                    }
                    8 => { // EMG — noise
                        noise_state ^= noise_state << 13;
                        noise_state ^= noise_state >> 17;
                        noise_state ^= noise_state << 5;
                        (noise_state as f32 / u32::MAX as f32 - 0.5) * 0.1
                    }
                    9 => { // POS — constant (Supine = 2)
                        2.0
                    }
                    _ => 0.0,
                };

                // Add small noise
                noise_state ^= noise_state << 13;
                noise_state ^= noise_state >> 17;
                noise_state ^= noise_state << 5;
                let noise = (noise_state as f32 / u32::MAX as f32 - 0.5) * 0.01;

                all_data.push(val + noise);
            }
        }
    }

    // Write as raw f32 LE
    let mut f = std::fs::File::create(&args.output)?;
    for &val in &all_data {
        f.write_all(&val.to_le_bytes())?;
    }

    let n_total = args.epochs * N_CHANNELS * SAMPLES;
    let size = std::fs::metadata(&args.output)?.len();
    println!("Generated {0} epochs × {N_CHANNELS} channels × {SAMPLES} samples", args.epochs);
    println!("  Shape: [{0}, {N_CHANNELS}, {SAMPLES}]", args.epochs);
    println!("  Channels: {}", CHANNEL_NAMES.join(", "));
    println!("  → {} ({:.1} KB, {} f32 values)", args.output, size as f64 / 1024.0, n_total);

    Ok(())
}
