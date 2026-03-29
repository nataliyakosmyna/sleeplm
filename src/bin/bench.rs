/// SleepLM inference benchmark — CPU + GPU in one binary.
///
/// Build (both backends):
///   cargo build --release --bin bench --features metal

use std::time::Instant;
use clap::Parser;

#[derive(Parser)]
struct Args {
    /// Which backends to run: cpu, gpu, all
    #[arg(long, default_value = "all")]
    backend: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let run_cpu = args.backend == "all" || args.backend == "cpu";

    println!("backend,component,batch,warmup,iters,mean_ms,std_ms,min_ms,max_ms");

    #[cfg(feature = "ndarray")]
    if run_cpu {
        run_bench::<burn::backend::NdArray>(
            "CPU-NdArray",
            &burn::backend::ndarray::NdArrayDevice::Cpu,
        )?;
    }

    #[cfg(feature = "wgpu")]
    {
        let run_gpu = args.backend == "all" || args.backend == "gpu";
        if run_gpu {
            run_bench::<burn::backend::Wgpu>(
                "GPU-Metal",
                &burn::backend::wgpu::WgpuDevice::DefaultDevice,
            )?;
        }
    }

    Ok(())
}

fn run_bench<B: burn::prelude::Backend>(
    name: &str,
    device: &B::Device,
) -> anyhow::Result<()> {
    use burn::prelude::*;
    use sleeplm::config::ModelConfig;
    use sleeplm::model::sleeplm::SleepLM;

    let cfg: ModelConfig = serde_json::from_str(include_str!(
        "../../../../../data/sleep_coca_base_dualtransformer.json"
    ))?;

    eprintln!("[{name}] building model…");
    let model = SleepLM::<B>::new(&cfg, device);

    let batches: &[usize] = &[1, 4];
    let warmup = 1;
    let iters = 3;

    for &bs in batches {
        // bio_encode
        {
            let sig = Tensor::<B, 3>::zeros([bs, 10, 1920], device);
            for _ in 0..warmup { let _ = model.encode_biosignals(sig.clone()); }
            let t = time_n(iters, || { let _ = model.encode_biosignals(sig.clone()); });
            csv(name, "bio_encode", bs, &t);
        }
        // text_encode
        {
            let txt = Tensor::<B, 2, Int>::zeros([bs, 77], device);
            for _ in 0..warmup { let _ = model.encode_text(txt.clone()); }
            let t = time_n(iters, || { let _ = model.encode_text(txt.clone()); });
            csv(name, "text_encode", bs, &t);
        }
        // full_forward
        {
            let sig = Tensor::<B, 3>::zeros([bs, 10, 1920], device);
            let txt = Tensor::<B, 2, Int>::zeros([bs, 77], device);
            let ch  = Tensor::<B, 2, Int>::zeros([bs, 2], device);
            for _ in 0..warmup { let _ = model.forward(sig.clone(), txt.clone(), Some(ch.clone())); }
            let t = time_n(iters, || { let _ = model.forward(sig.clone(), txt.clone(), Some(ch.clone())); });
            csv(name, "full_forward", bs, &t);
        }
    }
    Ok(())
}

fn time_n(n: usize, mut f: impl FnMut()) -> Vec<f64> {
    (0..n).map(|_| { let t = Instant::now(); f(); t.elapsed().as_secs_f64() * 1000.0 }).collect()
}

fn csv(backend: &str, comp: &str, bs: usize, times: &[f64]) {
    let n = times.len() as f64;
    let mean = times.iter().sum::<f64>() / n;
    let std = (times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n).sqrt();
    let min = times.iter().cloned().fold(f64::MAX, f64::min);
    let max = times.iter().cloned().fold(f64::MIN, f64::max);
    println!("{backend},{comp},{bs},{n},{n},{mean:.2},{std:.2},{min:.2},{max:.2}");
}
