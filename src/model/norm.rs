/// Normalization layers for SleepLM.
///
/// SleepLM uses:
/// - RMSNorm in the biosignals encoder (dual-axis transformer)
/// - LayerNorm in the text encoder and decoder

use burn::prelude::*;
type Param<T> = burn::module::Param<T>;
use burn::nn::{LayerNorm, LayerNormConfig};

// ── RMSNorm ───────────────────────────────────────────────────────────────────

/// Root Mean Square Layer Normalization.
///
/// Python: `class RMSNorm(nn.Module)` in biosignals_coca_model.py
///   output = x * rsqrt(mean(x^2) + eps) * weight
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub eps: f64,
    pub dim: usize,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: Param::initialized(
                burn::module::ParamId::new(),
                Tensor::ones([dim], device),
            ),
            eps,
            dim,
        }
    }

    /// x: [*, dim] → [*, dim]
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // mean(x^2, dim=-1, keepdim=True)
        let variance = (x.clone() * x.clone()).mean_dim(D - 1);
        let rsqrt = (variance + self.eps).sqrt().recip();
        let normed = x * rsqrt;
        // Broadcast weight across all dims except last
        let w = self.weight.val();
        normed * w.unsqueeze()
    }
}

// ── SleepLM LayerNorm ─────────────────────────────────────────────────────────

/// Thin wrapper around burn's LayerNorm for consistent API.
#[derive(Module, Debug)]
pub struct SleepLayerNorm<B: Backend> {
    pub inner: LayerNorm<B>,
}

impl<B: Backend> SleepLayerNorm<B> {
    pub fn new(dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            inner: LayerNormConfig::new(dim).with_epsilon(eps).init(device),
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.inner.forward(x)
    }
}
