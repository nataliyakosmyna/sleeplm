/// Channel-independent patching layer for SleepLM.
///
/// Python: `class ChannelPatching(nn.Module)` in biosignals_coca_model.py
///
/// Uses Conv1d(1, conv_embed_dim, kernel_size=patch_size, stride=patch_size)
/// applied independently to each channel.

use burn::prelude::*;
use burn::nn::conv::{Conv1d, Conv1dConfig};

#[derive(Module, Debug)]
pub struct ChannelPatching<B: Backend> {
    /// Single Conv1d applied to all channels: in=1, out=conv_embed_dim.
    pub conv: Conv1d<B>,
    pub patch_size: usize,
    pub conv_embed_dim: usize,
    pub num_channels: usize,
}

impl<B: Backend> ChannelPatching<B> {
    pub fn new(
        patch_size: usize,
        conv_embed_dim: usize,
        num_channels: usize,
        device: &B::Device,
    ) -> Self {
        let conv = Conv1dConfig::new(1, conv_embed_dim, patch_size)
            .with_stride(patch_size)
            .with_padding(burn::nn::PaddingConfig1d::Valid)
            .with_bias(true)
            .init(device);

        Self { conv, patch_size, conv_embed_dim, num_channels }
    }

    /// x: [B, C, signal_length] → [B, C, num_patches, conv_embed_dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, c, t] = x.dims();
        let num_patches = t / self.patch_size;

        // Reshape to process all channels: [B*C, 1, signal_length]
        let x = x.reshape([b * c, 1, t]);

        // Apply conv patching: [B*C, conv_embed_dim, num_patches]
        let patched = self.conv.forward(x);

        // Reshape: [B, C, conv_embed_dim, num_patches] → [B, C, num_patches, conv_embed_dim]
        patched
            .reshape([b, c, self.conv_embed_dim, num_patches])
            .swap_dims(2, 3)
    }
}
