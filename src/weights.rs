/// Load pretrained SleepLM weights from a safetensors file.
///
/// The checkpoint is originally a PyTorch `.pt` file. Use `convert_weights`
/// binary to convert it to safetensors format before loading.
///
/// Weight key patterns (after stripping `module.` prefix):
///
/// Biosignals encoder:
///   biosignals.patching.conv_patching.weight
///   biosignals.embed_projection.weight/bias
///   biosignals.channel_id_embed.weight
///   biosignals.shared_channel_rope.freqs
///   biosignals.transformer_blocks.{i}.channel_attention.{q,k,v,out}_proj.weight
///   biosignals.transformer_blocks.{i}.channel_norm.weight
///   biosignals.transformer_blocks.{i}.channel_mlp.*
///   biosignals.transformer_blocks.{i}.temporal_attention_layers.{j}.*
///   biosignals.transformer_blocks.{i}.temporal_norms.{j}.weight
///   biosignals.transformer_blocks.{i}.temporal_mlps.{j}.*
///   biosignals.ln_final.weight
///   biosignals.contrastive_pooler.{query_tokens,attn.*}
///   biosignals.decoder_pooler.{query_tokens,attn.*}
///   biosignals.proj_to_embed.weight/bias
///
/// Text encoder:
///   text.token_embedding.weight
///   text.positional_embedding
///   text.cls_emb
///   text.transformer.resblocks.{i}.attn.{in_proj_weight,in_proj_bias,out_proj.*}
///   text.transformer.resblocks.{i}.ln_1.{weight,bias}
///   text.transformer.resblocks.{i}.mlp.c_fc.{weight,bias}
///   text.transformer.resblocks.{i}.mlp.c_proj.{weight,bias}
///   text.transformer.resblocks.{i}.ln_2.{weight,bias}
///   text.ln_final.{weight,bias}
///   text.text_projection
///
/// Text decoder:
///   text_decoder.resblocks.{i}.attn.{in_proj_weight,...}
///   text_decoder.resblocks.{i}.ln_1.*, ln_2.*, mlp.*
///   text_decoder.cross_attn.{i}.attn.{in_proj_weight,...}
///   text_decoder.cross_attn.{i}.ln_1.*, ln_1_kv.*, ln_2.*, mlp.*
///   text_decoder.ln_final.{weight,bias}
///   text_decoder.text_projection
///
/// Top-level:
///   logit_scale
///   channel_embeddings
///   padding_embedding

use std::collections::HashMap;
use burn::prelude::*;
use half::bf16;
use safetensors::SafeTensors;

use crate::config::ModelConfig;
use crate::model::sleeplm::SleepLM;

// ── Raw tensor map ────────────────────────────────────────────────────────────

pub struct WeightMap {
    pub tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl WeightMap {
    /// Load all tensors from a safetensors file.
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let st = SafeTensors::deserialize(&bytes)?;
        let mut tensors = HashMap::with_capacity(st.len());

        for (raw_key, view) in st.tensors() {
            let key = raw_key
                .strip_prefix("module.")
                .unwrap_or(raw_key.as_str())
                .to_string();

            let shape: Vec<usize> = view.shape().to_vec();
            let data = view.data();

            let f32s: Vec<f32> = match view.dtype() {
                safetensors::Dtype::BF16 => data
                    .chunks_exact(2)
                    .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::F32 => data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect(),
                safetensors::Dtype::F16 => data
                    .chunks_exact(2)
                    .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect(),
                other => anyhow::bail!("unsupported dtype {:?} for key {key}", other),
            };

            tensors.insert(key, (f32s, shape));
        }

        Ok(Self { tensors })
    }

    /// Take a tensor by key, removing it from the map.
    pub fn take<B: Backend, const N: usize>(
        &mut self,
        key: &str,
        device: &B::Device,
    ) -> anyhow::Result<Tensor<B, N>> {
        let (data, shape) = self.tensors.remove(key)
            .ok_or_else(|| anyhow::anyhow!("weight key not found: {key}"))?;
        if shape.len() != N {
            anyhow::bail!("rank mismatch for {key}: expected {N}, got {}", shape.len());
        }
        Ok(Tensor::<B, N>::from_data(TensorData::new(data, shape), device))
    }

    pub fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    pub fn print_keys(&self) {
        let mut keys: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        keys.sort();
        for k in keys {
            let (_, s) = &self.tensors[k];
            println!("  {k:80}  {s:?}");
        }
    }

    pub fn remaining_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.tensors.keys().cloned().collect();
        keys.sort();
        keys
    }
}

// ── Weight assignment helpers ─────────────────────────────────────────────────

/// PyTorch [out, in] → burn [in, out]
fn set_linear_wb<B: Backend>(linear: &mut burn::nn::Linear<B>, w: Tensor<B, 2>, b: Tensor<B, 1>) {
    linear.weight = linear.weight.clone().map(|_| w.transpose());
    if let Some(ref bias) = linear.bias {
        linear.bias = Some(bias.clone().map(|_| b));
    }
}

fn set_linear_w<B: Backend>(linear: &mut burn::nn::Linear<B>, w: Tensor<B, 2>) {
    linear.weight = linear.weight.clone().map(|_| w.transpose());
}

fn set_layernorm<B: Backend>(norm: &mut crate::model::norm::SleepLayerNorm<B>, w: Tensor<B, 1>, b: Tensor<B, 1>) {
    norm.inner.gamma = norm.inner.gamma.clone().map(|_| w);
    if let Some(ref beta) = norm.inner.beta {
        norm.inner.beta = Some(beta.clone().map(|_| b));
    }
}

fn set_rmsnorm<B: Backend>(norm: &mut crate::model::norm::RmsNorm<B>, w: Tensor<B, 1>) {
    norm.weight = norm.weight.clone().map(|_| w);
}

fn set_conv1d_w<B: Backend>(conv: &mut burn::nn::conv::Conv1d<B>, w: Tensor<B, 3>) {
    conv.weight = conv.weight.clone().map(|_| w);
}

/// Load fused nn.MultiheadAttention weights.
/// PyTorch: in_proj_weight [3*D, D], in_proj_bias [3*D]
///          out_proj.weight [D, D], out_proj.bias [D]
///
/// Maps to our separate Q/K/V projection or fused in_proj Linear.
fn load_fused_mha_to_split<B: Backend>(
    wm: &mut WeightMap,
    q_proj: &mut burn::nn::Linear<B>,
    k_proj: &mut burn::nn::Linear<B>,
    v_proj: &mut burn::nn::Linear<B>,
    out_proj: &mut burn::nn::Linear<B>,
    prefix: &str,
    device: &B::Device,
) -> anyhow::Result<()> {
    if let Ok(w) = wm.take::<B, 2>(&format!("{prefix}.in_proj_weight"), device) {
        let [three_d, _d] = w.dims();
        let dim = three_d / 3;
        let b = wm.take::<B, 1>(&format!("{prefix}.in_proj_bias"), device)?;

        let wq = w.clone().narrow(0, 0, dim);
        let wk = w.clone().narrow(0, dim, dim);
        let wv = w.narrow(0, dim * 2, dim);
        let bq = b.clone().narrow(0, 0, dim);
        let bk = b.clone().narrow(0, dim, dim);
        let bv = b.narrow(0, dim * 2, dim);

        set_linear_wb(q_proj, wq, bq);
        set_linear_wb(k_proj, wk, bk);
        set_linear_wb(v_proj, wv, bv);
    }

    if let (Ok(w), Ok(b)) = (
        wm.take::<B, 2>(&format!("{prefix}.out_proj.weight"), device),
        wm.take::<B, 1>(&format!("{prefix}.out_proj.bias"), device),
    ) {
        set_linear_wb(out_proj, w, b);
    }

    Ok(())
}

/// Load fused MHA into a single in_proj Linear (for text encoder/decoder).
fn load_fused_mha_single<B: Backend>(
    wm: &mut WeightMap,
    in_proj: &mut burn::nn::Linear<B>,
    out_proj: &mut burn::nn::Linear<B>,
    prefix: &str,
    device: &B::Device,
) -> anyhow::Result<()> {
    if let (Ok(w), Ok(b)) = (
        wm.take::<B, 2>(&format!("{prefix}.in_proj_weight"), device),
        wm.take::<B, 1>(&format!("{prefix}.in_proj_bias"), device),
    ) {
        set_linear_wb(in_proj, w, b);
    }
    if let (Ok(w), Ok(b)) = (
        wm.take::<B, 2>(&format!("{prefix}.out_proj.weight"), device),
        wm.take::<B, 1>(&format!("{prefix}.out_proj.bias"), device),
    ) {
        set_linear_wb(out_proj, w, b);
    }
    Ok(())
}

// ── Full model loader ─────────────────────────────────────────────────────────

/// Load a SleepLM model from a safetensors file.
pub fn load_model<B: Backend>(
    cfg: &ModelConfig,
    weights_path: &str,
    device: &B::Device,
) -> anyhow::Result<SleepLM<B>> {
    let mut wm = WeightMap::from_file(weights_path)?;
    eprintln!("Loading {} weight tensors...", wm.tensors.len());
    let model = load_model_from_wm(cfg, &mut wm, device)?;

    let remaining = wm.remaining_keys();
    if !remaining.is_empty() {
        eprintln!("Warning: {} unused weight keys", remaining.len());
        for k in remaining.iter().take(10) {
            eprintln!("  unused: {k}");
        }
    }

    Ok(model)
}

pub fn load_model_from_wm<B: Backend>(
    cfg: &ModelConfig,
    wm: &mut WeightMap,
    device: &B::Device,
) -> anyhow::Result<SleepLM<B>> {
    let mut model = SleepLM::new(cfg, device);
    load_sleeplm_weights(wm, &mut model, cfg, device)?;
    Ok(model)
}

fn load_sleeplm_weights<B: Backend>(
    wm: &mut WeightMap,
    model: &mut SleepLM<B>,
    cfg: &ModelConfig,
    device: &B::Device,
) -> anyhow::Result<()> {
    // ── Biosignals encoder ──────────────────────────────────────────────
    load_biosignals_encoder(wm, &mut model.biosignals, cfg, device)?;

    // ── Text encoder ────────────────────────────────────────────────────
    load_text_encoder(wm, &mut model.text, cfg, device)?;

    // ── Text decoder ────────────────────────────────────────────────────
    load_text_decoder(wm, &mut model.text_decoder, cfg, device)?;

    // ── Top-level parameters ────────────────────────────────────────────
    // logit_scale is scalar ([]) in Python, we store as [1] in Rust.
    // Handle both shapes.
    if wm.has("logit_scale") {
        let (data, shape) = wm.tensors.remove("logit_scale").unwrap();
        let t = if shape.is_empty() || (shape.len() == 1 && shape[0] == 1) {
            // Scalar or [1] — wrap into [1]
            Tensor::<B, 1>::from_data(TensorData::new(data, vec![1]), device)
        } else {
            Tensor::<B, 1>::from_data(TensorData::new(data, vec![shape.iter().product()]), device)
        };
        model.logit_scale = model.logit_scale.clone().map(|_| t);
    }
    if let Ok(t) = wm.take::<B, 2>("channel_embeddings", device) {
        model.channel_embeddings = model.channel_embeddings.clone().map(|_| t);
    }
    if let Ok(t) = wm.take::<B, 1>("padding_embedding", device) {
        model.padding_embedding = model.padding_embedding.clone().map(|_| t);
    }

    Ok(())
}

fn load_biosignals_encoder<B: Backend>(
    wm: &mut WeightMap,
    enc: &mut crate::model::biosignals_encoder::BiosignalsEncoder<B>,
    _cfg: &ModelConfig,
    device: &B::Device,
) -> anyhow::Result<()> {
    // Patching conv (has bias)
    if let Ok(w) = wm.take::<B, 3>("biosignals.patching.conv_patching.weight", device) {
        set_conv1d_w(&mut enc.patching.conv, w);
    }
    if let Ok(b) = wm.take::<B, 1>("biosignals.patching.conv_patching.bias", device) {
        if let Some(ref bias) = enc.patching.conv.bias {
            enc.patching.conv.bias = Some(bias.clone().map(|_| b));
        }
    }

    // Embed projection
    if let (Ok(w), Ok(b)) = (
        wm.take::<B, 2>("biosignals.embed_projection.weight", device),
        wm.take::<B, 1>("biosignals.embed_projection.bias", device),
    ) {
        set_linear_wb(&mut enc.embed_projection, w, b);
    }

    // Channel ID embedding
    if let Ok(w) = wm.take::<B, 2>("biosignals.channel_id_embed.weight", device) {
        enc.channel_id_embed.weight = enc.channel_id_embed.weight.clone().map(|_| w);
    }

    // Shared channel RoPE frequencies
    if let Ok(t) = wm.take::<B, 1>("biosignals.shared_channel_rope.freqs", device) {
        enc.shared_channel_freqs = enc.shared_channel_freqs.clone().map(|_| t);
    }

    // Dual-axis transformer blocks
    for (i, block) in enc.blocks.iter_mut().enumerate() {
        let p = format!("biosignals.transformer_blocks.{i}");

        // Channel attention
        load_dual_rope_attn(wm, &mut block.channel_attention.q_proj,
            &mut block.channel_attention.k_proj, &mut block.channel_attention.v_proj,
            &mut block.channel_attention.out_proj, &format!("{p}.channel_attention"), device)?;

        // Channel norm + MLP
        if let Ok(w) = wm.take::<B, 1>(&format!("{p}.channel_norm.weight"), device) {
            set_rmsnorm(&mut block.channel_norm, w);
        }
        load_swiglu_mlp(wm, &mut block.channel_mlp, &format!("{p}.channel_mlp"), device)?;
        if let Ok(w) = wm.take::<B, 1>(&format!("{p}.channel_mlp_norm.weight"), device) {
            set_rmsnorm(&mut block.channel_mlp_norm, w);
        }

        // Temporal attention layers
        for (j, t_attn) in block.temporal_attentions.iter_mut().enumerate() {
            load_dual_rope_attn(wm, &mut t_attn.q_proj, &mut t_attn.k_proj,
                &mut t_attn.v_proj, &mut t_attn.out_proj,
                &format!("{p}.temporal_attention_layers.{j}"), device)?;
        }
        for (j, t_norm) in block.temporal_norms.iter_mut().enumerate() {
            if let Ok(w) = wm.take::<B, 1>(&format!("{p}.temporal_norms.{j}.weight"), device) {
                set_rmsnorm(t_norm, w);
            }
        }
        for (j, t_mlp) in block.temporal_mlps.iter_mut().enumerate() {
            load_swiglu_mlp(wm, t_mlp, &format!("{p}.temporal_mlps.{j}"), device)?;
        }
        for (j, t_norm) in block.temporal_mlp_norms.iter_mut().enumerate() {
            if let Ok(w) = wm.take::<B, 1>(&format!("{p}.temporal_mlp_norms.{j}.weight"), device) {
                set_rmsnorm(t_norm, w);
            }
        }
    }

    // Final norm
    if let Ok(w) = wm.take::<B, 1>("biosignals.ln_final.weight", device) {
        set_rmsnorm(&mut enc.ln_final, w);
    }

    // Contrastive pooler
    load_attn_pooler(wm, &mut enc.contrastive_pooler, "biosignals.contrastive_pooler", device)?;

    // Decoder pooler
    load_attn_pooler(wm, &mut enc.decoder_pooler, "biosignals.decoder_pooler", device)?;

    // BaseBiosignalsEncoder.attn_pool (pool_type='attn')
    load_fused_mha_single(
        wm,
        &mut enc.attn_pool.in_proj, &mut enc.attn_pool.out_proj,
        "biosignals.attn_pool", device,
    )?;

    // Projection to embed dim
    if let (Ok(w), Ok(b)) = (
        wm.take::<B, 2>("biosignals.proj_to_embed.weight", device),
        wm.take::<B, 1>("biosignals.proj_to_embed.bias", device),
    ) {
        set_linear_wb(&mut enc.proj_to_embed, w, b);
    }

    Ok(())
}

fn load_dual_rope_attn<B: Backend>(
    wm: &mut WeightMap,
    q_proj: &mut burn::nn::Linear<B>,
    k_proj: &mut burn::nn::Linear<B>,
    v_proj: &mut burn::nn::Linear<B>,
    out_proj: &mut burn::nn::Linear<B>,
    prefix: &str,
    device: &B::Device,
) -> anyhow::Result<()> {
    // DualRoPEAttention has separate q/k/v/out projections
    if let Ok(w) = wm.take::<B, 2>(&format!("{prefix}.q_proj.weight"), device) {
        set_linear_w(q_proj, w);
    }
    if let Ok(w) = wm.take::<B, 2>(&format!("{prefix}.k_proj.weight"), device) {
        set_linear_w(k_proj, w);
    }
    if let Ok(w) = wm.take::<B, 2>(&format!("{prefix}.v_proj.weight"), device) {
        set_linear_w(v_proj, w);
    }
    if let (Ok(w), Ok(b)) = (
        wm.take::<B, 2>(&format!("{prefix}.out_proj.weight"), device),
        wm.take::<B, 1>(&format!("{prefix}.out_proj.bias"), device),
    ) {
        set_linear_wb(out_proj, w, b);
    }
    Ok(())
}

fn load_swiglu_mlp<B: Backend>(
    wm: &mut WeightMap,
    mlp: &mut crate::model::dual_attention::Mlp<B>,
    prefix: &str,
    device: &B::Device,
) -> anyhow::Result<()> {
    if let Some(ref mut gate) = mlp.gate_proj {
        if let Ok(w) = wm.take::<B, 2>(&format!("{prefix}.gate_proj.w1.weight"), device) {
            set_linear_w(&mut gate.w1, w);
        }
        if let Ok(w) = wm.take::<B, 2>(&format!("{prefix}.gate_proj.w2.weight"), device) {
            set_linear_w(&mut gate.w2, w);
        }
    }
    if let Ok(w) = wm.take::<B, 2>(&format!("{prefix}.down_proj.weight"), device) {
        set_linear_w(&mut mlp.down_proj, w);
    }
    Ok(())
}

fn load_attn_pooler<B: Backend>(
    wm: &mut WeightMap,
    pooler: &mut crate::model::attn_pooler::AttnPooler<B>,
    prefix: &str,
    device: &B::Device,
) -> anyhow::Result<()> {
    if let Ok(t) = wm.take::<B, 3>(&format!("{prefix}.query_tokens"), device) {
        pooler.query_tokens = pooler.query_tokens.clone().map(|_| t);
    }
    load_fused_mha_to_split(
        wm,
        &mut pooler.q_proj, &mut pooler.k_proj,
        &mut pooler.v_proj, &mut pooler.out_proj,
        &format!("{prefix}.attn"), device,
    )?;
    Ok(())
}

fn load_text_encoder<B: Backend>(
    wm: &mut WeightMap,
    text: &mut crate::model::text_encoder::TextEncoder<B>,
    _cfg: &ModelConfig,
    device: &B::Device,
) -> anyhow::Result<()> {
    // Token embedding
    if let Ok(w) = wm.take::<B, 2>("text.token_embedding.weight", device) {
        text.token_embedding.weight = text.token_embedding.weight.clone().map(|_| w);
    }

    // Positional embedding
    if let Ok(t) = wm.take::<B, 2>("text.positional_embedding", device) {
        text.positional_embedding = text.positional_embedding.clone().map(|_| t);
    }

    // CLS embedding
    if let Some(ref mut cls) = text.cls_emb {
        if let Ok(t) = wm.take::<B, 1>("text.cls_emb", device) {
            *cls = cls.clone().map(|_| t);
        }
    }

    // Transformer blocks
    for (i, block) in text.blocks.iter_mut().enumerate() {
        let p = format!("text.transformer.resblocks.{i}");

        // LayerNorm 1
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>(&format!("{p}.ln_1.weight"), device),
            wm.take::<B, 1>(&format!("{p}.ln_1.bias"), device),
        ) { set_layernorm(&mut block.ln_1, w, b); }

        // Self-attention (fused MHA)
        load_fused_mha_single(wm, &mut block.attn.in_proj, &mut block.attn.out_proj,
            &format!("{p}.attn"), device)?;

        // LayerNorm 2
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>(&format!("{p}.ln_2.weight"), device),
            wm.take::<B, 1>(&format!("{p}.ln_2.bias"), device),
        ) { set_layernorm(&mut block.ln_2, w, b); }

        // MLP
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>(&format!("{p}.mlp.c_fc.weight"), device),
            wm.take::<B, 1>(&format!("{p}.mlp.c_fc.bias"), device),
        ) { set_linear_wb(&mut block.mlp_c_fc, w, b); }
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>(&format!("{p}.mlp.c_proj.weight"), device),
            wm.take::<B, 1>(&format!("{p}.mlp.c_proj.bias"), device),
        ) { set_linear_wb(&mut block.mlp_c_proj, w, b); }
    }

    // Final norm
    if let (Ok(w), Ok(b)) = (
        wm.take::<B, 1>("text.ln_final.weight", device),
        wm.take::<B, 1>("text.ln_final.bias", device),
    ) { set_layernorm(&mut text.ln_final, w, b); }

    // Text projection
    if let Ok(t) = wm.take::<B, 2>("text.text_projection", device) {
        text.text_projection = text.text_projection.clone().map(|_| t);
    }

    Ok(())
}

fn load_text_decoder<B: Backend>(
    wm: &mut WeightMap,
    dec: &mut crate::model::text_decoder::TextDecoder<B>,
    _cfg: &ModelConfig,
    device: &B::Device,
) -> anyhow::Result<()> {
    // Self-attention blocks
    for (i, block) in dec.self_attn_blocks.iter_mut().enumerate() {
        let p = format!("text_decoder.resblocks.{i}");

        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>(&format!("{p}.ln_1.weight"), device),
            wm.take::<B, 1>(&format!("{p}.ln_1.bias"), device),
        ) { set_layernorm(&mut block.ln_1, w, b); }

        load_fused_mha_single(wm, &mut block.attn.in_proj, &mut block.attn.out_proj,
            &format!("{p}.attn"), device)?;

        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>(&format!("{p}.ln_2.weight"), device),
            wm.take::<B, 1>(&format!("{p}.ln_2.bias"), device),
        ) { set_layernorm(&mut block.ln_2, w, b); }

        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>(&format!("{p}.mlp.c_fc.weight"), device),
            wm.take::<B, 1>(&format!("{p}.mlp.c_fc.bias"), device),
        ) { set_linear_wb(&mut block.mlp_c_fc, w, b); }
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>(&format!("{p}.mlp.c_proj.weight"), device),
            wm.take::<B, 1>(&format!("{p}.mlp.c_proj.bias"), device),
        ) { set_linear_wb(&mut block.mlp_c_proj, w, b); }
    }

    // Cross-attention blocks
    for (i, block) in dec.cross_attn_blocks.iter_mut().enumerate() {
        let p = format!("text_decoder.cross_attn.{i}");

        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>(&format!("{p}.ln_1.weight"), device),
            wm.take::<B, 1>(&format!("{p}.ln_1.bias"), device),
        ) { set_layernorm(&mut block.ln_1, w, b); }

        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>(&format!("{p}.ln_1_kv.weight"), device),
            wm.take::<B, 1>(&format!("{p}.ln_1_kv.bias"), device),
        ) { set_layernorm(&mut block.ln_1_kv, w, b); }

        load_fused_mha_single(wm, &mut block.attn.in_proj, &mut block.attn.out_proj,
            &format!("{p}.attn"), device)?;

        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>(&format!("{p}.ln_2.weight"), device),
            wm.take::<B, 1>(&format!("{p}.ln_2.bias"), device),
        ) { set_layernorm(&mut block.ln_2, w, b); }

        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>(&format!("{p}.mlp.c_fc.weight"), device),
            wm.take::<B, 1>(&format!("{p}.mlp.c_fc.bias"), device),
        ) { set_linear_wb(&mut block.mlp_c_fc, w, b); }
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>(&format!("{p}.mlp.c_proj.weight"), device),
            wm.take::<B, 1>(&format!("{p}.mlp.c_proj.bias"), device),
        ) { set_linear_wb(&mut block.mlp_c_proj, w, b); }
    }

    // Final norm
    if let (Ok(w), Ok(b)) = (
        wm.take::<B, 1>("text_decoder.ln_final.weight", device),
        wm.take::<B, 1>("text_decoder.ln_final.bias", device),
    ) { set_layernorm(&mut dec.ln_final, w, b); }

    // Text projection
    if let Ok(t) = wm.take::<B, 2>("text_decoder.text_projection", device) {
        dec.text_projection = dec.text_projection.clone().map(|_| t);
    }

    Ok(())
}
