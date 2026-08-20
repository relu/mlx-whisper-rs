// Translated from mlx-whisper/load_models.py (Apple Inc.)

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use mlx_rs::{
    module::Param,
    nn::{Embedding, Linear, QuantizedEmbedding, QuantizedLinear},
    quantization::MaybeQuantized,
    Array, Dtype,
};

use crate::whisper::{ModelDimensions, MultiHeadAttention, ResidualAttentionBlock, Whisper};

// ── Quantization ─────────────────────────────────────────────────────────────

/// `(group_size, bits)` parsed from config.json's `quantization` object, or
/// `None` when the model isn't quantized at all. Threaded through every
/// loader below so each `Linear`/`Embedding` module can apply upstream's
/// per-module `class_predicate` (`load_models.py:36-39`):
///
/// ```text
/// class_predicate = (
///     lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
///     and f"{p}.scales" in weights
/// )
/// ```
///
/// i.e. a module is quantized only when the checkpoint *both* declares a
/// `quantization` config *and* actually saved that module's `.scales`
/// tensor — some mlx-community repos leave individual layers dense even
/// inside an otherwise-quantized model. `Conv1d` is never a candidate:
/// upstream's `isinstance` check excludes it, and this crate never threads
/// `quant` into `load_conv1d`.
type QuantConfig = Option<(i32, i32)>;

// ── HuggingFace path resolution ───────────────────────────────────────────────

/// 解析 path_or_hf_repo：若為本地目錄直接回傳；否則透過 hf-hub 下載 snapshot。
fn resolve_model_path(path_or_hf_repo: &str) -> Result<PathBuf> {
    let local = PathBuf::from(path_or_hf_repo);
    if local.is_dir() {
        return Ok(local);
    }
    // HuggingFace repo ID: download config + weights
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| anyhow!("Failed to create HF API: {e}"))?;
    let repo = api.model(path_or_hf_repo.to_string());

    // Download config.json
    repo.get("config.json")
        .map_err(|e| anyhow!("Failed to download config.json: {e}"))?;

    // Download weights (prefer safetensors)
    let weights_path = repo.get("weights.safetensors")
        .or_else(|_| repo.get("weights.npz"))
        .map_err(|e| anyhow!("Failed to download weights: {e}"))?;

    // Parent directory of the weights file is the model snapshot dir
    Ok(weights_path.parent().unwrap().to_path_buf())
}

// ── NPZ loading (ZIP of .npy files) ──────────────────────────────────────────

/// Parse a numpy dtype string to mlx Dtype.
fn parse_npy_dtype(descr: &str) -> Result<Dtype> {
    // descr examples: '<f2', '<f4', '<f8', '<i4', '<i8', '<u1', '|u1'
    let s = descr.trim_matches(['\'', '"', ' ']);
    match s {
        "<f2" | "=f2" => Ok(Dtype::Float16),
        "<f4" | "=f4" => Ok(Dtype::Float32),
        "<f8" | "=f8" => Ok(Dtype::Float64),
        "<i2" | "=i2" => Ok(Dtype::Int16),
        "<i4" | "=i4" => Ok(Dtype::Int32),
        "<i8" | "=i8" => Ok(Dtype::Int64),
        "<u1" | "=u1" | "|u1" => Ok(Dtype::Uint8),
        "<u2" | "=u2" => Ok(Dtype::Uint16),
        "<u4" | "=u4" => Ok(Dtype::Uint32),
        "|b1" => Ok(Dtype::Bool),
        other => bail!("Unsupported numpy dtype: {other}"),
    }
}

/// Parse numpy shape string like `(1024,)` or `(1024, 80)` → Vec<i32>.
fn parse_npy_shape(shape_str: &str) -> Result<Vec<i32>> {
    let s = shape_str.trim().trim_matches(['(', ')']);
    if s.is_empty() {
        return Ok(vec![]); // scalar
    }
    s.split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| {
            p.trim()
                .parse::<i32>()
                .map_err(|_| anyhow!("Bad shape token: {p}"))
        })
        .collect()
}

/// Parse a raw .npy byte slice → Array.
fn parse_npy_bytes(bytes: &[u8]) -> Result<Array> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        bail!("Not a valid .npy file");
    }
    let major = bytes[6];
    let (header_len, data_start) = if major >= 2 {
        let hl = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (hl, 12 + hl)
    } else {
        let hl = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (hl, 10 + hl)
    };

    let header = std::str::from_utf8(&bytes[10.min(data_start - header_len)..data_start])
        .map_err(|_| anyhow!("npy header not UTF-8"))?;

    // Parse 'descr' from header string
    let descr = {
        let start = header.find("'descr'")
            .or_else(|| header.find("\"descr\""))
            .ok_or_else(|| anyhow!("No 'descr' in npy header"))?;
        let rest = &header[start + 7..];
        // rest looks like: ": '<f2', 'fortran..."
        let colon = rest.find(':').ok_or_else(|| anyhow!("Bad npy header"))?;
        let val = rest[colon + 1..].trim();
        let end = val[1..].find(val.chars().next().unwrap())
            .ok_or_else(|| anyhow!("Bad descr value"))?;
        val[1..end + 1].to_string()
    };

    // Parse 'shape' from header string
    let shape_str = {
        let start = header.find("'shape'")
            .or_else(|| header.find("\"shape\""))
            .ok_or_else(|| anyhow!("No 'shape' in npy header"))?;
        let rest = &header[start + 7..];
        let colon = rest.find(':').ok_or_else(|| anyhow!("Bad npy header"))?;
        let val = rest[colon + 1..].trim();
        let paren_end = val.find(')').ok_or_else(|| anyhow!("Bad shape"))?;
        val[..=paren_end].to_string()
    };

    let dtype = parse_npy_dtype(&descr)?;
    let shape = parse_npy_shape(&shape_str)?;
    let data = &bytes[data_start..];

    // Validate size
    let n_elem: usize = if shape.is_empty() { 1 } else { shape.iter().map(|&d| d as usize).product() };
    let elem_size = dtype_size(dtype)?;
    if data.len() < n_elem * elem_size {
        bail!("npy data shorter than expected: {} < {}", data.len(), n_elem * elem_size);
    }

    Ok(unsafe { Array::from_raw_data(data.as_ptr() as *const std::ffi::c_void, &shape, dtype) })
}

fn dtype_size(dtype: Dtype) -> Result<usize> {
    Ok(match dtype {
        Dtype::Bool | Dtype::Uint8 | Dtype::Int8 => 1,
        Dtype::Uint16 | Dtype::Int16 | Dtype::Float16 | Dtype::Bfloat16 => 2,
        Dtype::Uint32 | Dtype::Int32 | Dtype::Float32 => 4,
        Dtype::Uint64 | Dtype::Int64 | Dtype::Float64 | Dtype::Complex64 => 8,
    })
}

/// Load a .npz file (ZIP of .npy files) → HashMap<key, Array>.
fn load_npz(path: &Path) -> Result<HashMap<String, Array>> {
    let file = std::fs::File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut map = HashMap::new();

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry.name().to_string();
        // Key is filename without ".npy" extension
        let key = if name.ends_with(".npy") {
            name[..name.len() - 4].to_string()
        } else {
            name
        };
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes)?;
        let arr = parse_npy_bytes(&bytes)?;
        map.insert(key, arr);
    }
    Ok(map)
}

/// Load weights from model directory (safetensors or npz).
fn load_weights(model_path: &Path) -> Result<HashMap<String, Array>> {
    let st_path = model_path.join("weights.safetensors");
    if st_path.exists() {
        return Ok(Array::load_safetensors(&st_path)
            .map_err(|e| anyhow!("Failed to load safetensors: {e}"))?);
    }
    let npz_path = model_path.join("weights.npz");
    if npz_path.exists() {
        return load_npz(&npz_path);
    }
    bail!("No weights.safetensors or weights.npz found in {:?}", model_path)
}

// ── Weight application helpers ────────────────────────────────────────────────

fn get_w(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| anyhow!("Missing weight: {key}"))
}

/// Loads one `Linear` (or, per the `quant` class predicate, `QuantizedLinear`)
/// module from `weights`.
///
/// mlx-rs's own `QuantizedLinear`/`QuantizedEmbedding` constructors all
/// *re-quantize a live float `Array`* via `ops::quantize` — there is no
/// constructor that accepts already-packed weights loaded from disk. So
/// unlike the dense case (which mutated the existing `Linear` in place),
/// this builds the whole `QuantizedLinear` struct literal directly from the
/// `.weight` (packed uint32), `.scales` and `.biases` keys and replaces the
/// `MaybeQuantized` variant wholesale. All of `QuantizedLinear`'s fields are
/// `pub`, so this needs no builder.
fn load_linear(
    linear: &mut MaybeQuantized<Linear>,
    weights: &HashMap<String, Array>,
    prefix: &str,
    quant: QuantConfig,
) -> Result<()> {
    let bias = weights.get(&format!("{prefix}.bias")).cloned();

    if let Some((group_size, bits)) = quant {
        if weights.contains_key(&format!("{prefix}.scales")) {
            *linear = MaybeQuantized::Quantized(QuantizedLinear {
                group_size,
                bits,
                scales: Param::new(get_w(weights, &format!("{prefix}.scales"))?),
                biases: Param::new(get_w(weights, &format!("{prefix}.biases"))?),
                inner: Linear {
                    weight: Param::new(get_w(weights, &format!("{prefix}.weight"))?),
                    bias: Param::new(bias),
                },
            });
            return Ok(());
        }
    }

    *linear = MaybeQuantized::Original(Linear {
        weight: Param::new(get_w(weights, &format!("{prefix}.weight"))?),
        bias: Param::new(bias),
    });
    Ok(())
}

fn load_layer_norm(
    ln: &mut mlx_rs::nn::LayerNorm,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<()> {
    ln.weight = Param::new(Some(get_w(weights, &format!("{prefix}.weight"))?));
    ln.bias = Param::new(Some(get_w(weights, &format!("{prefix}.bias"))?));
    Ok(())
}

fn load_conv1d(
    conv: &mut mlx_rs::nn::Conv1d,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<()> {
    conv.weight = Param::new(get_w(weights, &format!("{prefix}.weight"))?);
    if let Some(b) = weights.get(&format!("{prefix}.bias")) {
        conv.bias = Param::new(Some(b.clone()));
    } else {
        conv.bias = Param::new(None);
    }
    Ok(())
}

fn load_mha(
    mha: &mut MultiHeadAttention,
    weights: &HashMap<String, Array>,
    prefix: &str,
    quant: QuantConfig,
) -> Result<()> {
    load_linear(&mut mha.query, weights, &format!("{prefix}.query"), quant)?;
    load_linear(&mut mha.key, weights, &format!("{prefix}.key"), quant)?;
    load_linear(&mut mha.value, weights, &format!("{prefix}.value"), quant)?;
    load_linear(&mut mha.out, weights, &format!("{prefix}.out"), quant)?;
    Ok(())
}

fn load_block(
    block: &mut ResidualAttentionBlock,
    weights: &HashMap<String, Array>,
    prefix: &str,
    quant: QuantConfig,
) -> Result<()> {
    load_mha(&mut block.attn, weights, &format!("{prefix}.attn"), quant)?;
    load_layer_norm(&mut block.attn_ln, weights, &format!("{prefix}.attn_ln"))?;

    if let Some(cross_attn) = &mut block.cross_attn {
        load_mha(cross_attn, weights, &format!("{prefix}.cross_attn"), quant)?;
        load_layer_norm(
            block.cross_attn_ln.as_mut().unwrap(),
            weights,
            &format!("{prefix}.cross_attn_ln"),
        )?;
    }

    load_linear(&mut block.mlp1, weights, &format!("{prefix}.mlp1"), quant)?;
    load_linear(&mut block.mlp2, weights, &format!("{prefix}.mlp2"), quant)?;
    load_layer_norm(&mut block.mlp_ln, weights, &format!("{prefix}.mlp_ln"))?;
    Ok(())
}

/// Loads the decoder's tied token-embedding / output-projection module,
/// mirroring `load_linear`'s quantized-vs-dense branch (see its doc comment).
///
/// Unlike `QuantizedLinear`, mlx-rs's `QuantizedEmbedding` (`nn/quantized.rs`)
/// does not tag its `scales`/`biases`/`inner` fields with `#[param]`, so
/// they're excluded from `ModuleParameters::parameters()` — a generic
/// `update()`/eval-of-parameters walk over the model would silently skip
/// them. Harmless here: this crate always assigns loaded weights by direct
/// field access, as below, never through that generic path.
fn load_token_embedding(
    embedding: &mut MaybeQuantized<Embedding>,
    weights: &HashMap<String, Array>,
    prefix: &str,
    quant: QuantConfig,
) -> Result<()> {
    if let Some((group_size, bits)) = quant {
        if weights.contains_key(&format!("{prefix}.scales")) {
            *embedding = MaybeQuantized::Quantized(QuantizedEmbedding {
                group_size,
                bits,
                scales: Param::new(get_w(weights, &format!("{prefix}.scales"))?),
                biases: Param::new(get_w(weights, &format!("{prefix}.biases"))?),
                inner: Embedding {
                    weight: Param::new(get_w(weights, &format!("{prefix}.weight"))?),
                },
            });
            return Ok(());
        }
    }

    // Optional, like the original code: if the checkpoint has no dense
    // weight under this prefix either, leave the random-init default from
    // `TextDecoder::new` in place.
    if let Some(w) = weights.get(&format!("{prefix}.weight")) {
        *embedding = MaybeQuantized::Original(Embedding {
            weight: Param::new(w.clone()),
        });
    }
    Ok(())
}

/// Apply weight dict to Whisper model.
fn apply_weights(model: &mut Whisper, weights: &HashMap<String, Array>, quant: QuantConfig) -> Result<()> {
    // Encoder. conv1/conv2 are never quantized — upstream's `isinstance(m,
    // (nn.Linear, nn.Embedding))` class predicate excludes `Conv1d` — so
    // `load_conv1d` doesn't take `quant` at all.
    load_conv1d(&mut model.encoder.conv1, weights, "encoder.conv1")?;
    load_conv1d(&mut model.encoder.conv2, weights, "encoder.conv2")?;
    load_layer_norm(&mut model.encoder.ln_post, weights, "encoder.ln_post")?;
    for (i, block) in model.encoder.blocks.iter_mut().enumerate() {
        load_block(block, weights, &format!("encoder.blocks.{i}"), quant)?;
    }

    // Decoder
    load_token_embedding(
        &mut model.decoder.token_embedding,
        weights,
        "decoder.token_embedding",
        quant,
    )?;
    if let Some(pe) = weights.get("decoder.positional_embedding") {
        model.decoder.positional_embedding = pe.clone();
    }
    load_layer_norm(&mut model.decoder.ln, weights, "decoder.ln")?;
    for (i, block) in model.decoder.blocks.iter_mut().enumerate() {
        load_block(block, weights, &format!("decoder.blocks.{i}"), quant)?;
    }

    // alignment_heads (optional, overrides the default)
    if let Some(ah) = weights.get("alignment_heads") {
        model.alignment_heads = ah.clone();
    }

    Ok(())
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Load a Whisper model from a local directory or HuggingFace repo ID.
///
/// `dtype` decides the dtype of the model's computed buffers (the encoder's
/// sinusoidal positional embedding and the decoder's causal mask — see
/// `Whisper::new`); it does *not* cast the loaded weight tensors, matching
/// upstream `load_models.py`, which forwards `dtype` straight into
/// `whisper.Whisper(model_args, dtype)` and applies `weights` to the model
/// as-is. Pass `mlx_rs::Dtype::Float16` for upstream's effective default
/// (`transcribe.py` picks it whenever `fp16` isn't disabled) or
/// `mlx_rs::Dtype::Float32` to opt out.
///
/// # Examples
/// ```no_run
/// use mlx_rs::Dtype;
/// let model = mlx_whisper_rs::load_models::load_model(
///     "mlx-community/whisper-medium-mlx",
///     Dtype::Float16,
/// )?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn load_model(path_or_hf_repo: &str, dtype: Dtype) -> Result<Whisper> {
    let model_path = resolve_model_path(path_or_hf_repo)?;

    // Load config.json → ModelDimensions
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow!("Failed to read config.json: {e}"))?;
    let mut config: serde_json::Value = serde_json::from_str(&config_str)?;
    // Remove fields not in ModelDimensions
    let mut quant: QuantConfig = None;
    if let Some(obj) = config.as_object_mut() {
        obj.remove("model_type");
        // Upstream: `quantization = config.pop("quantization", None)`, then
        // (after weights are loaded) `nn.quantize(model, **quantization,
        // class_predicate=...)`. The dict is normally `{"group_size": ...,
        // "bits": ...}`; fall back to the same defaults `nn.quantize` and
        // mlx-rs's own `QuantizedLinear`/`QuantizedEmbedding` use if either
        // key is missing. Which modules actually get quantized is decided
        // per-module in `load_linear`/`load_token_embedding`, based on
        // whether the checkpoint saved that module's `.scales` tensor — see
        // `QuantConfig`'s doc comment.
        if let Some(q) = obj.remove("quantization") {
            let group_size = q
                .get("group_size")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32)
                .unwrap_or(QuantizedLinear::DEFAULT_GROUP_SIZE);
            let bits = q
                .get("bits")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32)
                .unwrap_or(QuantizedLinear::DEFAULT_BITS);
            quant = Some((group_size, bits));
        }
    }
    let dims: ModelDimensions = serde_json::from_value(config)?;

    // Build model skeleton
    let mut model = Whisper::new(dims, dtype)?;

    // Load and apply weights
    let weights = load_weights(&model_path)?;
    apply_weights(&mut model, &weights, quant)?;

    // Force evaluation of key arrays
    mlx_rs::transforms::eval([&model.encoder.positional_embedding])?;

    Ok(model)
}
