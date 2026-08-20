// Translated from mlx-whisper/whisper.py (Apple Inc.)

use anyhow::Result;
use serde::{Deserialize, Serialize};

use mlx_rs::{
    Array, Dtype,
    builder::Builder,
    module::Module,
    nn::{Conv1d, Conv1dBuilder, Embedding, LayerNorm, Linear, LinearBuilder,
         MultiHeadAttention as MHAnn},
    ops::{add, arange, concatenate_axis, matmul, softmax_axis, zeros},
    ops::indexing::IndexOp,
    quantization::MaybeQuantized,
};

// ── Type aliases ─────────────────────────────────────────────────────────────

/// Cached (key, value) for one attention layer
pub type KVCache = (Array, Array);
/// Cache for one residual block: (self-attn cache, cross-attn cache)
pub type BlockCache = (Option<KVCache>, Option<KVCache>);

// ── ModelDimensions ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDimensions {
    pub n_mels: usize,
    pub n_audio_ctx: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_audio_layer: usize,
    pub n_vocab: usize,
    pub n_text_ctx: usize,
    pub n_text_state: usize,
    pub n_text_head: usize,
    pub n_text_layer: usize,
}

// ── sinusoids ────────────────────────────────────────────────────────────────

/// Returns sinusoids for positional embedding, shape [length, channels].
/// 對應 Python: sinusoids(length, channels)
pub fn sinusoids(length: usize, channels: usize) -> Result<Array> {
    assert!(channels % 2 == 0, "channels must be even");
    let half = channels / 2;
    let log_ts = 10000.0f32.ln() / (half as f32 - 1.0);
    let half_range = arange::<_, f32>(0.0f32, half as f32, 1.0)?;
    // inv_timescales = exp(-log_ts_inc * arange(half))
    let inv_ts = (&half_range * (-log_ts)).exp()?;

    let t = arange::<_, f32>(0.0f32, length as f32, 1.0)?;
    let t_col = t.reshape(&[length as i32, 1])?;
    let inv_row = inv_ts.reshape(&[1, half as i32])?;
    // scaled_time = t[:, None] * inv_ts[None, :]  → [length, half]
    let scaled = &t_col * &inv_row;

    Ok(concatenate_axis(&[scaled.sin()?, scaled.cos()?], 1)?)
}

/// `gelu` that preserves its input's dtype.
///
/// `mlx_rs::nn::gelu` computes `x * (1 + erf(x / sqrt(2))) / 2`, but builds the
/// constants with `array!(2f32.sqrt())` and `array!(2.0)` — genuine
/// `Dtype::Float32` scalar arrays, not Python's weak scalars. Ordinary type
/// promotion therefore makes the result f32 no matter what went in, exactly
/// like the `q * scale` case in `qkv_attention` above.
///
/// That matters far more here than it looks. gelu sits immediately after both
/// encoder convolutions and inside every residual block's MLP, so an f32 result
/// re-promotes the activations at the very first layer and then again in every
/// block — the residual add, the following layer norm and the next block's
/// matmuls all inherit f32. Left alone it would quietly turn the whole dtype
/// knob (PARITY.md MODEL-3) into a no-op while still reporting fp16 weights.
///
/// Upstream computes gelu in the model dtype: `mlx.nn.gelu` divides by a plain
/// Python float, which stays weak and keeps fp16. Casting the result back is
/// the conservative version of that — the elementwise gelu itself is evaluated
/// in f32 and rounded down afterwards, which is slightly *more* accurate than
/// upstream rather than less, and costs only an elementwise pass while keeping
/// every matmul and convolution downstream in fp16.
fn gelu_keep_dtype(x: &Array) -> Result<Array> {
    let dtype = x.dtype();
    Ok(mlx_rs::nn::gelu(x)?.as_dtype(dtype)?)
}

// ── MultiHeadAttention ───────────────────────────────────────────────────────

pub struct MultiHeadAttention {
    pub n_head: usize,
    // `MaybeQuantized<Linear>` rather than `Linear`: upstream's `nn.quantize`
    // (load_models.py:36-41) replaces individual `Linear`/`Embedding`
    // submodules with their quantized counterpart, per-module, based on
    // whether that module's weights were actually saved packed (see
    // `load_models::load_linear`). `MaybeQuantized` implements `Module` for
    // both variants, so every `.forward(x)` call below is unchanged either
    // way; only `TextDecoder::forward`'s `as_linear` call (no `Module`
    // equivalent) needs to match on the variant explicitly.
    pub query: MaybeQuantized<Linear>,
    pub key: MaybeQuantized<Linear>,   // no bias
    pub value: MaybeQuantized<Linear>,
    pub out: MaybeQuantized<Linear>,
}

impl MultiHeadAttention {
    pub fn new(n_state: usize, n_head: usize) -> Result<Self> {
        let n = n_state as i32;
        Ok(Self {
            n_head,
            query: MaybeQuantized::new(Linear::new(n, n)?),
            key: MaybeQuantized::new(LinearBuilder::new(n, n).bias(false).build()?),
            value: MaybeQuantized::new(Linear::new(n, n)?),
            out: MaybeQuantized::new(Linear::new(n, n)?),
        })
    }

    /// Forward pass.
    /// Returns (output, (k, v), qk).
    pub fn forward(
        &mut self,
        x: &Array,
        xa: Option<&Array>,
        mask: Option<&Array>,
        kv_cache: Option<KVCache>,
    ) -> Result<(Array, KVCache, Option<Array>)> {
        let q = self.query.forward(x)?;

        let (k, v) = if let Some(xa) = xa {
            // cross-attention: k/v from encoder output
            if let Some(kv) = kv_cache {
                kv // reuse cached cross-attention k, v
            } else {
                (self.key.forward(xa)?, self.value.forward(xa)?)
            }
        } else {
            // self-attention
            let mut k = self.key.forward(x)?;
            let mut v = self.value.forward(x)?;
            if let Some((k_cache, v_cache)) = kv_cache {
                k = concatenate_axis(&[k_cache, k], 1)?;
                v = concatenate_axis(&[v_cache, v], 1)?;
            }
            (k, v)
        };

        let (wv, qk) = self.qkv_attention(&q, &k, &v, mask)?;
        let out = self.out.forward(&wv)?;
        Ok((out, (k, v), qk))
    }

    fn qkv_attention(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        mask: Option<&Array>,
    ) -> Result<(Array, Option<Array>)> {
        let qs = q.shape();
        let (n_batch, n_ctx, n_state) = (qs[0], qs[1], qs[2]);
        let nh = self.n_head as i32;
        let head_dim = n_state / nh;
        let scale = (head_dim as f32).powf(-0.25f32);

        // Python's `q * scale` (`scale` a plain float) relies on `mx.array`'s
        // weak-scalar typing: multiplying a float16 array by a raw Python
        // number keeps float16. mlx-rs has no such weak typing — `Array *
        // f32` (`ScalarOrArray for f32`) builds a genuine `Dtype::Float32`
        // scalar via `Array::from_f32`, and multiplying that against an fp16
        // `q`/`k` promotes the result to f32 by ordinary type promotion,
        // silently undoing the dtype knob (PARITY.md MODEL-3) on every
        // attention call. Cast the scalar to the operand's own dtype first so
        // the multiply is same-dtype and fp16 survives.

        // q: [B, n_ctx, n_state] → [B, n_head, n_ctx, head_dim]
        let q = q.reshape(&[n_batch, n_ctx, nh, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let q = &q * &Array::from_f32(scale).as_dtype(q.dtype())?;

        // k: [B, n_kv, n_state] → [B, n_head, head_dim, n_kv]
        let kl = k.shape()[1];
        let k = k.reshape(&[n_batch, kl, nh, -1])?
            .transpose_axes(&[0, 2, 3, 1])?;
        let k = &k * &Array::from_f32(scale).as_dtype(k.dtype())?;

        // v: [B, n_kv, n_state] → [B, n_head, n_kv, head_dim]
        let v = v.reshape(&[n_batch, kl, nh, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // qk: [B, n_head, n_ctx, n_kv]
        let mut qk = matmul(&q, &k)?;
        if let Some(mask) = mask {
            let m = mask.index((..n_ctx, ..kl));
            qk = &qk + &m;
        }

        let w = softmax_axis(&qk, -1, true)?;
        // out: [B, n_ctx, n_state]
        let out = matmul(&w, &v)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[n_batch, n_ctx, n_state])?;

        Ok((out, Some(qk)))
    }
}

// ── ResidualAttentionBlock ────────────────────────────────────────────────────

pub struct ResidualAttentionBlock {
    pub attn: MultiHeadAttention,
    pub attn_ln: LayerNorm,
    pub cross_attn: Option<MultiHeadAttention>,
    pub cross_attn_ln: Option<LayerNorm>,
    pub mlp1: MaybeQuantized<Linear>,
    pub mlp2: MaybeQuantized<Linear>,
    pub mlp_ln: LayerNorm,
}

impl ResidualAttentionBlock {
    pub fn new(n_state: usize, n_head: usize, cross_attention: bool) -> Result<Self> {
        let n = n_state as i32;
        let n_mlp = n * 4;
        let (cross_attn, cross_attn_ln) = if cross_attention {
            (
                Some(MultiHeadAttention::new(n_state, n_head)?),
                Some(LayerNorm::new(n)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            attn: MultiHeadAttention::new(n_state, n_head)?,
            attn_ln: LayerNorm::new(n)?,
            cross_attn,
            cross_attn_ln,
            mlp1: MaybeQuantized::new(Linear::new(n, n_mlp)?),
            mlp2: MaybeQuantized::new(Linear::new(n_mlp, n)?),
            mlp_ln: LayerNorm::new(n)?,
        })
    }

    /// Returns (x, (kv, cross_kv), cross_qk).
    pub fn forward(
        &mut self,
        x: &Array,
        xa: Option<&Array>,
        mask: Option<&Array>,
        kv_cache: Option<BlockCache>,
    ) -> Result<(Array, BlockCache, Option<Array>)> {
        let (kv, cross_kv) = kv_cache.unwrap_or((None, None));

        // self-attention
        let ln_x = self.attn_ln.forward(x)?;
        let (y, new_kv, _) = self.attn.forward(&ln_x, None, mask, kv)?;
        let x = add(x, &y)?;

        // cross-attention (TextDecoder only)
        let (x, new_cross_kv, cross_qk) = if self.cross_attn.is_some() {
            let ln_x = self.cross_attn_ln.as_mut().unwrap().forward(&x)?;
            let (y, c_kv, qk) = self.cross_attn.as_mut().unwrap()
                .forward(&ln_x, xa, None, cross_kv)?;
            (&x + &y, Some(c_kv), qk)
        } else {
            (x, None, None)
        };

        // MLP
        let ln_x = self.mlp_ln.forward(&x)?;
        let hidden = gelu_keep_dtype(&self.mlp1.forward(&ln_x)?)?;
        let mlp_out = self.mlp2.forward(&hidden)?;
        let x = &x + &mlp_out;

        Ok((x, (Some(new_kv), new_cross_kv), cross_qk))
    }
}

// ── AudioEncoder ─────────────────────────────────────────────────────────────

pub struct AudioEncoder {
    pub conv1: Conv1d,
    pub conv2: Conv1d,
    pub positional_embedding: Array,
    pub blocks: Vec<ResidualAttentionBlock>,
    pub ln_post: LayerNorm,
}

impl AudioEncoder {
    pub fn new(
        n_mels: usize,
        n_ctx: usize,
        n_state: usize,
        n_head: usize,
        n_layer: usize,
        dtype: Dtype,
    ) -> Result<Self> {
        let nm = n_mels as i32;
        let ns = n_state as i32;
        Ok(Self {
            conv1: Conv1dBuilder::new(nm, ns, 3).padding(1).build()?,
            conv2: Conv1dBuilder::new(ns, ns, 3).stride(2).padding(1).build()?,
            // Python: `sinusoids(n_ctx, n_state).astype(dtype)`. `sinusoids()`
            // itself stays f32-only (trig ops), so the cast happens here, right
            // where upstream applies it — before this buffer is ever added to
            // fp16 conv activations, which would otherwise promote the whole
            // encoder graph back to f32 (PARITY.md MODEL-3).
            positional_embedding: sinusoids(n_ctx, n_state)?.as_dtype(dtype)?,
            blocks: (0..n_layer)
                .map(|_| ResidualAttentionBlock::new(n_state, n_head, false))
                .collect::<Result<Vec<_>>>()?,
            ln_post: LayerNorm::new(ns)?,
        })
    }

    /// Input x: [batch, n_frames, n_mels] (channels-last, as expected by mlx Conv1d)
    ///
    /// `audio_ctx` is an optional caller override for the encoder context
    /// length, mirroring whisper.cpp's `-ac` / `audio_ctx` flag. Whisper
    /// always encodes a full padded 30-second window, so on short audio
    /// most of the encoder's work is spent on silence padding; truncating
    /// the context cuts that wasted work at some accuracy cost.
    ///
    /// `None` reproduces today's behaviour exactly, byte-for-byte: the full
    /// `positional_embedding` table is added, with no extra checks on this
    /// path.
    ///
    /// `Some(n)` truncates the *mel input* to `2*n` frames before conv1,
    /// rather than running the full conv stack and truncating the
    /// post-conv activations: conv1 (stride 1) preserves the frame count
    /// and conv2 (stride 2) halves it, so slicing upstream of both convs
    /// skips the conv work on the discarded frames instead of computing it
    /// and throwing it away afterwards. `n` must be in `1..=n_audio_ctx`
    /// (the model's configured `dims.n_audio_ctx`, i.e.
    /// `positional_embedding`'s row count); anything else is a clear `Err`
    /// rather than a panic or an opaque mlx broadcast error.
    pub fn forward(&mut self, x: &Array, audio_ctx: Option<usize>) -> Result<Array> {
        let n_audio_ctx = self.positional_embedding.shape()[0] as usize;

        let (x, pos_emb): (Array, Array) = match audio_ctx {
            None => (x.clone(), self.positional_embedding.clone()),
            Some(n) => {
                if n == 0 || n > n_audio_ctx {
                    anyhow::bail!(
                        "audio_ctx override ({n}) out of range: must be in 1..={n_audio_ctx} \
                         (this model's dims.n_audio_ctx)"
                    );
                }
                let mel_frames_needed = (2 * n) as i32;
                let total_mel_frames = x.shape()[1];
                if mel_frames_needed > total_mel_frames {
                    anyhow::bail!(
                        "audio_ctx override ({n}) needs {mel_frames_needed} mel frames, \
                         but the input only has {total_mel_frames}"
                    );
                }
                let mel = x.index((.., ..mel_frames_needed, ..));
                let pos = self.positional_embedding.index((..n as i32, ..));
                (mel, pos)
            }
        };

        let x = gelu_keep_dtype(&self.conv1.forward(&x)?)?;
        let mut x = gelu_keep_dtype(&self.conv2.forward(&x)?)?;
        x = &x + &pos_emb;
        for block in &mut self.blocks {
            let (new_x, _, _) = block.forward(&x, None, None, None)?;
            x = new_x;
        }
        Ok(self.ln_post.forward(&x)?)
    }
}

// ── TextDecoder ───────────────────────────────────────────────────────────────

pub struct TextDecoder {
    // See the comment on `MultiHeadAttention`'s fields — quantized per-module
    // based on the checkpoint, mirroring `nn.quantize`'s `class_predicate`.
    // Also used tied as the output projection (`as_linear`, below), which
    // `MaybeQuantized` does not forward automatically.
    pub token_embedding: MaybeQuantized<Embedding>,
    pub positional_embedding: Array,
    pub blocks: Vec<ResidualAttentionBlock>,
    pub ln: LayerNorm,
    mask: Array,
}

impl TextDecoder {
    pub fn new(
        n_vocab: usize,
        n_ctx: usize,
        n_state: usize,
        n_head: usize,
        n_layer: usize,
        dtype: Dtype,
    ) -> Result<Self> {
        let nv = n_vocab as i32;
        let nc = n_ctx as i32;
        let ns = n_state as i32;
        Ok(Self {
            token_embedding: MaybeQuantized::new(Embedding::new(nv, ns)?),
            // Left as f32 zeros, uncast: this is a real learned parameter, not a
            // computed buffer, so `load_models::apply_weights` overwrites it
            // wholesale with the (already fp16-on-disk) loaded tensor. Upstream
            // agrees — `TextDecoder.__init__` never calls `.astype(dtype)` on
            // `self.positional_embedding`, only on `self._mask` below.
            positional_embedding: zeros::<f32>(&[nc, ns])?,
            blocks: (0..n_layer)
                .map(|_| ResidualAttentionBlock::new(n_state, n_head, true))
                .collect::<Result<Vec<_>>>()?,
            ln: LayerNorm::new(ns)?,
            // Python: `create_additive_causal_mask(n_ctx).astype(dtype)`. This
            // buffer is never part of the loaded weights (Python stores it as
            // `self._mask`, excluded from the parameter tree), so unlike the
            // positional embedding above it must be cast explicitly or it stays
            // f32 forever and promotes every attention score it's added to.
            mask: MHAnn::create_additive_causal_mask::<f32>(nc)?.as_dtype(dtype)?,
        })
    }

    /// x: [batch, seq_len] — integer token IDs
    /// xa: [batch, n_audio_ctx, n_audio_state] — encoder output
    /// Returns (logits, updated_kv_caches, cross_qks)
    pub fn forward(
        &mut self,
        x: &Array,
        xa: &Array,
        kv_cache: Option<Vec<Option<BlockCache>>>,
    ) -> Result<(Array, Vec<Option<BlockCache>>, Vec<Option<Array>>)> {
        let n = self.blocks.len();

        // offset = length of already-decoded tokens (from kv_cache)
        let offset = kv_cache.as_ref()
            .and_then(|c| c.first())
            .and_then(|bc| bc.as_ref())
            .and_then(|(kv, _)| kv.as_ref())
            .map(|(k, _)| k.shape()[1] as usize)
            .unwrap_or(0);

        let seq_len = x.shape()[x.ndim() - 1] as usize;
        let tok_emb = self.token_embedding.forward(x)?;
        let pos_emb = self.positional_embedding
            .index((offset as i32..(offset + seq_len) as i32,));
        let mut x = &tok_emb + &pos_emb;

        let mut caches: Vec<Option<BlockCache>> =
            kv_cache.unwrap_or_else(|| vec![None; n]);
        let mut cross_qks: Vec<Option<Array>> = vec![None; n];

        // Slice the causal mask according to the current offset and seq_len
        let mask = if seq_len == 1 {
            None
        } else {
            let m = self.mask.index((
                offset as i32..(offset + seq_len) as i32,
                ..(offset + seq_len) as i32,
            ));
            Some(m)
        };

        for (i, block) in self.blocks.iter_mut().enumerate() {
            let cache = caches[i].take();
            let (new_x, new_cache, cross_qk) =
                block.forward(&x, Some(xa), mask.as_ref(), cache)?;
            x = new_x;
            caches[i] = Some(new_cache);
            cross_qks[i] = cross_qk;
        }

        let x = self.ln.forward(&x)?;
        // `MaybeQuantized` forwards `Module::forward` for both variants (used
        // above and throughout this file), but `as_linear` isn't part of
        // `Module` — `Embedding` and `QuantizedEmbedding` each expose it
        // directly, so the tied-weights output projection has to match.
        let logits = match &self.token_embedding {
            MaybeQuantized::Original(emb) => emb.as_linear(&x),
            MaybeQuantized::Quantized(qemb) => qemb.as_linear(&x),
        }?;
        Ok((logits, caches, cross_qks))
    }
}

// ── Whisper ───────────────────────────────────────────────────────────────────

pub struct Whisper {
    pub dims: ModelDimensions,
    pub encoder: AudioEncoder,
    pub decoder: TextDecoder,
    /// Shape [n_pairs, 2] — (layer, head) pairs for alignment
    pub alignment_heads: Array,
    /// The dtype the encoder's positional embedding and the decoder's causal
    /// mask were built in (see `AudioEncoder::new` / `TextDecoder::new`).
    /// Upstream reads `self.model.dtype` off the model the same way to decide
    /// what to cast each mel segment to before encoding it — see
    /// `transcribe::detect_language` and `decoding::decode`.
    pub dtype: Dtype,
}

impl Whisper {
    /// `dtype` mirrors upstream's `Whisper.__init__(self, dims, dtype=mx.float16)`:
    /// it decides the dtype of the encoder's sinusoidal positional embedding
    /// and the decoder's additive causal mask (the two buffers that are
    /// computed rather than loaded from weights — see PARITY.md MODEL-3).
    /// Model weights themselves are untouched here; they take on whatever
    /// dtype the checkpoint was saved in once `load_models::load_model` applies
    /// them.
    pub fn new(dims: ModelDimensions, dtype: Dtype) -> Result<Self> {
        // alignment_heads = nonzero positions where last-half layers are True
        let half = dims.n_text_layer / 2;
        let pairs: Vec<i32> = (half..dims.n_text_layer)
            .flat_map(|l| (0..dims.n_text_head).flat_map(move |h| [l as i32, h as i32]))
            .collect();
        let n_pairs = pairs.len() / 2;
        let alignment_heads = Array::from_slice(&pairs, &[n_pairs as i32, 2]);

        Ok(Self {
            encoder: AudioEncoder::new(
                dims.n_mels,
                dims.n_audio_ctx,
                dims.n_audio_state,
                dims.n_audio_head,
                dims.n_audio_layer,
                dtype,
            )?,
            decoder: TextDecoder::new(
                dims.n_vocab,
                dims.n_text_ctx,
                dims.n_text_state,
                dims.n_text_head,
                dims.n_text_layer,
                dtype,
            )?,
            alignment_heads,
            dtype,
            dims,
        })
    }

    pub fn embed_audio(&mut self, mel: &Array) -> Result<Array> {
        self.encoder.forward(mel, None)
    }

    pub fn logits(&mut self, tokens: &Array, audio_features: &Array) -> Result<Array> {
        let (logits, _, _) = self.decoder.forward(tokens, audio_features, None)?;
        Ok(logits)
    }

    pub fn call(&mut self, mel: &Array, tokens: &Array) -> Result<Array> {
        let audio = self.encoder.forward(mel, None)?;
        let (logits, _, _) = self.decoder.forward(tokens, &audio, None)?;
        Ok(logits)
    }

    pub fn is_multilingual(&self) -> bool {
        self.dims.n_vocab >= 51865
    }

    pub fn num_languages(&self) -> usize {
        self.dims.n_vocab - 51765 - if self.is_multilingual() { 1 } else { 0 }
    }
}
