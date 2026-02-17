"""
MLX implementation of KaniTTS-2 model.

Extends mlx-lm's LFM2 with:
- Learnable RoPE (per-layer alpha-scaled frequencies)
- Frame-level position encoding (4 audio tokens share same position per frame)
- Speaker embedding projection (128-dim -> hidden_size)
- Custom generation loop with frame-level position tracking

Requires: mlx, mlx-lm
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.lfm2 import (
    MLP,
    ModelArgs as Lfm2ModelArgs,
    ShortConv,
)
from mlx_lm.models.base import (
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache


@dataclass
class KaniTTSModelArgs(Lfm2ModelArgs):
    """Extended model args for KaniTTS-2."""
    audio_tokens_start: int = 64410
    tokens_per_frame: int = 4
    audio_step: float = 1.0
    use_learnable_rope: bool = False
    alpha_min: float = 0.1
    alpha_max: float = 2.0
    speaker_emb_dim: int = 128


def compute_frame_level_positions(
    input_ids: mx.array,
    audio_tokens_start: int,
    tokens_per_frame: int = 4,
    audio_step: float = 1.0,
) -> mx.array:
    """
    Compute frame-level position IDs for KaniTTS-2.

    Text tokens get sequential positions. Audio tokens are grouped into frames
    of `tokens_per_frame`, where all tokens in a frame share the same position.

    Args:
        input_ids: [batch_size, seq_len]
        audio_tokens_start: Token ID where audio tokens begin
        tokens_per_frame: Tokens per audio frame (typically 4)
        audio_step: Position step per frame

    Returns:
        position_ids: [batch_size, seq_len] as float array
    """
    batch_size, seq_len = input_ids.shape

    is_audio = input_ids >= audio_tokens_start
    text_mask = mx.logical_not(is_audio)

    # Cumulative text token count (prepend zero)
    zeros = mx.zeros((batch_size, 1), dtype=mx.int32)
    text_count = mx.concatenate([zeros, text_mask.astype(mx.int32)], axis=1)
    text_count = mx.cumsum(text_count, axis=1)[:, :-1]

    # Cumulative audio token count -> frame count
    audio_count = mx.concatenate([zeros, is_audio.astype(mx.int32)], axis=1)
    audio_count = mx.cumsum(audio_count, axis=1)[:, :-1]
    audio_frame_count = audio_count // tokens_per_frame

    # Final positions
    position_ids = text_count.astype(mx.float32) + audio_frame_count.astype(mx.float32) * audio_step
    return position_ids


class LearnableRotaryEmbedding(nn.Module):
    """
    Per-layer learnable RoPE with alpha-scaled frequencies.

    theta_i^(l) = alpha^(l) * base^(-2i/d)
    alpha^(l) = alpha_min + (alpha_max - alpha_min) * sigmoid(w^(l))
    """

    def __init__(self, dim: int, base: float, alpha_min: float = 0.1, alpha_max: float = 2.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self._inv_freq_base = inv_freq
        self.alpha_weight = mx.zeros((1,))

    @property
    def alpha(self):
        return self.alpha_min + (self.alpha_max - self.alpha_min) * mx.sigmoid(self.alpha_weight)

    @property
    def inv_freq(self):
        return self._inv_freq_base * self.alpha

    def __call__(self, queries: mx.array, keys: mx.array, positions: mx.array):
        """
        Apply learnable RoPE to queries and keys using explicit positions.

        Args:
            queries: [batch, n_heads, seq_len, head_dim]
            keys: [batch, n_kv_heads, seq_len, head_dim]
            positions: [batch, seq_len] float positions

        Returns:
            (rotated_queries, rotated_keys)
        """
        inv_freq = self.inv_freq  # [dim/2]

        # positions: [B, L] -> [B, 1, L, 1]
        pos = positions[:, None, :, None].astype(mx.float32)
        # inv_freq: [dim/2] -> [1, 1, 1, dim/2]
        freq = inv_freq[None, None, None, :]
        # angles: [B, 1, L, dim/2]
        angles = pos * freq

        cos_val = mx.cos(angles)
        sin_val = mx.sin(angles)

        # Expand to full head_dim: [B, 1, L, dim/2] -> [B, 1, L, dim]
        cos_val = mx.concatenate([cos_val, cos_val], axis=-1)
        sin_val = mx.concatenate([sin_val, sin_val], axis=-1)

        queries = self._apply_rotary(queries, cos_val, sin_val)
        keys = self._apply_rotary(keys, cos_val, sin_val)
        return queries, keys

    @staticmethod
    def _apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        """Apply rotary embeddings (non-interleaved / GPT-NeoX style)."""
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        rotated = mx.concatenate([-x2, x1], axis=-1)
        return x * cos + rotated * sin


class KaniAttention(nn.Module):
    """
    Attention module supporting both standard RoPE (via cache.offset)
    and learnable RoPE with explicit positions.
    """

    def __init__(self, args: KaniTTSModelArgs, layer_idx: int,
                 learnable_rope: Optional[LearnableRotaryEmbedding] = None):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.learnable_rope = learnable_rope

        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        # Standard RoPE fallback (used when learnable_rope is None)
        self.rope = nn.RoPE(self.head_dim, base=args.rope_theta, traditional=False)

    def __call__(self, x, mask=None, cache=None, positions=None):
        """
        Args:
            x: [B, L, D]
            mask: attention mask
            cache: KVCache
            positions: [B, L] explicit position IDs (float). If None, uses cache.offset.
        """
        B, L, D = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_layernorm(
            queries.reshape(B, L, self.n_heads, -1)
        ).transpose(0, 2, 1, 3)
        keys = self.k_layernorm(
            keys.reshape(B, L, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if self.learnable_rope is not None and positions is not None:
            # Use learnable RoPE with explicit positions
            queries, keys = self.learnable_rope(queries, keys, positions)
        else:
            # Standard RoPE with cache offset
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, mask=mask, scale=self.scale
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class KaniDecoderLayer(nn.Module):
    """Decoder layer that routes to KaniAttention or ShortConv."""

    def __init__(self, args: KaniTTSModelArgs, layer_idx: int,
                 learnable_rope: Optional[LearnableRotaryEmbedding] = None):
        super().__init__()
        self.is_attention_layer = layer_idx in args.full_attn_idxs

        if self.is_attention_layer:
            self.self_attn = KaniAttention(args, layer_idx, learnable_rope=learnable_rope)
        else:
            self.conv = ShortConv(args, layer_idx)

        self.feed_forward = MLP(
            dim=args.block_dim,
            ff_dim=args.block_ff_dim,
            multiple_of=args.block_multiple_of,
            auto_adjust_ff_dim=args.block_auto_adjust_ff_dim,
            ffn_dim_multiplier=args.block_ffn_dim_multiplier,
        )
        self.operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(self, x, mask=None, cache=None, positions=None):
        if self.is_attention_layer:
            r = self.self_attn(self.operator_norm(x), mask=mask, cache=cache, positions=positions)
        else:
            r = self.conv(self.operator_norm(x), mask=mask, cache=cache)
        h = x + r
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class KaniLfm2Model(nn.Module):
    """LFM2 model extended with KaniTTS-2 features."""

    def __init__(self, args: KaniTTSModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        # Speaker embedding projection
        self.speaker_emb_projection = nn.Linear(args.speaker_emb_dim, args.hidden_size, bias=False)

        # Build learnable RoPE modules for attention layers (if enabled)
        learnable_ropes = {}
        if args.use_learnable_rope:
            head_dim = args.hidden_size // args.num_attention_heads
            for idx in args.full_attn_idxs:
                learnable_ropes[idx] = LearnableRotaryEmbedding(
                    dim=head_dim,
                    base=args.rope_theta,
                    alpha_min=args.alpha_min,
                    alpha_max=args.alpha_max,
                )

        # Build decoder layers
        self.layers = []
        for i in range(args.num_hidden_layers):
            rope = learnable_ropes.get(i, None)
            self.layers.append(KaniDecoderLayer(args, layer_idx=i, learnable_rope=rope))

        self.embedding_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

        # Cache index helpers
        self.fa_idx = args.full_attn_idxs[0]
        self.conv_idx = 0
        for i in range(args.num_hidden_layers):
            if i in args.full_attn_idxs:
                self.conv_idx += 1
            else:
                break

    def __call__(self, inputs, cache=None, input_embeddings=None, positions=None):
        """
        Args:
            inputs: token IDs [B, L]
            cache: list of KVCache/ArraysCache per layer
            input_embeddings: pre-computed embeddings [B, L, D] (overrides inputs)
            positions: explicit position IDs [B, L] (float)
        """
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        attn_mask = create_attention_mask(h, cache[self.fa_idx])
        conv_mask = create_ssm_mask(h, cache[self.conv_idx])

        for layer, c in zip(self.layers, cache):
            mask = attn_mask if layer.is_attention_layer else conv_mask
            h = layer(h, mask, cache=c, positions=positions)

        return self.embedding_norm(h)


class KaniTTS2MLXModel(nn.Module):
    """
    Top-level KaniTTS-2 model for MLX.

    Wraps KaniLfm2Model with lm_head and provides weight sanitization.
    """

    def __init__(self, args: KaniTTSModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = KaniLfm2Model(args)

    def __call__(self, inputs, cache=None, input_embeddings=None, positions=None):
        out = self.model(inputs, cache, input_embeddings, positions=positions)
        return self.model.embed_tokens.as_linear(out)

    def sanitize(self, weights):
        """Sanitize weights loaded from safetensors."""
        sanitized = {}
        for name, param in weights.items():
            # Conv weight transpose: PyTorch -> MLX layout
            if "conv.weight" in name and param.ndim == 3:
                if param.shape[-1] > param.shape[1]:
                    param = param.transpose(0, 2, 1)

            # Drop lm_head (tied to embed_tokens)
            if name == "lm_head.weight":
                continue

            # Map learnable RoPE inv_freq_base buffer
            if "learnable_rope_layers" in name and "inv_freq_base" in name:
                # Extract layer index and remap to our structure
                # From: model.learnable_rope_layers.{list_idx}.inv_freq_base
                # To: model.layers.{attn_layer_idx}.*  (handled via load_weights)
                # Keep as-is; the layer mapping is handled by matching indices
                pass

            sanitized[name] = param
        return sanitized

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            KVCache() if l.is_attention_layer else ArraysCache(size=1)
            for l in self.layers
        ]


# --------------------------------------------------------------------------- #
# Generation utilities
# --------------------------------------------------------------------------- #

def sample_token(logits: mx.array, temperature: float = 1.0, top_p: float = 0.95) -> mx.array:
    """Sample a single token from logits with temperature and top-p."""
    if temperature == 0:
        return mx.argmax(logits, axis=-1)

    logits = logits / temperature

    # Top-p (nucleus) sampling
    if top_p < 1.0:
        sorted_indices = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
        probs = mx.softmax(sorted_logits, axis=-1)
        cumulative_probs = mx.cumsum(probs, axis=-1)

        # Zero out tokens beyond top_p threshold
        mask = cumulative_probs - probs > top_p
        sorted_logits = mx.where(mask, mx.array(float("-inf")), sorted_logits)

        # Unsort
        unsort_indices = mx.argsort(sorted_indices, axis=-1)
        logits = mx.take_along_axis(sorted_logits, unsort_indices, axis=-1)

    probs = mx.softmax(logits, axis=-1)
    return mx.random.categorical(mx.log(probs + 1e-10))


def apply_repetition_penalty(logits: mx.array, generated_ids: list, penalty: float = 1.1) -> mx.array:
    """Apply repetition penalty to logits based on already-generated tokens."""
    if penalty == 1.0 or not generated_ids:
        return logits

    unique_ids = list(set(generated_ids))
    indices = mx.array(unique_ids)
    penalties = mx.take(logits[0], indices, axis=0)

    # Reduce logits for tokens that appeared
    positive_mask = penalties > 0
    penalties = mx.where(positive_mask, penalties / penalty, penalties * penalty)

    # Scatter back
    logits_flat = logits[0]
    for i, idx in enumerate(unique_ids):
        logits_flat = mx.where(
            mx.arange(logits_flat.shape[0]) == idx,
            penalties[i],
            logits_flat,
        )
    return logits_flat[None, :]


def generate(
    model: KaniTTS2MLXModel,
    input_ids: mx.array,
    speaker_emb: Optional[mx.array] = None,
    max_new_tokens: int = 3000,
    temperature: float = 1.0,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    eos_token_id: int = 64402,
) -> mx.array:
    """
    Generate audio tokens with frame-level position encoding.

    This is a custom generation loop that supports KaniTTS-2's
    frame-level position encoding during autoregressive decoding.

    Args:
        model: KaniTTS2MLXModel instance
        input_ids: [1, seq_len] input token IDs
        speaker_emb: [1, speaker_emb_dim] speaker embedding (optional)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        repetition_penalty: Penalty for repeated tokens
        eos_token_id: End-of-speech token ID

    Returns:
        Complete token sequence [1, total_len] including input and generated tokens
    """
    args = model.args

    # Build input embeddings with optional speaker embedding injection
    input_embeds = model.model.embed_tokens(input_ids)

    if speaker_emb is not None:
        # Project speaker embedding: [1, 128] -> [1, hidden_size]
        spk_proj = model.model.speaker_emb_projection(speaker_emb)
        spk_proj = spk_proj[:, None, :]  # [1, 1, hidden_size]

        # Insert at position 1 (after first token)
        input_embeds = mx.concatenate([
            input_embeds[:, :1, :],
            spk_proj,
            input_embeds[:, 1:, :],
        ], axis=1)

    # Compute frame-level positions for the full input
    # For the prefill, all tokens are text (no audio yet), so positions are sequential
    prefill_len = input_embeds.shape[1]
    positions = mx.arange(prefill_len, dtype=mx.float32)[None, :]  # [1, prefill_len]

    # Create cache
    cache = model.make_cache()

    # Prefill: process entire input sequence
    logits = model(None, cache=cache, input_embeddings=input_embeds, positions=positions)
    mx.eval([c.state for c in cache])

    # Sample first token
    next_logits = logits[:, -1:, :]
    generated_ids = []
    all_ids = input_ids[0].tolist()

    token = sample_token(next_logits[:, 0, :], temperature, top_p)
    token_id = token.item()
    generated_ids.append(token_id)
    all_ids.append(token_id)

    # Generation state for frame-level positions
    audio_tokens_generated = 0
    current_frame_position = None

    # Autoregressive decode loop
    for step in range(max_new_tokens - 1):
        if token_id == eos_token_id:
            break

        # Compute position for this token
        if token_id >= args.audio_tokens_start:
            # Audio token
            if current_frame_position is None:
                # First audio token -- position is next after the current cache length
                # (cache already has prefill_len tokens, so this is at prefill_len)
                current_frame_position = float(cache[model.model.fa_idx].offset)

            token_in_frame = audio_tokens_generated % args.tokens_per_frame
            if token_in_frame == 0 and audio_tokens_generated > 0:
                current_frame_position += args.audio_step

            pos = current_frame_position
            audio_tokens_generated += 1
        else:
            # Text/special token -- sequential position from cache
            pos = float(cache[model.model.fa_idx].offset)

        pos_ids = mx.array([[pos]], dtype=mx.float32)
        token_input = mx.array([[token_id]])

        # Forward pass for single token
        logits = model(token_input, cache=cache, positions=pos_ids)
        mx.eval([c.state for c in cache])

        # Apply repetition penalty and sample
        next_logits = logits[:, -1:, :]
        if repetition_penalty != 1.0:
            next_logits = apply_repetition_penalty(next_logits, generated_ids, repetition_penalty)
        token = sample_token(next_logits[:, 0, :] if next_logits.ndim == 3 else next_logits, temperature, top_p)
        token_id = token.item()
        generated_ids.append(token_id)
        all_ids.append(token_id)

    return mx.array([all_ids])
