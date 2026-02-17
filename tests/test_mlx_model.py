"""Tests for kani_tts.mlx_model — MLX frame-level positions, learnable RoPE, sampling."""

import pytest
import numpy as np

mlx = pytest.importorskip("mlx.core", reason="mlx not available")
mx = mlx

from kani_tts.mlx_model import (
    compute_frame_level_positions,
    LearnableRotaryEmbedding,
    KaniTTSModelArgs,
    KaniTTS2MLXModel,
    sample_token,
    apply_repetition_penalty,
)


# --------------------------------------------------------------------------- #
# compute_frame_level_positions (MLX)
# --------------------------------------------------------------------------- #

class TestMLXFrameLevelPositions:
    """Tests for the MLX version of frame-level position computation."""

    AUDIO_START = 64410

    def test_text_only_sequential(self):
        ids = mx.array([[10, 20, 30, 40, 50]])
        pos = compute_frame_level_positions(ids, self.AUDIO_START)
        expected = [0.0, 1.0, 2.0, 3.0, 4.0]
        np.testing.assert_allclose(np.array(pos[0].tolist()), expected)

    def test_single_audio_frame(self):
        aud = self.AUDIO_START
        ids = mx.array([[100, 200, aud, aud + 4032, aud + 8064, aud + 12096]])
        pos = compute_frame_level_positions(ids, self.AUDIO_START, tokens_per_frame=4)
        expected = [0.0, 1.0, 2.0, 2.0, 2.0, 2.0]
        np.testing.assert_allclose(np.array(pos[0].tolist()), expected)

    def test_two_audio_frames(self):
        aud = self.AUDIO_START
        ids = mx.array([[100, aud, aud, aud, aud, aud, aud, aud, aud]])
        pos = compute_frame_level_positions(ids, self.AUDIO_START, tokens_per_frame=4, audio_step=1.0)
        expected = [0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
        np.testing.assert_allclose(np.array(pos[0].tolist()), expected)

    def test_audio_step_half(self):
        aud = self.AUDIO_START
        ids = mx.array([[100, 200, aud, aud, aud, aud, aud, aud, aud, aud]])
        pos = compute_frame_level_positions(ids, self.AUDIO_START, tokens_per_frame=4, audio_step=0.5)
        expected = [0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 2.5]
        np.testing.assert_allclose(np.array(pos[0].tolist()), expected)

    def test_text_after_audio(self):
        aud = self.AUDIO_START
        ids = mx.array([[100, aud, aud, aud, aud, 300]])
        pos = compute_frame_level_positions(ids, self.AUDIO_START)
        expected = [0.0, 1.0, 1.0, 1.0, 1.0, 2.0]
        np.testing.assert_allclose(np.array(pos[0].tolist()), expected)

    def test_batch_dimension(self):
        ids = mx.array([
            [10, 20, 30],
            [10, self.AUDIO_START, self.AUDIO_START],
        ])
        pos = compute_frame_level_positions(ids, self.AUDIO_START)
        assert pos.shape == (2, 3)
        np.testing.assert_allclose(np.array(pos[0].tolist()), [0.0, 1.0, 2.0])
        np.testing.assert_allclose(np.array(pos[1].tolist()), [0.0, 1.0, 1.0])

    def test_output_is_float(self):
        ids = mx.array([[10, 20]])
        pos = compute_frame_level_positions(ids, self.AUDIO_START, audio_step=0.5)
        assert pos.dtype == mx.float32


# --------------------------------------------------------------------------- #
# LearnableRotaryEmbedding (MLX)
# --------------------------------------------------------------------------- #

class TestMLXLearnableRotaryEmbedding:
    """Tests for the MLX learnable RoPE module."""

    def test_alpha_at_initialization(self):
        """With alpha_weight=0, sigmoid(0)=0.5, alpha = min + 0.5*(max-min)."""
        rope = LearnableRotaryEmbedding(dim=16, base=10000.0, alpha_min=0.0, alpha_max=2.0)
        alpha = rope.alpha.item()
        assert abs(alpha - 1.0) < 1e-4

    def test_alpha_in_range(self):
        rope = LearnableRotaryEmbedding(dim=16, base=10000.0, alpha_min=0.5, alpha_max=3.0)
        alpha = rope.alpha.item()
        assert 0.5 <= alpha <= 3.0

    def test_inv_freq_shape(self):
        rope = LearnableRotaryEmbedding(dim=16, base=10000.0)
        assert rope.inv_freq.shape == (8,)  # dim/2

    def test_forward_output_shapes(self):
        """Forward should return rotated queries and keys."""
        rope = LearnableRotaryEmbedding(dim=16, base=10000.0)
        batch, n_heads, seq, head_dim = 2, 4, 10, 16
        queries = mx.random.normal((batch, n_heads, seq, head_dim))
        keys = mx.random.normal((batch, n_heads, seq, head_dim))
        positions = mx.arange(seq, dtype=mx.float32)[None, :].broadcast_to((batch, seq))

        rot_q, rot_k = rope(queries, keys, positions)
        assert rot_q.shape == queries.shape
        assert rot_k.shape == keys.shape

    def test_apply_rotary_identity_at_position_zero(self):
        """At position 0, angles are 0, cos=1, sin=0 → output equals input."""
        rope = LearnableRotaryEmbedding(dim=4, base=10000.0)
        batch, n_heads, seq, head_dim = 1, 1, 1, 4
        queries = mx.random.normal((batch, n_heads, seq, head_dim))
        keys = mx.random.normal((batch, n_heads, seq, head_dim))
        positions = mx.zeros((batch, seq))

        rot_q, rot_k = rope(queries, keys, positions)
        # At position 0, cos=1, sin=0. x*1 + rotated*0 = x
        np.testing.assert_allclose(
            np.array(rot_q.tolist()), np.array(queries.tolist()), atol=1e-5
        )


# --------------------------------------------------------------------------- #
# sample_token
# --------------------------------------------------------------------------- #

class TestSampleToken:
    """Tests for the MLX token sampling function."""

    def test_greedy_sampling(self):
        """temperature=0 should return argmax."""
        logits = mx.array([0.1, 0.5, 0.9, 0.2])
        token = sample_token(logits, temperature=0, top_p=1.0)
        assert token.item() == 2

    def test_output_is_valid_index(self):
        """Sampled token should be a valid index into the logits."""
        logits = mx.random.normal((100,))
        token = sample_token(logits, temperature=1.0, top_p=0.95)
        assert 0 <= token.item() < 100

    def test_top_p_restricts_sampling(self):
        """With top_p very low, only the top token should be sampled."""
        # Create logits where one token dominates
        logits = mx.array([-100.0] * 10)
        logits = logits.at[5].add(200.0)  # token 5 dominates
        token = sample_token(logits, temperature=1.0, top_p=0.01)
        assert token.item() == 5


# --------------------------------------------------------------------------- #
# apply_repetition_penalty
# --------------------------------------------------------------------------- #

class TestApplyRepetitionPenalty:
    """Tests for the MLX repetition penalty function."""

    def test_no_penalty(self):
        """penalty=1.0 should return logits unchanged."""
        logits = mx.array([[1.0, 2.0, 3.0]])
        result = apply_repetition_penalty(logits, [0, 1], penalty=1.0)
        np.testing.assert_allclose(np.array(result.tolist()), np.array(logits.tolist()))

    def test_empty_generated_ids(self):
        """No generated IDs → logits unchanged."""
        logits = mx.array([[1.0, 2.0, 3.0]])
        result = apply_repetition_penalty(logits, [], penalty=1.5)
        np.testing.assert_allclose(np.array(result.tolist()), np.array(logits.tolist()))

    def test_positive_logits_reduced(self):
        """Positive logits for repeated tokens should be divided by penalty."""
        logits = mx.array([[0.0, 2.0, 0.0]])
        result = apply_repetition_penalty(logits, [1], penalty=2.0)
        # Token 1 has logit 2.0 > 0, should become 2.0/2.0 = 1.0
        assert abs(result[0, 1].item() - 1.0) < 1e-5

    def test_negative_logits_amplified(self):
        """Negative logits for repeated tokens should be multiplied by penalty."""
        logits = mx.array([[0.0, -2.0, 0.0]])
        result = apply_repetition_penalty(logits, [1], penalty=2.0)
        # Token 1 has logit -2.0 < 0, should become -2.0*2.0 = -4.0
        assert abs(result[0, 1].item() - (-4.0)) < 1e-5


# --------------------------------------------------------------------------- #
# KaniTTS2MLXModel.sanitize
# --------------------------------------------------------------------------- #

class TestWeightSanitization:
    """Tests for the MLX model weight sanitization."""

    def _make_minimal_args(self):
        return KaniTTSModelArgs(
            model_type="lfm2",
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            norm_eps=1e-5,
            conv_bias=True,
            conv_L_cache=4,
            block_dim=64,
            block_ff_dim=256,
            block_multiple_of=64,
            block_ffn_dim_multiplier=1.0,
            block_auto_adjust_ff_dim=False,
            rope_theta=10000.0,
            layer_types=["full_attention", "sliding_window"],
        )

    def test_conv_weight_transposed(self):
        """Conv weights in PyTorch layout should be transposed."""
        args = self._make_minimal_args()
        model = KaniTTS2MLXModel(args)
        # PyTorch conv layout: (out_channels, in_channels/groups, kernel_size)
        # If shape[-1] > shape[1], transpose
        weight_pt = mx.random.normal((64, 1, 4))  # shape[-1]=4 > shape[1]=1
        weights = {"model.layers.1.conv.conv.weight": weight_pt}
        sanitized = model.sanitize(weights)
        result = sanitized["model.layers.1.conv.conv.weight"]
        assert result.shape == (64, 4, 1)  # transposed

    def test_conv_weight_not_transposed_if_already_mlx(self):
        """Conv weights already in MLX layout should not be transposed."""
        args = self._make_minimal_args()
        model = KaniTTS2MLXModel(args)
        weight_mlx = mx.random.normal((64, 4, 1))  # shape[-1]=1 < shape[1]=4
        weights = {"model.layers.1.conv.conv.weight": weight_mlx}
        sanitized = model.sanitize(weights)
        result = sanitized["model.layers.1.conv.conv.weight"]
        assert result.shape == (64, 4, 1)

    def test_lm_head_dropped(self):
        """lm_head.weight should be dropped (tied to embed_tokens)."""
        args = self._make_minimal_args()
        model = KaniTTS2MLXModel(args)
        weights = {
            "lm_head.weight": mx.random.normal((100, 64)),
            "model.embed_tokens.weight": mx.random.normal((100, 64)),
        }
        sanitized = model.sanitize(weights)
        assert "lm_head.weight" not in sanitized
        assert "model.embed_tokens.weight" in sanitized

    def test_regular_weights_pass_through(self):
        """Non-special weights should pass through unchanged."""
        args = self._make_minimal_args()
        model = KaniTTS2MLXModel(args)
        weight = mx.random.normal((64, 64))
        weights = {"model.layers.0.self_attn.q_proj.weight": weight}
        sanitized = model.sanitize(weights)
        assert "model.layers.0.self_attn.q_proj.weight" in sanitized


# --------------------------------------------------------------------------- #
# PyTorch ↔ MLX frame-level position parity
# --------------------------------------------------------------------------- #

class TestPositionParity:
    """Verify that PyTorch and MLX frame-level positions produce identical results."""

    AUDIO_START = 64410

    def _pytorch_positions(self, ids_list, **kwargs):
        import torch
        from kani_tts.model import compute_frame_level_positions as pt_fn
        ids = torch.tensor([ids_list])
        return pt_fn(ids, self.AUDIO_START, **kwargs).numpy().tolist()[0]

    def _mlx_positions(self, ids_list, **kwargs):
        ids = mx.array([ids_list])
        return compute_frame_level_positions(ids, self.AUDIO_START, **kwargs).tolist()[0]

    def test_text_only_parity(self):
        ids = [10, 20, 30, 40, 50]
        pt = self._pytorch_positions(ids)
        ml = self._mlx_positions(ids)
        np.testing.assert_allclose(pt, ml, atol=1e-5)

    def test_mixed_parity(self):
        aud = self.AUDIO_START
        ids = [100, 200, aud, aud, aud, aud, 300]
        pt = self._pytorch_positions(ids)
        ml = self._mlx_positions(ids)
        np.testing.assert_allclose(pt, ml, atol=1e-5)

    def test_half_step_parity(self):
        aud = self.AUDIO_START
        ids = [100, aud, aud, aud, aud, aud, aud, aud, aud]
        pt = self._pytorch_positions(ids, tokens_per_frame=4, audio_step=0.5)
        ml = self._mlx_positions(ids, tokens_per_frame=4, audio_step=0.5)
        np.testing.assert_allclose(pt, ml, atol=1e-5)
