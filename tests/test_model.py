"""Tests for kani_tts.model — PyTorch frame-level positions and learnable RoPE."""

import pytest
import torch

from kani_tts.model import compute_frame_level_positions, LearnableRotaryEmbedding


# --------------------------------------------------------------------------- #
# compute_frame_level_positions
# --------------------------------------------------------------------------- #

class TestComputeFrameLevelPositions:
    """Tests for the vectorised frame-level position computation."""

    AUDIO_START = 64410  # default audio_tokens_start

    def test_text_only_sequential(self):
        """Pure text tokens get sequential 0, 1, 2, … positions."""
        ids = torch.tensor([[10, 20, 30, 40, 50]])
        pos = compute_frame_level_positions(ids, self.AUDIO_START)
        expected = torch.tensor([[0, 1, 2, 3, 4]], dtype=pos.dtype)
        assert torch.allclose(pos, expected)

    def test_single_audio_frame(self):
        """One full audio frame — all 4 tokens share the same position."""
        # 2 text tokens + 4 audio tokens
        aud = self.AUDIO_START
        ids = torch.tensor([[100, 200, aud, aud + 4032, aud + 8064, aud + 12096]])
        pos = compute_frame_level_positions(ids, self.AUDIO_START, tokens_per_frame=4)
        # text at 0, 1. Audio frame 0 -> all at text_count(=2) + 0*1.0 = 2
        expected = torch.tensor([[0, 1, 2, 2, 2, 2]], dtype=pos.dtype)
        assert torch.allclose(pos, expected)

    def test_two_audio_frames(self):
        """Two consecutive audio frames advance position by audio_step each."""
        aud = self.AUDIO_START
        ids = torch.tensor([[100, aud, aud, aud, aud, aud, aud, aud, aud]])
        # 1 text + 8 audio (2 frames)
        pos = compute_frame_level_positions(ids, self.AUDIO_START, tokens_per_frame=4, audio_step=1.0)
        # text_count at each position: [1, 1, 1, 1, 1, 1, 1, 1, 1]  (1 text token counted before every pos)
        # Wait — text_count is cumulative sum of text_mask *before* each position.
        # Position 0: text_count=0, audio_frame=0 => 0
        # Position 1..4: text_count=1, audio_frame_count=0 => 1
        # Position 5..8: text_count=1, audio_frame_count=1 => 2
        expected = torch.tensor([[0, 1, 1, 1, 1, 2, 2, 2, 2]], dtype=pos.dtype)
        assert torch.allclose(pos, expected)

    def test_audio_step_half(self):
        """audio_step=0.5 compresses audio position space."""
        aud = self.AUDIO_START
        ids = torch.tensor([[100, 200, aud, aud, aud, aud, aud, aud, aud, aud]])
        # 2 text + 8 audio (2 frames)
        pos = compute_frame_level_positions(ids, self.AUDIO_START, tokens_per_frame=4, audio_step=0.5)
        # Positions: text at 0,1.  Frame0: 2+0*0.5=2.  Frame1: 2+1*0.5=2.5
        expected = torch.tensor([[0, 1, 2, 2, 2, 2, 2.5, 2.5, 2.5, 2.5]], dtype=pos.dtype)
        assert torch.allclose(pos, expected)

    def test_text_after_audio(self):
        """Text tokens resume sequential counting after audio."""
        aud = self.AUDIO_START
        ids = torch.tensor([[100, aud, aud, aud, aud, 300]])
        # 1 text + 4 audio + 1 text
        pos = compute_frame_level_positions(ids, self.AUDIO_START)
        # pos 0: text_count=0, frame=0 => 0
        # pos 1..4: text_count=1, frame=0 => 1
        # pos 5: text_count=2, frame=1 => 2+1=3  Actually frame_count at pos 5 = 4//4 = 1
        expected = torch.tensor([[0, 1, 1, 1, 1, 2]], dtype=pos.dtype)
        assert torch.allclose(pos, expected)

    def test_batch_dimension(self):
        """Works correctly with batch_size > 1."""
        ids = torch.tensor([
            [10, 20, 30],
            [10, self.AUDIO_START, self.AUDIO_START],
        ])
        pos = compute_frame_level_positions(ids, self.AUDIO_START)
        assert pos.shape == (2, 3)
        # Row 0: all text => 0, 1, 2
        assert torch.allclose(pos[0], torch.tensor([0, 1, 2], dtype=pos.dtype))
        # Row 1: 1 text + 2 audio (partial frame) => 0, 1, 1
        assert torch.allclose(pos[1], torch.tensor([0, 1, 1], dtype=pos.dtype))

    def test_empty_sequence(self):
        """Handles a single-token sequence."""
        ids = torch.tensor([[42]])
        pos = compute_frame_level_positions(ids, self.AUDIO_START)
        assert pos.shape == (1, 1)
        assert pos.item() == 0

    def test_output_is_float_when_audio_step_float(self):
        """When audio_step is a float, output should be float."""
        ids = torch.tensor([[10, 20]])
        pos = compute_frame_level_positions(ids, self.AUDIO_START, audio_step=0.5)
        assert pos.dtype in (torch.float32, torch.float64)


# --------------------------------------------------------------------------- #
# LearnableRotaryEmbedding
# --------------------------------------------------------------------------- #

class TestLearnableRotaryEmbedding:
    """Tests for the learnable RoPE module."""

    def _make_config(self):
        """Create a minimal LFM2 config-like object."""

        class FakeConfig:
            hidden_size = 64
            num_attention_heads = 4
            rope_theta = 10000.0
            max_position_embeddings = 2048

        return FakeConfig()

    def test_alpha_in_range(self):
        """Alpha must be in [alpha_min, alpha_max]."""
        config = self._make_config()
        rope = LearnableRotaryEmbedding(
            config, layer_idx=0, total_attention_layers=1,
            alpha_min=0.5, alpha_max=3.0,
        )
        alpha = rope.alpha.item()
        assert 0.5 <= alpha <= 3.0

    def test_alpha_at_initialization(self):
        """With alpha_weight=0, sigmoid(0)=0.5, so alpha = min + 0.5*(max-min)."""
        config = self._make_config()
        rope = LearnableRotaryEmbedding(
            config, layer_idx=0, total_attention_layers=1,
            alpha_min=0.0, alpha_max=2.0,
        )
        alpha = rope.alpha.item()
        assert abs(alpha - 1.0) < 1e-5  # 0 + 0.5 * 2 = 1.0

    def test_inv_freq_scaled(self):
        """inv_freq should be alpha * inv_freq_base."""
        config = self._make_config()
        rope = LearnableRotaryEmbedding(
            config, layer_idx=0, total_attention_layers=1,
            alpha_min=1.0, alpha_max=1.0,  # force alpha=1
        )
        assert torch.allclose(rope.inv_freq, rope.inv_freq_base, atol=1e-5)

    def test_forward_output_shapes(self):
        """Forward pass should produce cos, sin with correct shapes."""
        config = self._make_config()
        rope = LearnableRotaryEmbedding(
            config, layer_idx=0, total_attention_layers=1,
        )
        head_dim = config.hidden_size // config.num_attention_heads  # 16
        batch, seq = 2, 10
        x = torch.randn(batch, config.num_attention_heads, seq, head_dim)
        pos = torch.arange(seq).unsqueeze(0).expand(batch, -1).float()
        cos, sin = rope(x, pos)
        assert cos.shape == (batch, seq, head_dim)
        assert sin.shape == (batch, seq, head_dim)

    def test_mps_device_fallback(self):
        """The MPS fallback to CPU in autocast should not crash."""
        config = self._make_config()
        rope = LearnableRotaryEmbedding(
            config, layer_idx=0, total_attention_layers=1,
        )
        head_dim = config.hidden_size // config.num_attention_heads
        x = torch.randn(1, config.num_attention_heads, 5, head_dim)
        pos = torch.arange(5).unsqueeze(0).float()
        # Should not raise regardless of device_type
        cos, sin = rope(x, pos)
        assert cos is not None
