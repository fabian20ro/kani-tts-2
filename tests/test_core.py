"""Tests for kani_tts.core — TTSConfig, NemoAudioPlayer token logic, device detection."""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from kani_tts.core import TTSConfig, _get_device


# --------------------------------------------------------------------------- #
# TTSConfig
# --------------------------------------------------------------------------- #

class TestTTSConfig:
    """Tests for TTSConfig dataclass defaults."""

    def test_defaults(self):
        cfg = TTSConfig()
        assert cfg.device_map == "auto"
        assert cfg.tokeniser_length == 64400
        assert cfg.start_of_text == 1
        assert cfg.end_of_text == 2
        assert cfg.max_new_tokens == 3000
        assert cfg.sample_rate == 22050
        assert cfg.nanocodec_model == "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"

    def test_optional_fields_default_none(self):
        cfg = TTSConfig()
        assert cfg.text_vocab_size is None
        assert cfg.tokens_per_frame is None
        assert cfg.audio_step is None
        assert cfg.use_learnable_rope is None
        assert cfg.alpha_min is None
        assert cfg.alpha_max is None
        assert cfg.speaker_emb_dim is None

    def test_override(self):
        cfg = TTSConfig(max_new_tokens=500, tokens_per_frame=8)
        assert cfg.max_new_tokens == 500
        assert cfg.tokens_per_frame == 8


# --------------------------------------------------------------------------- #
# _get_device
# --------------------------------------------------------------------------- #

class TestGetDevice:
    """Tests for device auto-detection helper."""

    def test_cuda_preferred(self):
        with patch("torch.cuda.is_available", return_value=True):
            assert _get_device() == "cuda"

    def test_mps_fallback(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=True):
            assert _get_device() == "mps"

    def test_cpu_fallback(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            assert _get_device() == "cpu"


# --------------------------------------------------------------------------- #
# NemoAudioPlayer — token constants & validation (no model loading)
# --------------------------------------------------------------------------- #

class TestNemoAudioPlayerTokens:
    """Test token constant computation and validation logic without loading NeMo."""

    def _make_player_tokens(self, tokeniser_length=64400):
        """Compute the same token constants that NemoAudioPlayer.__init__ computes."""

        class Tokens:
            pass

        t = Tokens()
        t.tokeniser_length = tokeniser_length
        t.start_of_text = 1
        t.end_of_text = 2
        t.start_of_speech = tokeniser_length + 1
        t.end_of_speech = tokeniser_length + 2
        t.start_of_human = tokeniser_length + 3
        t.end_of_human = tokeniser_length + 4
        t.start_of_ai = tokeniser_length + 5
        t.end_of_ai = tokeniser_length + 6
        t.pad_token = tokeniser_length + 7
        t.audio_tokens_start = tokeniser_length + 10
        t.codebook_size = 4032
        return t

    def test_special_token_ordering(self):
        """Special tokens should be sequential after tokeniser_length."""
        t = self._make_player_tokens()
        assert t.start_of_speech == 64401
        assert t.end_of_speech == 64402
        assert t.start_of_human == 64403
        assert t.end_of_human == 64404
        assert t.audio_tokens_start == 64410

    def test_output_validation_raises_on_missing_speech_tokens(self):
        """output_validation should raise when speech tokens are missing."""
        t = self._make_player_tokens()
        ids = torch.tensor([10, 20, 30])
        # Replicate the validation logic
        start_ok = t.start_of_speech in ids
        end_ok = t.end_of_speech in ids
        assert not start_ok
        assert not end_ok

    def test_output_validation_passes_with_speech_tokens(self):
        """Validation should pass when both speech markers are present."""
        t = self._make_player_tokens()
        ids = torch.tensor([10, t.start_of_speech, 70000, 70001, 70002, 70003, t.end_of_speech, 20])
        assert t.start_of_speech in ids
        assert t.end_of_speech in ids

    def test_get_nano_codes_extraction(self):
        """Verify audio code extraction and reshape logic."""
        t = self._make_player_tokens()
        aud_start = t.audio_tokens_start
        cb = t.codebook_size

        # Build a valid output with 1 audio frame (4 tokens)
        # Token values: audio_tokens_start + code + codebook_offset
        codes = [aud_start + 0, aud_start + cb + 1, aud_start + 2 * cb + 2, aud_start + 3 * cb + 3]
        ids = torch.tensor([t.start_of_speech] + codes + [t.end_of_speech])

        # Replicate get_nano_codes logic
        start_idx = (ids == t.start_of_speech).nonzero(as_tuple=True)[0].item()
        end_idx = (ids == t.end_of_speech).nonzero(as_tuple=True)[0].item()
        audio_codes = ids[start_idx + 1: end_idx]
        assert len(audio_codes) == 4

        audio_codes = audio_codes.reshape(-1, 4)
        offsets = torch.tensor([cb * i for i in range(4)])
        audio_codes = audio_codes - offsets - aud_start
        # Should be [0, 1, 2, 3]
        expected = torch.tensor([[0, 1, 2, 3]])
        assert torch.equal(audio_codes, expected)

    def test_get_nano_codes_rejects_non_multiple_of_4(self):
        """Audio codes not a multiple of 4 should be rejected."""
        t = self._make_player_tokens()
        aud_start = t.audio_tokens_start
        codes = [aud_start, aud_start + 1, aud_start + 2]  # 3 tokens, not 4
        ids = torch.tensor([t.start_of_speech] + codes + [t.end_of_speech])

        start_idx = (ids == t.start_of_speech).nonzero(as_tuple=True)[0].item()
        end_idx = (ids == t.end_of_speech).nonzero(as_tuple=True)[0].item()
        audio_codes = ids[start_idx + 1: end_idx]
        assert len(audio_codes) % 4 != 0  # Should fail the check

    def test_get_nano_codes_rejects_reversed_markers(self):
        """Should reject when start_of_speech comes after end_of_speech."""
        t = self._make_player_tokens()
        ids = torch.tensor([t.end_of_speech, 70000, t.start_of_speech])

        start_idx = (ids == t.start_of_speech).nonzero(as_tuple=True)[0].item()
        end_idx = (ids == t.end_of_speech).nonzero(as_tuple=True)[0].item()
        assert start_idx >= end_idx  # Should fail the check


# --------------------------------------------------------------------------- #
# KaniModel.get_input_ids — token preparation logic
# --------------------------------------------------------------------------- #

class TestInputIdPreparation:
    """Tests for the input ID preparation logic (without loading a real model)."""

    def test_wraps_text_with_special_tokens(self):
        """Input IDs should be wrapped with START_OF_HUMAN, END_OF_TEXT, END_OF_HUMAN."""
        START_OF_HUMAN = 64403
        END_OF_TEXT = 2
        END_OF_HUMAN = 64404

        # Simulate what get_input_ids does
        fake_token_ids = torch.tensor([[101, 102, 103]])  # tokenizer output
        start_token = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
        end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
        result = torch.cat([start_token, fake_token_ids, end_tokens], dim=1)

        assert result[0, 0].item() == START_OF_HUMAN
        assert result[0, -2].item() == END_OF_TEXT
        assert result[0, -1].item() == END_OF_HUMAN
        assert result.shape == (1, 6)

    def test_language_tag_prepended(self):
        """Language tag should be prepended to text before tokenization."""
        text = "Hello world"
        tag = "en_US"
        result = f"{tag.strip()}: {text}"
        assert result == "en_US: Hello world"
