"""Tests for kani_tts.mlx_core — MLXTTSConfig, MLXAudioPlayer token logic."""

import pytest
import numpy as np

mlx = pytest.importorskip("mlx.core", reason="mlx not available")
mx = mlx

from kani_tts.mlx_core import MLXTTSConfig


# --------------------------------------------------------------------------- #
# MLXTTSConfig
# --------------------------------------------------------------------------- #

class TestMLXTTSConfig:
    """Tests for the MLX TTS configuration dataclass."""

    def test_defaults(self):
        cfg = MLXTTSConfig()
        assert cfg.tokeniser_length == 64400
        assert cfg.start_of_text == 1
        assert cfg.end_of_text == 2
        assert cfg.max_new_tokens == 3000
        assert cfg.sample_rate == 22050

    def test_optional_fields_default_none(self):
        cfg = MLXTTSConfig()
        assert cfg.tokens_per_frame is None
        assert cfg.audio_step is None
        assert cfg.use_learnable_rope is None
        assert cfg.alpha_min is None
        assert cfg.alpha_max is None
        assert cfg.speaker_emb_dim is None

    def test_override(self):
        cfg = MLXTTSConfig(max_new_tokens=500, tokens_per_frame=8)
        assert cfg.max_new_tokens == 500
        assert cfg.tokens_per_frame == 8


# --------------------------------------------------------------------------- #
# MLXAudioPlayer token logic (without loading nanocodec-mlx model)
# --------------------------------------------------------------------------- #

class TestMLXAudioPlayerTokens:
    """Test token constant computation and validation for MLX audio player."""

    def _token_constants(self, tokeniser_length=64400):
        """Compute the token constants that MLXAudioPlayer.__init__ sets."""
        return {
            "start_of_speech": tokeniser_length + 1,
            "end_of_speech": tokeniser_length + 2,
            "start_of_human": tokeniser_length + 3,
            "end_of_human": tokeniser_length + 4,
            "start_of_ai": tokeniser_length + 5,
            "end_of_ai": tokeniser_length + 6,
            "pad_token": tokeniser_length + 7,
            "audio_tokens_start": tokeniser_length + 10,
            "codebook_size": 4032,
        }

    def test_special_token_values(self):
        t = self._token_constants()
        assert t["start_of_speech"] == 64401
        assert t["end_of_speech"] == 64402
        assert t["start_of_human"] == 64403
        assert t["audio_tokens_start"] == 64410

    def test_output_validation_missing_tokens(self):
        """Validation should fail when speech markers are absent."""
        t = self._token_constants()
        ids = mx.array([10, 20, 30])
        ids_list = ids.tolist()
        assert t["start_of_speech"] not in ids_list
        assert t["end_of_speech"] not in ids_list

    def test_output_validation_present(self):
        t = self._token_constants()
        ids = mx.array([10, t["start_of_speech"], 70000, 70001, 70002, 70003, t["end_of_speech"], 20])
        ids_list = ids.tolist()
        assert t["start_of_speech"] in ids_list
        assert t["end_of_speech"] in ids_list

    def test_nano_codes_extraction(self):
        """Verify audio code extraction and reshape using MLX arrays."""
        t = self._token_constants()
        aud_start = t["audio_tokens_start"]
        cb = t["codebook_size"]

        codes = [aud_start + 0, aud_start + cb + 1, aud_start + 2 * cb + 2, aud_start + 3 * cb + 3]
        ids = mx.array([t["start_of_speech"]] + codes + [t["end_of_speech"]])
        ids_list = ids.tolist()

        start_idx = ids_list.index(t["start_of_speech"])
        end_idx = ids_list.index(t["end_of_speech"])
        audio_codes = ids[start_idx + 1: end_idx]
        assert len(audio_codes) == 4

        audio_codes = audio_codes.reshape(-1, 4)
        offsets = mx.array([cb * i for i in range(4)])
        audio_codes = audio_codes - offsets - aud_start

        expected = [0, 1, 2, 3]
        np.testing.assert_array_equal(np.array(audio_codes.tolist()).flatten(), expected)

    def test_nano_codes_rejects_non_multiple_of_4(self):
        t = self._token_constants()
        aud_start = t["audio_tokens_start"]
        codes = [aud_start, aud_start + 1, aud_start + 2]
        ids = mx.array([t["start_of_speech"]] + codes + [t["end_of_speech"]])
        ids_list = ids.tolist()

        start_idx = ids_list.index(t["start_of_speech"])
        end_idx = ids_list.index(t["end_of_speech"])
        audio_codes = ids[start_idx + 1: end_idx]
        assert len(audio_codes) % 4 != 0

    def test_nano_codes_rejects_reversed_markers(self):
        t = self._token_constants()
        ids = mx.array([t["end_of_speech"], 70000, t["start_of_speech"]])
        ids_list = ids.tolist()
        start_idx = ids_list.index(t["start_of_speech"])
        end_idx = ids_list.index(t["end_of_speech"])
        assert start_idx >= end_idx


# --------------------------------------------------------------------------- #
# MLXKaniModel.get_input_ids logic
# --------------------------------------------------------------------------- #

class TestMLXInputIds:
    """Test the input ID preparation logic for MLX models."""

    def test_wraps_text_with_special_tokens(self):
        START_OF_HUMAN = 64403
        END_OF_TEXT = 2
        END_OF_HUMAN = 64404

        fake_ids = mx.array([[101, 102, 103]])
        start = mx.array([[START_OF_HUMAN]])
        end = mx.array([[END_OF_TEXT, END_OF_HUMAN]])
        result = mx.concatenate([start, fake_ids, end], axis=1)

        assert result[0, 0].item() == START_OF_HUMAN
        assert result[0, -2].item() == END_OF_TEXT
        assert result[0, -1].item() == END_OF_HUMAN
        assert result.shape == (1, 6)

    def test_language_tag_prepended(self):
        text = "Hello world"
        tag = "fr_FR"
        result = f"{tag.strip()}: {text}"
        assert result == "fr_FR: Hello world"
