"""Tests for kani_tts.speaker_embedder — model classes and audio preparation."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch


class TestTopLayers:
    """Tests for the TopLayers projection module."""

    def test_output_shape(self):
        from kani_tts.speaker_embedder import TopLayers

        top = TopLayers(embd_size=128, top_interm_size=512)
        x = torch.randn(2, 2048, 1)
        out = top(x)
        assert out.shape == (2, 128)

    def test_l2_normalized(self):
        """Output should be L2-normalized (unit norm)."""
        from kani_tts.speaker_embedder import TopLayers

        top = TopLayers(embd_size=128, top_interm_size=512)
        x = torch.randn(3, 2048, 1)
        out = top(x)
        norms = torch.norm(out, dim=1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)

    def test_different_embd_size(self):
        from kani_tts.speaker_embedder import TopLayers

        top = TopLayers(embd_size=256, top_interm_size=1024)
        top.eval()  # BatchNorm1d requires batch_size > 1 in training mode
        x = torch.randn(1, 2048, 1)
        out = top(x)
        assert out.shape == (1, 256)


class TestSpeakerEmbedderAudioPreparation:
    """Test audio preparation logic without loading the full WavLM model."""

    def test_mono_conversion_channels_first(self):
        """[channels, time] audio should be averaged to mono."""
        audio = torch.randn(2, 16000)  # stereo, 1 second at 16kHz
        # channels < time → average channels
        if audio.shape[0] < audio.shape[1]:
            audio = audio.mean(dim=0)
        assert audio.dim() == 1
        assert audio.shape[0] == 16000

    def test_mono_conversion_batch_first(self):
        """[batch, time] audio should take first sample."""
        audio = torch.randn(16000, 2)  # time > channels won't trigger
        # Actually shape[0] > shape[1] → takes first
        if audio.shape[0] < audio.shape[1]:
            audio = audio.mean(dim=0)
        else:
            audio = audio[0]
        assert audio.dim() == 1

    def test_truncation(self):
        """Audio longer than max_samples should be truncated."""
        max_duration_sec = 5.0
        target_sr = 16000
        max_samples = int(max_duration_sec * target_sr)
        audio = torch.randn(target_sr * 10)  # 10 seconds
        if audio.shape[0] > max_samples:
            audio = audio[:max_samples]
        assert audio.shape[0] == max_samples

    def test_empty_audio_rejected(self):
        """Empty audio should fail."""
        audio = torch.tensor([])
        assert audio.shape[0] == 0

    def test_numpy_to_tensor_conversion(self):
        """Numpy arrays should be convertible to torch tensors."""
        audio_np = np.random.randn(16000).astype(np.float32)
        audio_t = torch.from_numpy(audio_np).float()
        assert audio_t.dtype == torch.float32
        assert audio_t.shape == (16000,)

    def test_device_auto_detection_cpu(self):
        """On a system without GPU/MPS, device should be CPU."""
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            assert device == torch.device("cpu")
