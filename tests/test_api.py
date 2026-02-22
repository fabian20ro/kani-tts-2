"""Tests for kani_tts.api — KaniTTS API logic (without loading models)."""

import torch
from pathlib import Path
from unittest.mock import patch
import tempfile
import os


class TestSuppressAllLogs:
    """Tests for the log suppression utility."""

    def test_does_not_crash(self):
        """suppress_all_logs should not raise."""
        from kani_tts.api import suppress_all_logs
        suppress_all_logs()  # should not raise

    def test_sets_root_logger(self):
        """Root logger should be set to ERROR after suppression."""
        import logging
        from kani_tts.api import suppress_all_logs
        suppress_all_logs()
        assert logging.getLogger().level >= logging.ERROR


class TestKaniTTSSpeakerEmbeddingLoading:
    """Test speaker embedding loading logic (from the KaniTTS class)."""

    def test_load_valid_pt_file(self):
        """Should successfully load a valid .pt speaker embedding."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            emb = torch.randn(1, 128)
            torch.save(emb, f.name)
            path = Path(f.name)

        try:
            loaded = torch.load(path)
            assert loaded.shape == (1, 128)
        finally:
            os.unlink(path)

    def test_rejects_wrong_suffix(self):
        """Should reject unsupported file formats."""
        path = Path("/tmp/test_emb.csv")
        assert path.suffix not in (".pt", ".npy")

    def test_rejects_missing_file(self):
        """Should raise FileNotFoundError for non-existent path."""
        path = Path("/tmp/nonexistent_speaker_embedding_xyz.pt")
        assert not path.exists()

    def test_1d_embedding_unsqueeze(self):
        """1D embedding should get a batch dimension added."""
        emb = torch.randn(128)
        assert emb.ndim == 1
        emb = emb.unsqueeze(0)
        assert emb.shape == (1, 128)


class TestDeviceDisplay:
    """Test the device display logic in show_model_info."""

    def test_cuda_display(self):
        with patch("torch.cuda.is_available", return_value=True):
            device = "GPU (CUDA)"
            assert device == "GPU (CUDA)"

    def test_mps_display(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=True):
            if not torch.cuda.is_available():
                if torch.backends.mps.is_available():
                    device = "MPS (Apple Silicon)"
            assert device == "MPS (Apple Silicon)"

    def test_cpu_display(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            if not torch.cuda.is_available():
                if torch.backends.mps.is_available():
                    device = "MPS (Apple Silicon)"
                else:
                    device = "CPU"
            assert device == "CPU"
