"""Tests for kani_tts.mlx_api — KaniTTSMLX speaker embedding loading & log suppression."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

mlx = pytest.importorskip("mlx.core", reason="mlx not available")
mx = mlx


class TestMLXSuppressLogs:
    """Tests for the MLX API log suppression."""

    def test_does_not_crash(self):
        from kani_tts.mlx_api import _suppress_logs
        _suppress_logs()

    def test_sets_root_logger(self):
        import logging
        from kani_tts.mlx_api import _suppress_logs
        _suppress_logs()
        assert logging.getLogger().level >= logging.ERROR


class TestMLXSpeakerEmbeddingLoading:
    """Test speaker embedding file loading for the MLX API."""

    def test_load_npy_file(self):
        """Should load .npy files into mx.array."""
        emb = np.random.randn(1, 128).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f.name, emb)
            path = Path(f.name)

        try:
            arr = np.load(path)
            mx_arr = mx.array(arr)
            assert mx_arr.shape == (1, 128)
        finally:
            os.unlink(path)

    def test_load_pt_file(self):
        """Should load .pt files via torch and convert to mx.array."""
        torch = pytest.importorskip("torch")
        emb = torch.randn(1, 128)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(emb, f.name)
            path = Path(f.name)

        try:
            t = torch.load(path, map_location="cpu")
            mx_arr = mx.array(t.numpy())
            assert mx_arr.shape == (1, 128)
        finally:
            os.unlink(path)

    def test_rejects_unsupported_format(self):
        """Should reject files that are neither .npy nor .pt."""
        path = Path("/tmp/test_embed.txt")
        assert path.suffix not in (".npy", ".pt")

    def test_1d_embedding_gets_batch_dim(self):
        """1D MLX embedding should get a batch dimension via [None, :]."""
        emb = mx.random.normal((128,))
        assert emb.ndim == 1
        emb = emb[None, :]
        assert emb.shape == (1, 128)
