"""Tests for scripts/convert_to_mlx.py — weight conversion logic."""

import pytest
import numpy as np
import json
import os
import tempfile

mlx = pytest.importorskip("mlx.core", reason="mlx not available")
mx = mlx


class TestConvertWeights:
    """Tests for the weight conversion function."""

    def _make_fake_model_dir(self, tmpdir, weights, config):
        """Create a fake model directory with safetensors and config."""
        from safetensors.torch import save_file
        import torch

        model_dir = os.path.join(tmpdir, "model")
        os.makedirs(model_dir, exist_ok=True)

        # Save fake weights as safetensors
        torch_weights = {k: torch.tensor(v) for k, v in weights.items()}
        save_file(torch_weights, os.path.join(model_dir, "model.safetensors"))

        # Save config
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config, f)

        return model_dir

    def test_conv_weight_transposed(self):
        """Conv weights in PyTorch layout should be transposed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # PyTorch Conv1d: (out_channels, in_channels/groups, kernel_size)
            # When kernel_size > in_channels/groups, shape[-1] > shape[1]
            conv_weight = np.random.randn(32, 1, 4).astype(np.float32)
            weights = {"model.layers.0.conv.conv.weight": conv_weight}
            config = {"model_type": "lfm2", "hidden_size": 32}

            model_dir = self._make_fake_model_dir(tmpdir, weights, config)
            output_dir = os.path.join(tmpdir, "output")

            from scripts.convert_to_mlx import convert_weights
            convert_weights(model_dir, output_dir, dtype="float32")

            result = mx.load(os.path.join(output_dir, "model.safetensors"))
            key = "model.layers.0.conv.conv.weight"
            assert key in result
            # Should be transposed: (32, 4, 1)
            assert result[key].shape == (32, 4, 1)

    def test_lm_head_dropped_when_tied(self):
        """lm_head.weight should be dropped when embed_tokens.weight exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            embed = np.random.randn(100, 64).astype(np.float32)
            weights = {
                "model.embed_tokens.weight": embed,
                "lm_head.weight": embed.copy(),
            }
            config = {"model_type": "lfm2"}
            model_dir = self._make_fake_model_dir(tmpdir, weights, config)
            output_dir = os.path.join(tmpdir, "output")

            from scripts.convert_to_mlx import convert_weights
            convert_weights(model_dir, output_dir)

            result = mx.load(os.path.join(output_dir, "model.safetensors"))
            assert "lm_head.weight" not in result
            assert "model.embed_tokens.weight" in result

    def test_config_copied_with_mlx_metadata(self):
        """Output config should include mlx_backend=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights = {"model.embed_tokens.weight": np.random.randn(10, 8).astype(np.float32)}
            config = {"model_type": "lfm2", "hidden_size": 8}
            model_dir = self._make_fake_model_dir(tmpdir, weights, config)
            output_dir = os.path.join(tmpdir, "output")

            from scripts.convert_to_mlx import convert_weights
            convert_weights(model_dir, output_dir)

            with open(os.path.join(output_dir, "config.json")) as f:
                out_config = json.load(f)
            assert out_config["mlx_backend"] is True
            assert out_config["model_type"] == "lfm2"

    def test_custom_kanitts_weights_preserved(self):
        """Speaker embedding projection and learnable RoPE weights should be kept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights = {
                "model.embed_tokens.weight": np.random.randn(10, 8).astype(np.float32),
                "model.speaker_emb_projection.weight": np.random.randn(8, 128).astype(np.float32),
                "model.learnable_rope_layers.0.alpha_weight": np.array([0.5], dtype=np.float32),
            }
            config = {"model_type": "lfm2"}
            model_dir = self._make_fake_model_dir(tmpdir, weights, config)
            output_dir = os.path.join(tmpdir, "output")

            from scripts.convert_to_mlx import convert_weights
            convert_weights(model_dir, output_dir, dtype="float32")

            result = mx.load(os.path.join(output_dir, "model.safetensors"))
            assert "model.speaker_emb_projection.weight" in result
            assert "model.learnable_rope_layers.0.alpha_weight" in result

    def test_rotary_emb_buffers_skipped(self):
        """Standard rotary_emb/pos_emb buffers should be dropped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights = {
                "model.embed_tokens.weight": np.random.randn(10, 8).astype(np.float32),
                "model.rotary_emb.inv_freq": np.random.randn(4).astype(np.float32),
                "model.pos_emb.inv_freq": np.random.randn(4).astype(np.float32),
            }
            config = {"model_type": "lfm2"}
            model_dir = self._make_fake_model_dir(tmpdir, weights, config)
            output_dir = os.path.join(tmpdir, "output")

            from scripts.convert_to_mlx import convert_weights
            convert_weights(model_dir, output_dir)

            result = mx.load(os.path.join(output_dir, "model.safetensors"))
            assert "model.rotary_emb.inv_freq" not in result
            assert "model.pos_emb.inv_freq" not in result
