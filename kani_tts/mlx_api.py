"""
MLX-based API for KaniTTS-2 on Apple Silicon.

Usage:
    from kani_tts import KaniTTSMLX

    model = KaniTTSMLX("./kani-tts-2-en-mlx")
    audio, text = model("Hello, world!")
    model.save_audio(audio, "output.wav")
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ._utils import suppress_all_logs, save_audio as _save_audio, show_language_tags as _show_language_tags
from .mlx_core import MLXAudioPlayer, MLXKaniModel, MLXTTSConfig


class KaniTTSMLX:
    """
    MLX-based interface for KaniTTS-2 text-to-speech on Apple Silicon.

    Example:
        >>> model = KaniTTSMLX("./kani-tts-2-en-mlx")
        >>> audio, text = model("Hello, world!")
        >>> model.save_audio(audio, "output.wav")
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 3000,
        tokeniser_length: int = 64400,
        suppress_logs: bool = True,
        show_info: bool = True,
        tokens_per_frame: Optional[int] = None,
        audio_step: Optional[float] = None,
        use_learnable_rope: Optional[bool] = None,
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
        speaker_emb_dim: Optional[int] = None,
    ):
        """
        Initialize MLX-based KaniTTS-2 model.

        Args:
            model_path: Path to converted MLX model directory
            max_new_tokens: Maximum tokens to generate (default: 3000)
            tokeniser_length: Text tokenizer vocabulary size (default: 64400)
            suppress_logs: Suppress library logs (default: True)
            show_info: Display model info on init (default: True)
            tokens_per_frame: Tokens per audio frame (None = from config)
            audio_step: Position step per frame (None = from config)
            use_learnable_rope: Enable learnable RoPE (None = from config)
            alpha_min: Min alpha for learnable RoPE (None = from config)
            alpha_max: Max alpha for learnable RoPE (None = from config)
            speaker_emb_dim: Speaker embedding dimension (None = from config)
        """
        if suppress_logs:
            suppress_all_logs()

        self.config = MLXTTSConfig(
            tokeniser_length=tokeniser_length,
            max_new_tokens=max_new_tokens,
            tokens_per_frame=tokens_per_frame,
            audio_step=audio_step,
            use_learnable_rope=use_learnable_rope,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            speaker_emb_dim=speaker_emb_dim,
        )
        self.model_path = model_path
        self.player = MLXAudioPlayer(self.config)
        self.model = MLXKaniModel(self.config, model_path, self.player)
        self.status = self.model.status
        self.language_tags_list = self.model.language_tags_list
        self.sample_rate = self.config.sample_rate

        # Sync config from loaded model
        self.config.tokens_per_frame = self.model.args.tokens_per_frame
        self.config.audio_step = self.model.args.audio_step
        self.config.use_learnable_rope = self.model.args.use_learnable_rope
        self.config.alpha_min = self.model.args.alpha_min
        self.config.alpha_max = self.model.args.alpha_max
        self.config.speaker_emb_dim = self.model.args.speaker_emb_dim

        if show_info:
            self._show_model_info()

    def __call__(
        self,
        text: str,
        language_tag: Optional[str] = None,
        speaker_emb=None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> Tuple[np.ndarray, str]:
        """
        Generate audio from text.

        Args:
            text: Input text to synthesize
            language_tag: Language tag (for multilingual models)
            speaker_emb: Speaker embedding (mx.array, numpy array, or path to .npy file)
            temperature: Sampling temperature (default: 1.0)
            top_p: Nucleus sampling threshold (default: 0.95)
            repetition_penalty: Repetition penalty (default: 1.1)

        Returns:
            (audio_waveform, text) tuple
        """
        return self.generate(text, language_tag, speaker_emb, temperature, top_p, repetition_penalty)

    def generate(
        self,
        text: str,
        language_tag: Optional[str] = None,
        speaker_emb=None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> Tuple[np.ndarray, str]:
        """Generate audio from text."""
        import mlx.core as mx

        # Handle speaker embedding loading
        if speaker_emb is not None:
            if isinstance(speaker_emb, (str, Path)):
                speaker_emb = self.load_speaker_embedding(speaker_emb)
            elif isinstance(speaker_emb, np.ndarray):
                speaker_emb = mx.array(speaker_emb)
            # If it's already mx.array, use as-is

            # Ensure batch dimension
            if speaker_emb.ndim == 1:
                speaker_emb = speaker_emb[None, :]

        return self.model.run_model(
            text, language_tag, speaker_emb, temperature, top_p, repetition_penalty
        )

    def load_speaker_embedding(self, path):
        """Load speaker embedding from file (.npy or .pt)."""
        import mlx.core as mx

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Speaker embedding file not found: {path}")

        if path.suffix == ".npy":
            arr = np.load(path)
            return mx.array(arr)
        elif path.suffix == ".pt":
            import torch
            t = torch.load(path, map_location="cpu", weights_only=True)
            return mx.array(t.numpy())
        else:
            raise ValueError(f"Unsupported speaker embedding format: {path.suffix}. Use .npy or .pt")

    def save_audio(self, audio: np.ndarray, output_path: str):
        """Save audio waveform to file."""
        _save_audio(audio, output_path, self.sample_rate)

    def _show_model_info(self):
        """Display model information."""
        print()
        print("=" * 58)
        print("  KaniTTS-2 (MLX - Apple Silicon)")
        print("=" * 58)
        print()
        print(f"  Model: {self.model_path}")
        print("  Device: MLX (Apple Silicon - Unified Memory)")
        print()

        if self.status == "available_language_tags":
            print(f"  Mode: Available language tags ({len(self.language_tags_list)} tags)")
            if self.language_tags_list and len(self.language_tags_list) <= 5:
                print(f"  Tags: {', '.join(self.language_tags_list)}")
        else:
            print("  Mode: No language tags")

        print()
        print("  Configuration:")
        print(f"    Sample Rate: {self.sample_rate} Hz")
        print(f"    Max Tokens: {self.config.max_new_tokens}")
        print(f"    Speaker Embedding Dim: {self.config.speaker_emb_dim}")
        print(f"    Tokens per Frame: {self.config.tokens_per_frame}")
        print(f"    Audio Step: {self.config.audio_step}")
        if self.config.use_learnable_rope:
            print(f"    Learnable RoPE: Enabled [{self.config.alpha_min}, {self.config.alpha_max}]")
        else:
            print("    Learnable RoPE: Disabled (standard RoPE)")
        print()
        print("  Ready to generate speech!")
        print()

    def show_language_tags(self):
        """Display available language tags."""
        _show_language_tags(self.status, self.language_tags_list)
