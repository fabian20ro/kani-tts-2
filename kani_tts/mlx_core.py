"""MLX-based core components for KaniTTS-2 on Apple Silicon."""

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from .mlx_model import KaniTTS2MLXModel, KaniTTSModelArgs, generate


@dataclass
class MLXTTSConfig:
    """Configuration for MLX TTS model."""
    tokeniser_length: int = 64400
    start_of_text: int = 1
    end_of_text: int = 2
    max_new_tokens: int = 3000
    sample_rate: int = 22050

    # Model architecture parameters (None = read from model config)
    tokens_per_frame: Optional[int] = None
    audio_step: Optional[float] = None
    use_learnable_rope: Optional[bool] = None
    alpha_min: Optional[float] = None
    alpha_max: Optional[float] = None
    speaker_emb_dim: Optional[int] = None


class MLXAudioPlayer:
    """Handles audio codec operations using nanocodec-mlx."""

    def __init__(self, config: MLXTTSConfig):
        self.conf = config
        self.tokeniser_length = config.tokeniser_length
        self.start_of_text = config.start_of_text
        self.end_of_text = config.end_of_text
        self.start_of_speech = self.tokeniser_length + 1
        self.end_of_speech = self.tokeniser_length + 2
        self.start_of_human = self.tokeniser_length + 3
        self.end_of_human = self.tokeniser_length + 4
        self.start_of_ai = self.tokeniser_length + 5
        self.end_of_ai = self.tokeniser_length + 6
        self.pad_token = self.tokeniser_length + 7
        self.audio_tokens_start = self.tokeniser_length + 10
        self.codebook_size = 4032

        # Load nanocodec-mlx
        try:
            from nanocodec_mlx.models.audio_codec import AudioCodecModel
            self.codec = AudioCodecModel.from_pretrained(
                "nineninesix/nemo-nano-codec-22khz-0.6kbps-12.5fps-MLX"
            )
        except ImportError:
            raise ImportError(
                "nanocodec-mlx is required for MLX audio decoding. "
                "Install it with: pip install nanocodec-mlx"
            )

    def output_validation(self, out_ids: mx.array) -> None:
        """Validate that output contains required speech markers."""
        ids_list = out_ids.tolist()
        if self.start_of_speech not in ids_list or self.end_of_speech not in ids_list:
            raise ValueError("Special speech tokens not found in output!")

    def get_nano_codes(self, out_ids: mx.array) -> Tuple[mx.array, mx.array]:
        """Extract and process audio codes from model output."""
        ids_list = out_ids.tolist()
        start_idx = ids_list.index(self.start_of_speech)
        end_idx = ids_list.index(self.end_of_speech)

        if start_idx >= end_idx:
            raise ValueError("Invalid audio codes sequence!")

        audio_codes = out_ids[start_idx + 1: end_idx]
        if len(audio_codes) % 4:
            raise ValueError("Audio sequence length must be a multiple of 4!")

        audio_codes = audio_codes.reshape(-1, 4)
        offsets = mx.array([self.codebook_size * i for i in range(4)])
        audio_codes = audio_codes - offsets
        audio_codes = audio_codes - self.audio_tokens_start

        if mx.any(audio_codes < 0).item():
            raise ValueError("Invalid audio tokens!")

        # Transpose to [4, num_frames] and add batch dim -> [1, 4, num_frames]
        audio_codes = audio_codes.T[None, :]
        length = mx.array([audio_codes.shape[-1]])
        return audio_codes, length

    def get_waveform(self, out_ids: mx.array) -> np.ndarray:
        """Convert model output tokens to audio waveform."""
        out_ids = out_ids.reshape(-1)
        self.output_validation(out_ids)
        audio_codes, length = self.get_nano_codes(out_ids)

        reconstructed, _ = self.codec.decode(tokens=audio_codes, tokens_len=length)
        output_audio = np.array(reconstructed).squeeze()
        return output_audio


class MLXKaniModel:
    """MLX-based TTS model for Apple Silicon."""

    def __init__(self, config: MLXTTSConfig, model_path: str, player: MLXAudioPlayer):
        self.conf = config
        self.player = player

        # Load model config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            model_config = json.load(f)

        # Build model args
        self.args = KaniTTSModelArgs(
            model_type=model_config.get("model_type", "lfm2"),
            vocab_size=model_config["vocab_size"],
            hidden_size=model_config["hidden_size"],
            num_hidden_layers=model_config["num_hidden_layers"],
            num_attention_heads=model_config["num_attention_heads"],
            num_key_value_heads=model_config.get("num_key_value_heads", model_config["num_attention_heads"]),
            max_position_embeddings=model_config.get("max_position_embeddings", 4096),
            norm_eps=model_config.get("norm_eps", 1e-5),
            conv_bias=model_config.get("conv_bias", True),
            conv_L_cache=model_config.get("conv_L_cache", 4),
            block_dim=model_config.get("block_dim", model_config["hidden_size"]),
            block_ff_dim=model_config.get("block_ff_dim", model_config["hidden_size"] * 4),
            block_multiple_of=model_config.get("block_multiple_of", 256),
            block_ffn_dim_multiplier=model_config.get("block_ffn_dim_multiplier", 1.0),
            block_auto_adjust_ff_dim=model_config.get("block_auto_adjust_ff_dim", True),
            rope_theta=model_config.get("rope_theta", 10000.0),
            layer_types=model_config.get("layer_types"),
            # KaniTTS-2 specific
            audio_tokens_start=player.audio_tokens_start,
            tokens_per_frame=config.tokens_per_frame or model_config.get("tokens_per_frame", 4),
            audio_step=config.audio_step or model_config.get("audio_step", 1.0),
            use_learnable_rope=config.use_learnable_rope if config.use_learnable_rope is not None
                else model_config.get("use_learnable_rope", False),
            alpha_min=config.alpha_min or model_config.get("alpha_min", 0.1),
            alpha_max=config.alpha_max or model_config.get("alpha_max", 2.0),
            speaker_emb_dim=config.speaker_emb_dim or model_config.get("speaker_emb_dim", 128),
        )

        # Create model
        self.model = KaniTTS2MLXModel(self.args)

        # Load weights
        weights_path = os.path.join(model_path, "model.safetensors")
        weights = mx.load(weights_path)
        weights = self.model.sanitize(weights)
        self.model.load_weights(list(weights.items()), strict=False)
        mx.eval(self.model.parameters())

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Language settings
        self.language_settings = model_config.get("language_settings")
        self.status = "no_language_tags"
        self.language_tags_list = []
        if self.language_settings is not None:
            self.status = self.language_settings.get("status", "no_language_tags")
            self.language_tags_list = self.language_settings.get("language_tags_list", [])

    def get_input_ids(self, text: str, language_tag: Optional[str] = None) -> mx.array:
        """Prepare input tokens with special markers."""
        if language_tag is not None:
            text = f"{language_tag.strip()}: {text}"

        input_ids = self.tokenizer(text, return_tensors="np").input_ids
        input_ids = mx.array(input_ids)

        start_token = mx.array([[self.player.start_of_human]])
        end_tokens = mx.array([[self.player.end_of_text, self.player.end_of_human]])
        modified_ids = mx.concatenate([start_token, input_ids, end_tokens], axis=1)
        return modified_ids

    def run_model(
        self,
        text: str,
        language_tag: Optional[str] = None,
        speaker_emb: Optional[mx.array] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> Tuple[np.ndarray, str]:
        """Generate audio from text using MLX."""
        if self.status == "available_language_tags" and language_tag is None:
            print("=" * 40)
            print("!!! YOU NEED TO SELECT THE LANGUAGE TAG !!!")
            print("Languages available:")
            print(*self.language_tags_list, sep="\n")
            print("=" * 40)
        elif self.status == "no_language_tags" and language_tag is not None:
            print("=" * 40)
            print("!!! This model does not support language tag selection !!!")
            print("=" * 40)

        input_ids = self.get_input_ids(text, language_tag)

        output_ids = generate(
            model=self.model,
            input_ids=input_ids,
            speaker_emb=speaker_emb,
            max_new_tokens=self.conf.max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.player.end_of_speech,
        )

        audio = self.player.get_waveform(output_ids)
        return audio, text
