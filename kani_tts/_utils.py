"""Shared utility functions for KaniTTS-2 backends."""

import logging
import warnings

import numpy as np


def suppress_all_logs():
    """
    Suppress all logging output from transformers, NeMo, PyTorch, and other libraries.
    Only print() statements from user code will be visible.
    """
    warnings.filterwarnings('ignore')

    try:
        import transformers
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
    except ImportError:
        pass

    for logger_name in ["nemo", "nemo_logger", "torch", "pytorch",
                         "transformers", "numba", "matplotlib", "PIL"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)


def save_audio(audio: np.ndarray, output_path: str, sample_rate: int):
    """
    Save audio waveform to file.

    Args:
        audio: Audio waveform as numpy array
        output_path: Path to save audio file (e.g., "output.wav")
        sample_rate: Audio sample rate in Hz
    """
    try:
        import soundfile as sf
        sf.write(output_path, audio, sample_rate)
    except ImportError:
        raise ImportError(
            "soundfile is required to save audio. "
            "Install it with: pip install soundfile"
        )


def show_language_tags(status: str, language_tags_list: list):
    """Display available language tags."""
    print("=" * 50)
    if status == "available_language_tags":
        print("Available language tags:")
        print("-" * 50)
        if language_tags_list:
            for i, tag in enumerate(language_tags_list, 1):
                print(f"  {i}. {tag}")
        else:
            print("  No tags configured")
    else:
        print("This model does not support language tag selection.")
    print("=" * 50)
