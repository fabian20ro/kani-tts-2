<div align="center">
    <img width="1728" height="862" alt="logo" src="https://github.com/user-attachments/assets/856eae00-29b2-4f65-a934-57d2677676f0" />
      <br><br>

  [![](https://dcbadge.limes.pink/api/server/https://discord.gg/NzP3rjB4SB?style=flat)](https://discord.gg/NzP3rjB4SB) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# KaniTTS2

**The Second Coming of the Kani** - A significantly improved text-to-speech library that pushes the boundaries of neural audio generation.
</div>



KaniTTS-2 is a research-grade TTS system built on causal language models with advanced architectural innovations. It's simple to use, but powerful under the hood.

## What's New in KaniTTS-2?

Major architectural improvements over the first release:

- **Speaker Embeddings**: True voice control through learned speaker representations. No more fine-tuning for each speaker - just clone any voice with a reference audio sample!
- **Learnable RoPE Theta**: Per-layer frequency scaling for better position encoding across the model depth
- **Frame-Level Position Encoding**: Precise temporal control with configurable audio frame positioning
- **Language Tag Support**: Multi-lingual and multi-accent support through language identifiers (when model is trained with tags)
- **Extended Generation**: Up to 40 seconds of continuous high-quality audio generation
- **Flexible Sampling**: Temperature, top-p, and repetition penalty moved to generation-time for easier experimentation



## Installation

```bash
pip install kani-tts-2
pip install -U "transformers==4.56.0"
```


## Quick Start

```python
from kani_tts import KaniTTS

# Initialize model
model = KaniTTS('nineninesix/your-model-name-here')

# Generate speech (simple)
audio, text = model("Hello, world!")

# Save to file (requires soundfile)
model.save_audio(audio, "output.wav")
```

That's it! Three lines for high-quality TTS. 🎉

## Advanced Usage

### Voice Cloning with Speaker Embeddings

KaniTTS-2 introduces **speaker embeddings** for true voice control. Extract a speaker's voice characteristics from a reference audio sample and use it to generate speech in that voice!

```python
from kani_tts import KaniTTS
from kani_tts import SpeakerEmbedder

# Initialize TTS model
model = KaniTTS('nineninesix/your-model-name')

# Initialize speaker embedder
embedder = SpeakerEmbedder()

# Extract speaker embedding from reference audio (any sample rate supported)
speaker_embedding = embedder.embed_audio_file("reference_voice.wav")  # Returns [1, 128] tensor

# Generate speech with that voice
audio, text = model(
    "This is a cloned voice speaking!",
    speaker_emb=speaker_embedding
)
model.save_audio(audio, "cloned_voice.wav")
```

#### How Speaker Embeddings Work

The speaker embedder uses a **WavLM-based model** trained to extract speaker characteristics:

1. **Input**: Audio at any sample rate (3-30 seconds recommended, automatically resampled to 16kHz)
2. **Processing**:
   - Automatic resampling to 16kHz if needed
   - Mean-Variance Normalization (MVN) on input audio
   - WavLM encoder extracts temporal features
   - Stats pooling (mean + std) aggregates features across time
   - Projection layers compress to 128-dimensional space
   - L2 normalization for consistent magnitude
3. **Output**: 128-dim L2-normalized speaker embedding ready for TTS

```python
from kani_tts import SpeakerEmbedder

embedder = SpeakerEmbedder(
    model_name="nineninesix/speaker-emb-tbr",  # Default WavLM model
    device="cuda",  # or "cpu"
    max_duration_sec=30.0  # Max audio length (longer will be truncated)
)

# From audio file (any sample rate, automatically resampled)
embedding = embedder.embed_audio_file("voice.wav")

# From numpy array (specify sample rate for automatic resampling)
import numpy as np
audio_array = np.random.randn(16000 * 5)  # 5 seconds
embedding = embedder.embed_audio(audio_array, sample_rate=16000)

# Save embedding for later use
import torch
torch.save(embedding, "my_voice.pt")

# Load and use saved embedding
audio, text = model("Hello!", speaker_emb="my_voice.pt")
```

> **Pro tip**: Longer reference audio (10-20 seconds) generally produces better embeddings. Audio at any sample rate is supported (automatic resampling). Make sure the audio is clean and contains only the target speaker! See [Voice Cloning Best Practices](#voice-cloning-best-practices-) for more details.

### Language Tag Support

**Some models** are trained with language/accent tags for better multi-lingual control:

```python
from kani_tts import KaniTTS

model = KaniTTS('nineninesix/your-multilingual-model')

# Check if model supports language tags
print(f"Status: {model.status}")  # 'available_language_tags' or 'no_language_tags'

# Show available language tags
model.show_language_tags()
# Output:
# ==================================================
# Available language tags:
# --------------------------------------------------
#   1. en_US
#   2. fr_FR
#   3. de_DE
# ==================================================

# Generate with specific language tag
audio, text = model(
    "Bonjour le monde!",
    language_tag="fr_FR",
    speaker_emb=speaker_embedding
)
```

> **Note**: Language tags are particularly useful for controlling accents when your model was trained with accent labels. Check model metadata to see if tags are available.

### Controlling Generation Parameters

KaniTTS-2 moves sampling parameters to **generation time** for easier experimentation:

```python
from kani_tts import KaniTTS

# Initialize model (basic config only)
model = KaniTTS(
    'nineninesix/your-model-name',
    max_new_tokens=3000,       # Max generation length (default: 3000)
    suppress_logs=True,        # Suppress library logs (default: True)
    show_info=True,            # Show model info on init (default: True)
)

# Control sampling at generation time
audio, text = model(
    "Your text here",
    temperature=0.7,           # Lower = more deterministic (default: 1.0)
    top_p=0.9,                 # Nucleus sampling threshold (default: 0.95)
    repetition_penalty=1.2,    # Penalize repetition (default: 1.1)
    speaker_emb=speaker_emb,   # Optional: speaker embedding
    language_tag="en_US"       # Optional: language tag
)
```

**Why move to generation time?** This lets you:
- Quickly experiment with different sampling strategies
- Use the same loaded model with different generation configs
- Change voice and language per-generation without reloading

When initialized, KaniTTS-2 displays a beautiful banner with model information:
```
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║                   N I N E N I N E S I X  😼                ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝

              /\_/\
             ( o.o )
              > ^ <

──────────────────────────────────────────────────────────────
  Model: nineninesix/kani-tts-2
  Device: GPU (CUDA)
  Mode: Available language tags (3 language tags)
  Tags: en_US, fr_FR, de_DE

  Configuration:
    • Sample Rate: 22050 Hz
    • Max Tokens: 3000
    • Speaker Embedding Dim: 128
    • Text Vocab Size: 64400
    • Tokens per Frame: 4
    • Audio Step: 0.25
    • Learnable RoPE: Enabled (per-layer frequency scaling)
    • Alpha Range: [0.5, 2.0]
──────────────────────────────────────────────────────────────

  Ready to generate speech! 🎵
```

You can disable this banner by setting `show_info=False`, or show it again anytime with `model.show_model_info()`.

### Controlling Logging Output

By default, Kani-TTS suppresses all logging output from transformers, NeMo, and PyTorch to keep your console clean. Only your `print()` statements will be visible.

```python
from kani_tts import KaniTTS

# Default behavior - logs are suppressed
model = KaniTTS('your-model-name')

# To see all library logs (for debugging)
model = KaniTTS('your-model-name', suppress_logs=False)

# You can also manually suppress logs at any time
from kani_tts import suppress_all_logs
suppress_all_logs()
```

### Working with Audio Output

The generated audio is a NumPy array sampled at 22kHz:

```python
import numpy as np
import soundfile as sf

audio, text = model("Generate speech from this text")

# Audio is a numpy array
print(audio.shape)  # (num_samples,)
print(audio.dtype)  # float32/float64

# Save using soundfile
sf.write('output.wav', audio, 22050)

# Or use the built-in method
model.save_audio(audio, 'output.wav', sample_rate=22050)
```



### Playing Audio in Jupyter Notebooks

You can listen to generated audio directly in Jupyter notebooks or IPython:

```python
from kani_tts import KaniTTS
from IPython.display import Audio as aplay

model = KaniTTS('nineninesix/your-model-name')
audio, text = model("Hello, world!")

# Play audio in notebook
aplay(audio, rate=model.sample_rate)
```

## API Reference

### Main Classes

#### `KaniTTS(model_name, **kwargs)`
Main TTS interface.

**Parameters:**
- `model_name` (str): HuggingFace model ID or local path
- `max_new_tokens` (int): Max generation length (default: 3000)
- `device_map` (str): Device mapping for model (default: "auto")
- `suppress_logs` (bool): Suppress library logs (default: True)
- `show_info` (bool): Display model info banner (default: True)
- Architecture params: `text_vocab_size`, `tokens_per_frame`, `audio_step`, `use_learnable_rope`, `alpha_min`, `alpha_max`, `speaker_emb_dim` (all optional, read from model config if None)

**Methods:**
- `model(text, language_tag=None, speaker_emb=None, temperature=1.0, top_p=0.95, repetition_penalty=1.1)` → `(audio, text)`
- `model.generate(...)` → Same as `__call__`
- `model.save_audio(audio, path)` → Save audio to file
- `model.show_model_info()` → Display model banner
- `model.show_language_tags()` → Display available language tags (if supported)
- `model.load_speaker_embedding(path)` → Load speaker embedding from .pt file

#### `SpeakerEmbedder(model_name, device, max_duration_sec)`
Extract speaker embeddings from audio.

**Parameters:**
- `model_name` (str): HuggingFace model ID (default: "nineninesix/speaker-emb-tbr")
- `device` (str): "cuda" or "cpu" (default: auto-detect)
- `max_duration_sec` (float): Max audio length in seconds (default: 30.0)

**Methods:**
- `embedder.embed_audio(audio, sample_rate=16000)` → `[1, 128]` tensor
- `embedder.embed_audio_file(path)` → `[1, 128]` tensor

**Convenience function:**
```python
from kani_tts import compute_speaker_embedding

embedding = compute_speaker_embedding(audio_or_path, sample_rate=16000)
```

## Complete Example: Voice Cloning Pipeline

Here's a complete example showing how to clone a voice and generate speech:

```python
from kani_tts import KaniTTS
from kani_tts import SpeakerEmbedder
import soundfile as sf

# 1. Initialize models
print("Loading TTS model...")
tts = KaniTTS('nineninesix/kani-tts-2-model')

print("Loading speaker embedder...")
embedder = SpeakerEmbedder()

# 2. Extract speaker embedding from reference audio (any sample rate)
print("Extracting speaker characteristics...")
speaker_emb = embedder.embed_audio_file("reference_speaker.wav")

# Save embedding for later reuse
import torch
torch.save(speaker_emb, "my_cloned_voice.pt")

# 3. Generate speech with cloned voice
print("Generating speech...")
audio, text = tts(
    "This is a test of voice cloning with KaniTTS-2. Pretty cool, right?",
    speaker_emb=speaker_emb,
    temperature=0.8,      # Slightly less random
    top_p=0.92,           # Nucleus sampling
    repetition_penalty=1.15  # Avoid repetition
)

# 4. Save output
tts.save_audio(audio, "cloned_output.wav")
print(f"✅ Generated {len(audio)/tts.sample_rate:.2f}s of audio")

# 5. For multi-lingual models, specify language
if tts.status == 'available_language_tags':
    tts.show_language_tags()
    audio_fr, _ = tts(
        "Bonjour, comment allez-vous?",
        language_tag="fr_FR",
        speaker_emb=speaker_emb
    )
    tts.save_audio(audio_fr, "french_cloned.wav")
```

## Architecture

### The Big Picture 🏗️

KaniTTS-2 is based on a **causal language model** architecture with specialized modifications for high-quality audio generation. Think of it as GPT, but instead of predicting the next word, it predicts the next audio token sequence.

**Two-Stage Pipeline:**

1. **Text → Audio Tokens**: A modified LLaMA-based causal LM generates discrete audio token sequences from text input
2. **Audio Tokens → Waveform**: NVIDIA NeMo's NanoCodec neural vocoder decodes tokens into continuous audio waveforms (22kHz, 12.5fps)

### Key Innovations in KaniTTS-2

#### 1. Learnable RoPE Theta (Per-Layer Frequency Scaling)
Standard RoPE (Rotary Position Embeddings) uses fixed frequencies for position encoding. KaniTTS-2 introduces **per-layer learnable alpha** parameters that scale RoPE frequencies:

- Each transformer layer learns its own `alpha` value in range `[alpha_min, alpha_max]`
- This allows different layers to focus on different temporal scales
- Better handling of long-range dependencies in audio sequences

#### 2. Frame-Level Position Encoding
Audio tokens are organized in **frames** (4 tokens per frame, representing 4 codebook channels):

- `tokens_per_frame`: Number of tokens in each audio frame (default: 4)
- `audio_step`: Position increment per frame (e.g., 0.25 means each frame advances position by 0.25)
- Text tokens use standard position encoding (1 step per token)
- Audio tokens use frame-based positioning for better temporal alignment

This dual encoding scheme helps the model understand the difference between text tokens (discrete linguistic units) and audio tokens (continuous temporal frames).

#### 3. Speaker Embeddings
Instead of discrete speaker IDs (which require fine-tuning), KaniTTS-2 uses **continuous speaker embeddings**:

- 128-dimensional learned representations injected into the model
- Extracted from reference audio using WavLM-based encoder
- Enables zero-shot voice cloning without retraining
- Conditions the entire generation process on speaker characteristics

#### 4. Language/Accent Tags
Optional language identifiers prepended to text input:
- Format: `<language_tag>: <text>` (e.g., `"en_US: Hello world"`)
- Helps model disambiguate accents and pronunciation
- Particularly useful for multi-lingual models

### Token Structure

The model uses an **extended vocabulary** with special control tokens:

**Text Tokens** (0 - 64399):
- Standard text vocabulary from tokenizer
- Special markers: `<start_of_text>` (1), `<end_of_text>` (2)

**Control Tokens** (64400+):
- `<start_of_speech>`, `<end_of_speech>`: Speech boundaries
- `<start_of_human>`, `<end_of_human>`: Human turn markers
- `<start_of_ai>`, `<end_of_ai>`: AI turn markers
- `<pad>`: Padding token

**Audio Tokens** (64410+):
- 4 codebook channels × 4032 codes per channel = 16,128 audio tokens
- Organized as frames: `[c0, c1, c2, c3]` where each `ci` is from codebook `i`
- Encoded using NVIDIA NeMo NanoCodec (22kHz, 0.6kbps, 12.5fps)

### Generation Process

```
Input text + optional (language_tag, speaker_emb)
         ↓
   Tokenization + special tokens
         ↓
   LLaMA-based causal LM with:
   - Learnable RoPE (per-layer alpha)
   - Frame-level position encoding
   - Speaker embedding conditioning
         ↓
   Audio token sequence (4 tokens per frame)
         ↓
   NeMo NanoCodec decoder
         ↓
   22kHz waveform output
```

## Requirements

- Python 3.10 or higher
- CUDA-capable GPU (recommended), Apple Silicon Mac (via MPS or MLX), or CPU
- PyTorch 2.0 or higher
- Transformers 4.57.1+ (for LLaMA-based models)
- NeMo Toolkit (for audio codec)
- soundfile (for saving audio)
- torchaudio (optional, for speaker embedding extraction from audio files)
- For MLX backend: `pip install kani-tts-2[mlx]` (macOS Apple Silicon only)

## Model Compatibility

KaniTTS-2 works with **modified LLaMA-based causal language models** (LFM2) trained for TTS with:

✅ **Required characteristics:**
- Extended vocabulary (text tokens + audio tokens + control tokens)
- Special tokens for speech/text/turn boundaries
- Compatible with NeMo NanoCodec (22kHz, 0.6kbps, 12.5fps, 4 codebooks)

✅ **Optional features** (configured via model metadata or init params):
- Speaker embedding support (`speaker_emb_dim` in config)
- Learnable RoPE theta (`use_learnable_rope`, `alpha_min`, `alpha_max`)
- Frame-level position encoding (`tokens_per_frame`, `audio_step`)
- Language tag support (`language_settings` in config)

**How to check model compatibility:**
```python
model = KaniTTS('model-name', show_info=True)
# The banner will display all supported features!
```



## Tips & Best Practices 💡

### Getting the Best Results

**For voice cloning:**
- Use 10-20 seconds of clean reference audio
- Any sample rate supported (automatic resampling to 16kHz)
- Choose audio with minimal background noise
- The reference speaker should be speaking clearly
- See [Voice Cloning Best Practices](#voice-cloning-best-practices-) for detailed recommendations

**For generation quality:**
- Start with default sampling parameters (temperature=1.0, top_p=0.95)
- Lower temperature (0.7-0.9) for more consistent output
- Increase repetition_penalty (1.1-1.3) if you hear loops
- Experiment with `max_new_tokens` for longer generations (up to ~3000 for ~40s)

**For multi-lingual models:**
- Always check `model.show_language_tags()` to see available tags
- Use language tags for better accent control
- Language tags are especially important for disambiguating homophones


### Voice Cloning Best Practices 🎯

For optimal voice cloning results, follow these critical recommendations:

**1. Reference Audio Quality is Critical**

The quality of your reference audio directly impacts model behavior and output quality:
- ✅ Use clean recordings without background noise or audio artifacts
- ✅ Ensure proper audio levels (not too quiet, not clipping)
- ✅ Choose clear speech samples without music, effects, or other speakers
- ✅ Any sample rate supported (automatic resampling to 16kHz)
- ❌ Avoid noisy, compressed, or low-quality recordings
- ❌ Avoid recordings with echo, reverb, or audio processing artifacts

**Poor reference quality** → Model confusion, inconsistent voice characteristics, artifacts in output
**Good reference quality** → Stable voice reproduction, natural-sounding speech, better prosody

**2. Multiple Audio Samples → Better Speaker Representation**

To capture a speaker's voice characteristics more accurately:

```python
from kani_tts import SpeakerEmbedder
import torch

embedder = SpeakerEmbedder()

# Record 5-10 different audio samples of the same speaker
# (different sentences, varied intonation and speaking styles)
sample_files = [
    "speaker_sample_1.wav",
    "speaker_sample_2.wav",
    "speaker_sample_3.wav",
    "speaker_sample_4.wav",
    "speaker_sample_5.wav",
]

# Extract embeddings from all samples
embeddings = [embedder.embed_audio_file(f) for f in sample_files]

# Average the embeddings to get a more generalized representation
averaged_embedding = torch.stack(embeddings).mean(dim=0)

# Use the averaged embedding for generation
audio, text = model(
    "Your text here",
    speaker_emb=averaged_embedding
)
```

**Why averaging helps:**
- **More robust**: Captures general speaker characteristics rather than peculiarities of one recording
- **Better generalization**: Reduces sensitivity to recording conditions or speaking style of individual samples
- **Consistent quality**: Produces more stable and natural voice across different texts

**Recommendation**: Record **5-10 different audio samples** (15-25 seconds each) with varied content and speaking styles, then average their embeddings for best results.

## Performance Notes

- **GPU**: ~2-5s for 10s of audio (depending on model size and GPU)
- **CPU**: ~20-60s for 10s of audio (not recommended for production)
- **Memory**: ~4-8GB VRAM for inference (bfloat16), ~2-16GB for model loading depending on size

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Citation

```bibtex

@article{liquidai2025lfm2,
 title={LFM2 Technical Report},
 author={Liquid AI},
 journal={arXiv preprint arXiv:2511.23404},
 year={2025}
}

@inproceedings{emilialarge,
  author={He, Haorui and Shang, Zengqiang and Wang, Chaoren and Li, Xuyuan and Gu, Yicheng and Hua, Hua and Liu, Liwei and Yang, Chen and Li, Jiaqi and Shi, Peiyang and Wang, Yuancheng and Chen, Kai and Zhang, Pengyuan and Wu, Zhizheng},
  title={Emilia: A Large-Scale, Extensive, Multilingual, and Diverse Dataset for Speech Generation},
  booktitle={arXiv:2501.15907},
  year={2025}
}

@article{emonet_voice_2025,
  author={Schuhmann, Christoph and Kaczmarczyk, Robert and Rabby, Gollam and Friedrich, Felix and Kraus, Maurice and Nadi, Kourosh and Nguyen, Huu and Kersting, Kristian and Auer, Sören},
  title={EmoNet-Voice: A Fine-Grained, Expert-Verified Benchmark for Speech Emotion Detection},
  journal={arXiv preprint arXiv:2506.09827},
  year={2025}
}

@inproceedings{gengembre24_interspeech,
  title     = {Disentangling prosody and timbre embeddings via voice conversion},
  author    = {Nicolas Gengembre and Olivier {Le Blouch} and Cédric Gendrot},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {2765--2769},
  doi       = {10.21437/Interspeech.2024-207},
  issn      = {2958-1796},
}
```

## Responsible Use

**Prohibited activities include:**
- Illegal content or harmful, threatening, defamatory, or obscene material
- Hate speech, harassment, or incitement of violence
- Generating false or misleading information
- Impersonating individuals without consent
- Malicious activities such as spamming, phishing, or fraud

By using this model, you agree to comply with these restrictions and all applicable laws.

## Resources

**Models:** [Pretrained Model](https://huggingface.co/nineninesix/kani-tts-2-pt), [English Model](https://huggingface.co/nineninesix/kani-tts-2-en)

**Pretraining Framework**
Train your own TTS model on your language or accent from scratch using this open-source pretraining framework: [KaniTTS2-Pretrain](https://github.com/nineninesix-ai/kani-tts-2-pretrain).

**Example Dataset**: https://huggingface.co/datasets/nineninesix/kanitts2-es-nano-codec-speaker-emb-dataset


The training code is under active development and will continue to receive updates and improvements.

If you use this code in your research, please cite:

```bibtex
@software{kani_tts_2,
  author = {Nineninesix},
  title = {KaniTTS2: Text-to-Speech Model with Frame-level Position Encoding},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://github.com/nineninesix-ai/kani-tts-2}},
  note = {Open-source TTS model}
}
```
## vLLM Integration

Coming soon.

## macOS / Apple Silicon (MLX)

KaniTTS-2 supports native inference on Apple Silicon Macs via [MLX](https://github.com/ml-explore/mlx).

### Quick Start (PyTorch MPS)

The PyTorch backend now auto-detects Apple Silicon and uses the MPS device:

```python
from kani_tts import KaniTTS

# Automatically uses MPS on Apple Silicon (falls back to CPU if unavailable)
model = KaniTTS('nineninesix/your-model-name')
audio, text = model("Hello from macOS!")
```

### MLX Backend (Recommended for Apple Silicon)

For best performance on Apple Silicon, use the MLX backend:

```bash
# Install MLX dependencies
pip install kani-tts-2[mlx]

# Convert weights to MLX format
python scripts/convert_to_mlx.py \
    --model nineninesix/kani-tts-2-en \
    --output ./kani-tts-2-en-mlx
```

```python
from kani_tts import KaniTTSMLX

model = KaniTTSMLX("./kani-tts-2-en-mlx")
audio, text = model("Hello from Apple Silicon!")
model.save_audio(audio, "output.wav")

# Voice cloning works the same way
audio, text = model("Cloned voice!", speaker_emb="my_voice.npy")
```

**Requirements**: macOS with Apple Silicon (M1/M2/M3/M4), Python 3.10+

**Memory**: The full pipeline uses ~1.6GB. A Mac with 16GB+ unified memory is sufficient.

## Tech Report

Coming soon.

## Contact
Have a question, feedback, or need support? Please fill out our [contact form](https://airtable.com/appX2G2TpoRk4M5Bf/pagO2xbIOjiwulPcP/form) and we'll get back to you as soon as possible.

---

**Made with 😼 by nineninesix**
