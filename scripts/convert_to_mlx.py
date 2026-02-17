"""
Convert KaniTTS-2 PyTorch safetensors weights to MLX format.

Handles:
- Standard LFM2 weights (1:1 name mapping, conv weight transpose)
- KaniTTS-2 custom weights (speaker_emb_projection, learnable_rope_layers)
- Weight tying (lm_head -> embed_tokens)

Usage:
    python scripts/convert_to_mlx.py \
        --model nineninesix/kani-tts-2-en \
        --output ./kani-tts-2-en-mlx
"""

import argparse
import json
import os
import shutil

import numpy as np


def convert_weights(model_path: str, output_path: str, dtype: str = "float16"):
    """Convert KaniTTS-2 safetensors to MLX-compatible format."""
    try:
        import mlx.core as mx
    except ImportError:
        raise ImportError("mlx is required: pip install mlx")

    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download, snapshot_download

    # Resolve model path
    if os.path.isdir(model_path):
        local_path = model_path
    else:
        print(f"Downloading model from {model_path}...")
        local_path = snapshot_download(repo_id=model_path)

    # Load safetensors
    safetensors_path = os.path.join(local_path, "model.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"model.safetensors not found in {local_path}")

    print(f"Loading weights from {safetensors_path}...")
    torch_weights = load_file(safetensors_path)

    # Load config
    config_path = os.path.join(local_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Map dtype
    dtype_map = {"float16": np.float16, "float32": np.float32, "bfloat16": np.float16}
    np_dtype = dtype_map.get(dtype, np.float16)

    # Convert weights
    mlx_weights = {}
    skipped = []
    for name, tensor in torch_weights.items():
        np_array = tensor.cpu().numpy()

        # Skip rotary_emb buffers -- mlx-lm computes RoPE from config
        if "rotary_emb" in name or "pos_emb" in name:
            # But keep learnable_rope_layers which are trained parameters
            if "learnable_rope_layers" not in name:
                skipped.append(name)
                continue

        # Conv weight transpose: PyTorch (out, in/groups, kernel) -> MLX (out, kernel, in/groups)
        if "conv.weight" in name and np_array.ndim == 3:
            if np_array.shape[-1] > np_array.shape[1]:
                np_array = np_array.transpose(0, 2, 1)

        # Drop lm_head.weight if tied (mlx-lm uses embed_tokens.as_linear)
        if name == "lm_head.weight":
            if "model.embed_tokens.weight" in torch_weights:
                skipped.append(f"{name} (tied to embed_tokens)")
                continue

        # Convert dtype
        if np_array.dtype in (np.float32, np.float64):
            np_array = np_array.astype(np_dtype)

        mlx_weights[name] = mx.array(np_array)

    if skipped:
        print(f"Skipped {len(skipped)} weights:")
        for s in skipped:
            print(f"  - {s}")

    # Save
    os.makedirs(output_path, exist_ok=True)

    # Save weights
    weights_path = os.path.join(output_path, "model.safetensors")
    print(f"Saving {len(mlx_weights)} weight tensors to {weights_path}...")
    mx.save_safetensors(weights_path, mlx_weights)

    # Copy config and add KaniTTS-2 metadata
    mlx_config = dict(config)
    mlx_config["model_type"] = "lfm2"  # mlx-lm model type
    mlx_config["mlx_backend"] = True
    config_out = os.path.join(output_path, "config.json")
    with open(config_out, "w") as f:
        json.dump(mlx_config, f, indent=2)

    # Copy tokenizer files
    for filename in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "tokenizer.model", "vocab.json", "merges.txt"]:
        src = os.path.join(local_path, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, filename))

    # Summary
    total_params = sum(v.size for v in mlx_weights.values())
    size_mb = sum(v.nbytes for v in mlx_weights.values()) / 1024 / 1024
    print(f"\nConversion complete:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Size: {size_mb:.1f} MB ({dtype})")
    print(f"  Output: {output_path}")

    # List custom KaniTTS-2 weights
    custom = [n for n in mlx_weights if "speaker_emb" in n or "learnable_rope" in n]
    if custom:
        print(f"\n  KaniTTS-2 custom weights ({len(custom)}):")
        for n in custom:
            print(f"    - {n}: {mlx_weights[n].shape}")


def main():
    parser = argparse.ArgumentParser(description="Convert KaniTTS-2 weights to MLX format")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--output", required=True, help="Output directory for MLX weights")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                        help="Target dtype (default: float16)")
    args = parser.parse_args()

    convert_weights(args.model, args.output, args.dtype)


if __name__ == "__main__":
    main()
